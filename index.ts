/**
 * supabase/functions/detect-objects/index.ts
 *
 * Satellite Object Detection API — Supabase Edge Function
 *
 * This is the commercial demo endpoint. It:
 *   1. Accepts a satellite image (URL or base64)
 *   2. Forwards it to your Python inference server (Hugging Face / RunPod)
 *   3. Returns structured detection results with confidence scores
 *   4. Logs the request to Supabase database for analytics
 *
 * Deployed globally at:
 *   https://[YOUR_PROJECT_ID].supabase.co/functions/v1/detect-objects
 *
 * Architecture:
 *   Client → Supabase Edge Function (auth, rate limiting, logging)
 *          → Python Inference Server (actual ML inference)
 *          → Response back to client
 *
 * Think of the Edge Function as a security desk at your building entrance —
 * it checks credentials, logs visitors, and routes them to the right room,
 * but the actual work happens inside (Python inference server).
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { getCorsHeaders } from "../_shared/cors.ts";

// ── Types ─────────────────────────────────────────────────────────────────────

interface DetectionRequest {
  image_url?: string;         // Public URL of satellite image
  image_base64?: string;      // Base64-encoded image (alternative to URL)
  confidence_threshold?: number; // Min confidence (default: 0.25)
  classes?: string[];         // Filter to specific classes (e.g., ["building", "vehicle"])
  return_image?: boolean;     // Return annotated image in response
  client_id?: string;         // For usage tracking / billing
}

interface Detection {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: {
    x_center: number;
    y_center: number;
    width: number;
    height: number;
  };
  bbox_pixels?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

interface DetectionResponse {
  success: boolean;
  request_id: string;
  model_version: string;
  processing_time_ms: number;
  image_width?: number;
  image_height?: number;
  detections: Detection[];
  summary: {
    total_objects: number;
    by_class: Record<string, number>;
    high_confidence: number;   // Detections above 0.7
    low_confidence: number;    // Detections between threshold and 0.7
  };
  annotated_image_base64?: string;  // Fixed: inference server returns base64, not a URL
  error?: string;
}

// ── Config ────────────────────────────────────────────────────────────────────

const INFERENCE_SERVER_URL = Deno.env.get("INFERENCE_SERVER_URL") ?? "";
const INFERENCE_API_KEY = Deno.env.get("INFERENCE_API_KEY") ?? "";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

const DEFAULT_CONFIDENCE = 0.25;
const MAX_IMAGE_SIZE_MB = 20;
const RATE_LIMIT_REQUESTS = 100;  // Per hour per client

// ── Helpers ───────────────────────────────────────────────────────────────────

function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function buildSummary(detections: Detection[], threshold: number) {
  const by_class: Record<string, number> = {};
  let high_confidence = 0;
  let low_confidence = 0;

  for (const det of detections) {
    by_class[det.class_name] = (by_class[det.class_name] ?? 0) + 1;
    if (det.confidence >= 0.7) high_confidence++;
    else low_confidence++;
  }

  return {
    total_objects: detections.length,
    by_class,
    high_confidence,
    low_confidence,
  };
}

async function logRequest(
  supabase: ReturnType<typeof createClient>,
  requestId: string,
  clientId: string | undefined,
  processingMs: number,
  nDetections: number,
  success: boolean,
  errorMsg?: string
) {
  try {
    await supabase.from("api_requests").insert({
      request_id: requestId,
      client_id: clientId ?? "anonymous",
      processing_time_ms: processingMs,
      n_detections: nDetections,
      success,
      error_message: errorMsg ?? null,
    });
  } catch (_e) {
    // Non-critical — don't fail the request if logging fails
    console.warn("Failed to log request:", _e);
  }
}

async function checkRateLimit(
  supabase: ReturnType<typeof createClient>,
  clientId: string
): Promise<boolean> {
  try {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const { count } = await supabase
      .from("api_requests")
      .select("*", { count: "exact", head: true })
      .eq("client_id", clientId)
      .gte("created_at", oneHourAgo);

    return (count ?? 0) < RATE_LIMIT_REQUESTS;
  } catch (_e) {
    return true; // Allow request if rate limit check fails
  }
}

// ── Main Handler ──────────────────────────────────────────────────────────────

Deno.serve(async (req: Request) => {
  // Handle CORS preflight
  const cors = getCorsHeaders(req);
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: cors });
  }

  const requestId = generateRequestId();
  const startTime = Date.now();

  // Initialize Supabase client for logging
  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  try {
    // ── Auth Check ─────────────────────────────────────────────────────────
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      return new Response(
        JSON.stringify({ success: false, error: "Missing Authorization header" }),
        { status: 401, headers: { ...cors, "Content-Type": "application/json" } }
      );
    }

    // Validate JWT with Supabase
    const { data: { user }, error: authError } = await supabase.auth.getUser(
      authHeader.replace("Bearer ", "")
    );

    if (authError || !user) {
      return new Response(
        JSON.stringify({ success: false, error: "Invalid or expired token" }),
        { status: 401, headers: { ...cors, "Content-Type": "application/json" } }
      );
    }

    // ── Parse Request ──────────────────────────────────────────────────────
    const body: DetectionRequest = await req.json();

    if (!body.image_url && !body.image_base64) {
      return new Response(
        JSON.stringify({ success: false, error: "Provide either image_url or image_base64" }),
        { status: 400, headers: { ...cors, "Content-Type": "application/json" } }
      );
    }

    const clientId = body.client_id ?? user.id;
    const confidenceThreshold = body.confidence_threshold ?? DEFAULT_CONFIDENCE;

    // ── Rate Limiting ──────────────────────────────────────────────────────
    const withinLimit = await checkRateLimit(supabase, clientId);
    if (!withinLimit) {
      return new Response(
        JSON.stringify({
          success: false,
          error: `Rate limit exceeded: ${RATE_LIMIT_REQUESTS} requests/hour. Contact us to upgrade.`,
        }),
        { status: 429, headers: { ...cors, "Content-Type": "application/json" } }
      );
    }

    // ── Forward to Python Inference Server ─────────────────────────────────
    if (!INFERENCE_SERVER_URL) {
      // Demo mode: return mock detections when inference server not configured
      console.warn("INFERENCE_SERVER_URL not set — returning mock response");
      const mockDetections: Detection[] = [
        {
          class_name: "building",
          class_id: 0,
          confidence: 0.91,
          bbox: { x_center: 0.45, y_center: 0.32, width: 0.12, height: 0.09 },
        },
        {
          class_name: "building",
          class_id: 0,
          confidence: 0.87,
          bbox: { x_center: 0.71, y_center: 0.58, width: 0.08, height: 0.11 },
        },
        {
          class_name: "vehicle",
          class_id: 1,
          confidence: 0.73,
          bbox: { x_center: 0.33, y_center: 0.67, width: 0.02, height: 0.015 },
        },
      ];

      const processingMs = Date.now() - startTime;
      const response: DetectionResponse = {
        success: true,
        request_id: requestId,
        model_version: "yolov8n-satellite-v1-DEMO",
        processing_time_ms: processingMs,
        detections: mockDetections,
        summary: buildSummary(mockDetections, confidenceThreshold),
      };

      await logRequest(supabase, requestId, clientId, processingMs, mockDetections.length, true);
      return new Response(JSON.stringify(response), {
        headers: { ...cors, "Content-Type": "application/json" },
      });
    }

    // Forward to real inference server
    const inferenceRes = await fetch(`${INFERENCE_SERVER_URL}/detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": INFERENCE_API_KEY,
      },
      body: JSON.stringify({
        image_url: body.image_url,
        image_base64: body.image_base64,
        confidence_threshold: confidenceThreshold,
        classes: body.classes,
        return_image: body.return_image ?? false,
      }),
    });

    if (!inferenceRes.ok) {
      const errText = await inferenceRes.text();
      throw new Error(`Inference server error ${inferenceRes.status}: ${errText}`);
    }

    const inferenceData = await inferenceRes.json();
    const processingMs = Date.now() - startTime;

    // ── Build Response ─────────────────────────────────────────────────────
    const detections: Detection[] = inferenceData.detections ?? [];
    const response: DetectionResponse = {
      success: true,
      request_id: requestId,
      model_version: inferenceData.model_version ?? "yolov8n-satellite-v1",
      processing_time_ms: processingMs,
      image_width: inferenceData.image_width,
      image_height: inferenceData.image_height,
      detections,
      summary: buildSummary(detections, confidenceThreshold),
      annotated_image_base64: inferenceData.annotated_image_base64,  // Fixed: was annotated_image_url
    };

    await logRequest(supabase, requestId, clientId, processingMs, detections.length, true);

    return new Response(JSON.stringify(response), {
      headers: { ...cors, "Content-Type": "application/json" },
    });

  } catch (error) {
    const processingMs = Date.now() - startTime;
    const errMsg = error instanceof Error ? error.message : String(error);

    console.error(`[${requestId}] Error:`, errMsg);
    await logRequest(supabase, requestId, undefined, processingMs, 0, false, errMsg);

    return new Response(
      JSON.stringify({ success: false, request_id: requestId, error: errMsg }),
      { status: 500, headers: { ...cors, "Content-Type": "application/json" } }
    );
  }
});
