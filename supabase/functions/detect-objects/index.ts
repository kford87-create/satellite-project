/**
 * supabase/functions/detect-objects/index.ts
 *
 * Satellite Object Detection API — Supabase Edge Function
 *
 * Accepts:
 *   - address or lat/lng  → fetches live Google Maps satellite imagery, then detects
 *   - image_url or image_base64 → detects on provided image directly
 *
 * Architecture:
 *   Client → Supabase Edge Function (auth, rate limiting, logging)
 *          → [Optional] Nominatim geocoding (address → lat/lng)
 *          → [Optional] Google Maps Static API (lat/lng → satellite image)
 *          → Python Inference Server (YOLOv8 detection)
 *          → Response back to client
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { getCorsHeaders } from "../_shared/cors.ts";

// ── Types ─────────────────────────────────────────────────────────────────────

interface DetectionRequest {
  // Image input — provide one of these four:
  image_url?: string;
  image_base64?: string;
  address?: string;           // Street address or place name
  lat?: number;               // Latitude (use with lng)
  lng?: number;               // Longitude (use with lat)

  // Geo options (only used with address/lat/lng)
  zoom?: number;              // Map zoom level, default 18 (~0.5m/px)

  // Detection options
  confidence_threshold?: number;
  classes?: string[];
  return_image?: boolean;
  client_id?: string;
}

interface Location {
  lat: number;
  lng: number;
  address?: string;
  zoom: number;
  meters_per_pixel: number;
  coverage_m: number;         // Side length of the captured square in metres
}

interface Detection {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: { x_center: number; y_center: number; width: number; height: number };
  bbox_pixels?: { x1: number; y1: number; x2: number; y2: number };
}

interface DetectionResponse {
  success: boolean;
  request_id: string;
  model_version: string;
  processing_time_ms: number;
  imagery_source: "google_maps" | "url" | "upload";
  location?: Location;
  image_width?: number;
  image_height?: number;
  detections: Detection[];
  summary: {
    total_objects: number;
    by_class: Record<string, number>;
    high_confidence: number;
    low_confidence: number;
  };
  annotated_image_base64?: string;
  satellite_image_base64?: string;
  error?: string;
}

// ── Config ────────────────────────────────────────────────────────────────────

const INFERENCE_SERVER_URL    = Deno.env.get("INFERENCE_SERVER_URL") ?? "";
const INFERENCE_API_KEY       = Deno.env.get("INFERENCE_API_KEY") ?? "";
const SUPABASE_URL            = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const GOOGLE_MAPS_API_KEY     = Deno.env.get("GOOGLE_MAPS_API_KEY") ?? "";
const GOOGLE_GEOCODING_API_KEY = Deno.env.get("GOOGLE_GEOCODING_API_KEY") ?? "";

const DEFAULT_CONFIDENCE    = 0.25;
const MIN_CONFIDENCE        = 0.10;
const DEFAULT_ZOOM          = 18;
const RATE_LIMIT_REQUESTS   = 100;
const MAX_ADDRESS_LENGTH    = 500;

// ── Helpers ───────────────────────────────────────────────────────────────────

function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function metersPerPixel(lat: number, zoom: number): number {
  // Web Mercator formula — accurate to within ~1% for zoom ≥ 10
  return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, zoom);
}

function buildSummary(detections: Detection[], _threshold: number) {
  const by_class: Record<string, number> = {};
  let high_confidence = 0;
  let low_confidence  = 0;
  for (const det of detections) {
    by_class[det.class_name] = (by_class[det.class_name] ?? 0) + 1;
    if (det.confidence >= 0.7) high_confidence++;
    else low_confidence++;
  }
  return { total_objects: detections.length, by_class, high_confidence, low_confidence };
}

// ── Geocoding (Nominatim — no API key required) ───────────────────────────────

async function geocodeAddress(address: string): Promise<{ lat: number; lng: number; display: string }> {
  const key = GOOGLE_GEOCODING_API_KEY || GOOGLE_MAPS_API_KEY;
  if (key) {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(address)}&key=${key}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Geocoding request failed: ${res.status}`);
    const data = await res.json();
    if (data.status !== "OK" || !data.results?.length) {
      throw new Error(`Could not find location: "${address}" (${data.status})`);
    }
    const loc = data.results[0].geometry.location;
    return { lat: loc.lat, lng: loc.lng, display: data.results[0].formatted_address };
  }
  // Fallback: Nominatim
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(address)}&format=json&limit=1`;
  const res = await fetch(url, {
    headers: { "User-Agent": "SatelliteVision/1.0 (satellite-detector-demo.vercel.app)" },
  });
  if (!res.ok) throw new Error(`Geocoding request failed: ${res.status}`);
  const data = await res.json();
  if (!data.length) throw new Error(`Could not find location: "${address}"`);
  return { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon), display: data[0].display_name };
}

// ── Satellite Imagery (Google Maps Static API) ────────────────────────────────

async function fetchSatelliteImage(
  lat: number,
  lng: number,
  zoom: number,
): Promise<{ base64: string; mpp: number; coverage: number }> {
  if (!GOOGLE_MAPS_API_KEY) {
    throw new Error(
      "GOOGLE_MAPS_API_KEY secret is not set. " +
      "Add it via: supabase secrets set GOOGLE_MAPS_API_KEY=your_key"
    );
  }

  // scale=2 doubles pixel density (1280×1280) at the same geographic coverage —
  // brings effective resolution to ~0.25m/px at zoom 18, closer to training data
  const url =
    `https://maps.googleapis.com/maps/api/staticmap` +
    `?center=${lat},${lng}` +
    `&zoom=${zoom}` +
    `&size=640x640` +
    `&scale=2` +
    `&maptype=satellite` +
    `&key=${GOOGLE_MAPS_API_KEY}`;

  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Google Maps Static API error ${res.status}: ${body.slice(0, 200)}`);
  }

  const bytes = new Uint8Array(await res.arrayBuffer());
  let binary = '';
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  const base64 = btoa(binary);
  const mpp     = metersPerPixel(lat, zoom);
  // 640px reported size × scale 2 = 1280 actual pixels; coverage = 1280 × mpp
  const coverage = Math.round(1280 * mpp);

  return { base64, mpp: Math.round(mpp * 1000) / 1000, coverage };
}

// ── Logging ───────────────────────────────────────────────────────────────────

async function logRequest(
  supabase: ReturnType<typeof createClient>,
  requestId: string,
  clientId: string | undefined,
  processingMs: number,
  nDetections: number,
  success: boolean,
  errorMsg?: string,
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
    console.warn("Failed to log request:", _e);
  }
}

async function checkRateLimit(
  supabase: ReturnType<typeof createClient>,
  clientId: string,
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
    return true;
  }
}

// ── Main Handler ──────────────────────────────────────────────────────────────

Deno.serve(async (req: Request) => {
  const cors = getCorsHeaders(req);

  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: cors });
  }

  const requestId = generateRequestId();
  const startTime = Date.now();
  const supabase  = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  try {
    // ── Auth ────────────────────────────────────────────────────────────────
    const authHeader = req.headers.get("Authorization");
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return new Response(
        JSON.stringify({ success: false, error: "Authentication required" }),
        { status: 401, headers: { ...cors, "Content-Type": "application/json" } },
      );
    }

    const { data: { user }, error: authError } = await supabase.auth.getUser(
      authHeader.slice(7),
    );
    if (authError || !user) {
      return new Response(
        JSON.stringify({ success: false, error: "Invalid or expired token" }),
        { status: 401, headers: { ...cors, "Content-Type": "application/json" } },
      );
    }

    // ── Server-side search limit enforcement ─────────────────────────────────
    const { data: profile } = await supabase
      .from("profiles")
      .select("searches_this_month, searches_limit, plan")
      .eq("id", user.id)
      .single();

    if (profile) {
      const used  = profile.searches_this_month ?? 0;
      const limit = profile.searches_limit ?? 0;
      if (limit > 0 && used >= limit) {
        console.warn(`[detect] Search limit reached for user ${user.id}: ${used}/${limit}`);
        return new Response(
          JSON.stringify({
            success: false,
            error: `Monthly search limit reached (${used}/${limit}). Please upgrade your plan.`,
            limit_reached: true,
            current_plan: profile.plan,
          }),
          { status: 429, headers: { ...cors, "Content-Type": "application/json" } },
        );
      }
    }

    // ── Parse request ────────────────────────────────────────────────────────
    const body: DetectionRequest = await req.json();

    // Clamp confidence threshold to safe minimum
    const confidenceThreshold = Math.max(
      body.confidence_threshold ?? DEFAULT_CONFIDENCE,
      MIN_CONFIDENCE,
    );
    const clientId = user.id; // Always use auth user id, ignore client-supplied id

    // Validate address length
    if (body.address && body.address.length > MAX_ADDRESS_LENGTH) {
      return new Response(
        JSON.stringify({ success: false, error: "Address too long (max 500 characters)" }),
        { status: 400, headers: { ...cors, "Content-Type": "application/json" } },
      );
    }

    // ── Rate limit ───────────────────────────────────────────────────────────
    if (!await checkRateLimit(supabase, clientId)) {
      return new Response(
        JSON.stringify({
          success: false,
          error: "Rate limit exceeded. Please slow down.",
        }),
        { status: 429, headers: { ...cors, "Content-Type": "application/json" } },
      );
    }

    // ── Resolve imagery ──────────────────────────────────────────────────────
    let imageBase64: string | undefined  = body.image_base64;
    let imageUrl:    string | undefined  = body.image_url;
    let imagerySource: DetectionResponse["imagery_source"] = "upload";
    let location: Location | undefined;

    if (body.address || (body.lat !== undefined && body.lng !== undefined)) {
      // Geo mode: geocode address if needed, then fetch satellite imagery
      const zoom = body.zoom ?? DEFAULT_ZOOM;
      let lat = body.lat!;
      let lng = body.lng!;
      let resolvedAddress: string | undefined;

      if (body.address) {
        const geo = await geocodeAddress(body.address);
        lat = geo.lat;
        lng = geo.lng;
        resolvedAddress = geo.display;
      }

      const { base64, mpp, coverage } = await fetchSatelliteImage(lat, lng, zoom);
      imageBase64   = base64;
      imagerySource = "google_maps";
      location = {
        lat,
        lng,
        address: resolvedAddress ?? body.address,
        zoom,
        meters_per_pixel: mpp,
        coverage_m: coverage,
      };

    } else if (body.image_url) {
      imagerySource = "url";
    } else if (body.image_base64) {
      imagerySource = "upload";
    } else {
      return new Response(
        JSON.stringify({
          success: false,
          error: "Provide one of: address, lat+lng, image_url, or image_base64",
        }),
        { status: 400, headers: { ...cors, "Content-Type": "application/json" } },
      );
    }

    // ── Forward to inference server ──────────────────────────────────────────
    if (!INFERENCE_SERVER_URL) {
      // Inference server not configured — fail with a clear error rather than returning mock data
      console.error("[detect] INFERENCE_SERVER_URL is not set");
      return new Response(
        JSON.stringify({ success: false, error: "Detection service unavailable. Please try again later." }),
        { status: 503, headers: { ...cors, "Content-Type": "application/json" } },
      );
    }

    const inferenceRes = await fetch(`${INFERENCE_SERVER_URL}/detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": INFERENCE_API_KEY,
      },
      body: JSON.stringify({
        image_url:            imageUrl,
        image_base64:         imageBase64,
        confidence_threshold: confidenceThreshold,
        classes:              body.classes,
        return_image:         body.return_image ?? false,
      }),
    });

    if (!inferenceRes.ok) {
      const errText = await inferenceRes.text();
      throw new Error(`Inference server error ${inferenceRes.status}: ${errText}`);
    }

    const inferenceData = await inferenceRes.json();
    const processingMs  = Date.now() - startTime;
    const detections: Detection[] = inferenceData.detections ?? [];

    const response: DetectionResponse = {
      success: true,
      request_id: requestId,
      model_version:      inferenceData.model_version ?? "yolov8n-satellite-v1",
      processing_time_ms: processingMs,
      imagery_source:     imagerySource,
      location,
      image_width:        inferenceData.image_width,
      image_height:       inferenceData.image_height,
      detections,
      summary:            buildSummary(detections, confidenceThreshold),
      annotated_image_base64: inferenceData.annotated_image_base64,
      satellite_image_base64: imagerySource === "google_maps" ? imageBase64 : undefined,
    };

    await logRequest(supabase, requestId, clientId, processingMs, detections.length, true);

    // Increment monthly search count atomically
    await supabase.rpc("increment_search_count", { user_id_input: user.id });

    return new Response(JSON.stringify(response), {
      headers: { ...cors, "Content-Type": "application/json" },
    });

  } catch (error) {
    const processingMs = Date.now() - startTime;
    const errMsg = error instanceof Error ? error.message : String(error);
    console.error(`[${requestId}] Error:`, errMsg);
    await logRequest(supabase, requestId, undefined, processingMs, 0, false, errMsg);
    // Never expose internal error details to client
    return new Response(
      JSON.stringify({ success: false, request_id: requestId, error: "Detection failed. Please try again." }),
      { status: 500, headers: { ...cors, "Content-Type": "application/json" } },
    );
  }
});
