/**
 * supabase/functions/stripe-webhook/index.ts
 *
 * Handles Stripe webhook events to keep Supabase in sync with subscription state.
 * Add this URL as a webhook endpoint in Stripe Dashboard:
 *   https://obdsgqjkjjmmtbcfjhnn.supabase.co/functions/v1/stripe-webhook
 *
 * Required secrets:
 *   STRIPE_WEBHOOK_SECRET   — whsec_... (from Stripe Dashboard → Webhooks)
 *   STRIPE_SECRET_KEY       — sk_live_...
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { getCorsHeaders } from "../_shared/cors.ts";

const STRIPE_WEBHOOK_SECRET     = Deno.env.get("STRIPE_WEBHOOK_SECRET") ?? "";
const SUPABASE_URL              = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

const VALID_PLANS = ["starter", "professional", "enterprise"] as const;
type Plan = typeof VALID_PLANS[number];

const PLAN_LIMITS: Record<Plan, number> = {
  starter:      200,
  professional: 1000,
  enterprise:   5000,
};

// Webhook timestamp tolerance — reject events older than 5 minutes (replay attack prevention)
const WEBHOOK_TOLERANCE_SECONDS = 300;

async function verifyStripeSignature(body: string, signature: string, secret: string): Promise<boolean> {
  const parts  = signature.split(",");
  const tsPart = parts.find(p => p.startsWith("t="));
  const v1Part = parts.find(p => p.startsWith("v1="));
  if (!tsPart || !v1Part) return false;

  const ts       = tsPart.slice(2);
  const expected = v1Part.slice(3);

  // Replay attack prevention — reject stale timestamps
  const tsNum  = parseInt(ts, 10);
  const nowSec = Math.floor(Date.now() / 1000);
  if (isNaN(tsNum) || Math.abs(nowSec - tsNum) > WEBHOOK_TOLERANCE_SECONDS) {
    console.warn(`[stripe-webhook] Rejected stale timestamp: ${tsNum}, now: ${nowSec}`);
    return false;
  }

  const payload = `${ts}.${body}`;
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  const hex = Array.from(new Uint8Array(sig)).map(b => b.toString(16).padStart(2, "0")).join("");
  return hex === expected;
}

Deno.serve(async (req: Request) => {
  const cors = getCorsHeaders(req);

  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: cors });
  }

  // Mandatory: webhook secret must be configured
  if (!STRIPE_WEBHOOK_SECRET) {
    console.error("[stripe-webhook] STRIPE_WEBHOOK_SECRET is not set — rejecting all webhook requests");
    return new Response("Webhook not configured", { status: 503 });
  }

  const body      = await req.text();
  const signature = req.headers.get("stripe-signature") ?? "";

  if (!signature) {
    console.warn("[stripe-webhook] Missing stripe-signature header");
    return new Response("Missing signature", { status: 400 });
  }

  const valid = await verifyStripeSignature(body, signature, STRIPE_WEBHOOK_SECRET);
  if (!valid) {
    console.warn("[stripe-webhook] Invalid or stale signature — possible replay attack");
    return new Response("Invalid signature", { status: 403 });
  }

  let event: Record<string, unknown>;
  try {
    event = JSON.parse(body);
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  try {
    switch (event.type) {
      case "customer.subscription.created":
      case "customer.subscription.updated": {
        const sub    = event.data.object as Record<string, unknown>;
        const meta   = sub.metadata as Record<string, string> | undefined;
        const userId = meta?.supabase_user_id;
        const rawPlan = meta?.plan;

        // Validate userId is a UUID
        if (!userId || !/^[0-9a-f-]{36}$/i.test(userId)) {
          console.warn(`[stripe-webhook] Invalid or missing user_id in metadata: ${userId}`);
          break;
        }

        // Validate plan is one of the allowed values
        if (!rawPlan || !VALID_PLANS.includes(rawPlan as Plan)) {
          console.warn(`[stripe-webhook] Invalid plan in metadata: ${rawPlan}`);
          break;
        }
        const plan = rawPlan as Plan;

        await supabase.from("subscriptions").upsert({
          user_id:                userId,
          plan,
          status:                 sub.status,
          stripe_subscription_id: sub.id,
          stripe_customer_id:     sub.customer,
          current_period_start:   new Date((sub.current_period_start as number) * 1000).toISOString(),
          current_period_end:     new Date((sub.current_period_end   as number) * 1000).toISOString(),
          updated_at:             new Date().toISOString(),
        }, { onConflict: "stripe_subscription_id" });

        await supabase.from("profiles").update({
          plan,
          searches_limit: PLAN_LIMITS[plan],
          updated_at: new Date().toISOString(),
        }).eq("id", userId);

        console.log(`[stripe-webhook] Updated user ${userId} to plan ${plan}`);
        break;
      }

      case "customer.subscription.deleted": {
        const sub    = event.data.object as Record<string, unknown>;
        const meta   = sub.metadata as Record<string, string> | undefined;
        const userId = meta?.supabase_user_id;

        if (!userId || !/^[0-9a-f-]{36}$/i.test(userId)) {
          console.warn(`[stripe-webhook] Invalid or missing user_id in metadata: ${userId}`);
          break;
        }

        await supabase.from("subscriptions").update({
          status:     "canceled",
          updated_at: new Date().toISOString(),
        }).eq("stripe_subscription_id", sub.id as string);

        // Downgrade to starter — set limit to 0 until they re-subscribe
        await supabase.from("profiles").update({
          plan:           "starter",
          searches_limit: 0,
          updated_at:     new Date().toISOString(),
        }).eq("id", userId);

        console.log(`[stripe-webhook] Canceled subscription for user ${userId}`);
        break;
      }

      case "invoice.payment_failed": {
        const invoice = event.data.object as Record<string, unknown>;
        const subId   = invoice.subscription as string | undefined;
        if (subId) {
          await supabase.from("subscriptions")
            .update({ status: "past_due", updated_at: new Date().toISOString() })
            .eq("stripe_subscription_id", subId);
          console.log(`[stripe-webhook] Marked subscription ${subId} as past_due`);
        }
        break;
      }

      default:
        console.log(`[stripe-webhook] Unhandled event type: ${event.type}`);
    }

    return new Response(JSON.stringify({ received: true }), {
      headers: { ...cors, "Content-Type": "application/json" },
    });

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[stripe-webhook] Processing error:", msg);
    // Return 200 to prevent Stripe from retrying — log the error internally
    return new Response(JSON.stringify({ received: true, error: "Internal processing error" }), {
      status: 200, headers: { ...cors, "Content-Type": "application/json" },
    });
  }
});
