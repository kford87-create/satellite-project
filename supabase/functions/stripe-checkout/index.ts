/**
 * supabase/functions/stripe-checkout/index.ts
 *
 * Creates a Stripe Checkout Session for a given plan.
 * Client receives a { url } and redirects to Stripe hosted checkout.
 *
 * Required secrets (set via `supabase secrets set`):
 *   STRIPE_SECRET_KEY       — sk_live_... or sk_test_...
 *   STRIPE_PRICE_STARTER    — price_... (Starter $99/mo)
 *   STRIPE_PRICE_PROFESSIONAL — price_... (Professional $399/mo)
 *   STRIPE_PRICE_ENTERPRISE — price_... (Enterprise $1499/mo)
 *   APP_URL                 — https://satellite-detector-demo.vercel.app
 */

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { getCorsHeaders } from "../_shared/cors.ts";

const STRIPE_SECRET_KEY         = Deno.env.get("STRIPE_SECRET_KEY") ?? "";
const STRIPE_PRICE_STARTER      = Deno.env.get("STRIPE_PRICE_STARTER") ?? "";
const STRIPE_PRICE_PROFESSIONAL = Deno.env.get("STRIPE_PRICE_PROFESSIONAL") ?? "";
const STRIPE_PRICE_ENTERPRISE   = Deno.env.get("STRIPE_PRICE_ENTERPRISE") ?? "";
const SUPABASE_URL              = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const APP_URL                   = Deno.env.get("APP_URL") ?? "https://satellite-detector-demo.vercel.app";

const VALID_PLANS = ["starter", "professional", "enterprise"] as const;
type Plan = typeof VALID_PLANS[number];

const PRICE_MAP: Record<Plan, string> = {
  starter:      STRIPE_PRICE_STARTER,
  professional: STRIPE_PRICE_PROFESSIONAL,
  enterprise:   STRIPE_PRICE_ENTERPRISE,
};

const SEARCH_LIMITS: Record<Plan, number> = {
  starter:      200,
  professional: 1000,
  enterprise:   5000,
};

// Checkout rate limit: max 5 sessions per user per hour
const CHECKOUT_RATE_LIMIT = 5;

async function checkCheckoutRateLimit(
  supabase: ReturnType<typeof createClient>,
  userId: string,
): Promise<boolean> {
  try {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const { count } = await supabase
      .from("checkout_attempts")
      .select("*", { count: "exact", head: true })
      .eq("user_id", userId)
      .gte("created_at", oneHourAgo);
    return (count ?? 0) < CHECKOUT_RATE_LIMIT;
  } catch {
    return true; // Fail open if table doesn't exist yet
  }
}

async function logCheckoutAttempt(
  supabase: ReturnType<typeof createClient>,
  userId: string,
  plan: string,
) {
  try {
    await supabase.from("checkout_attempts").insert({ user_id: userId, plan });
  } catch {
    // Non-critical — don't fail the request
  }
}

Deno.serve(async (req: Request) => {
  const cors = getCorsHeaders(req);

  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: cors });
  }

  if (!STRIPE_SECRET_KEY) {
    return new Response(
      JSON.stringify({ error: "Stripe not configured." }),
      { status: 503, headers: { ...cors, "Content-Type": "application/json" } },
    );
  }

  try {
    // ── Auth ─────────────────────────────────────────────────────────────────
    const authHeader = req.headers.get("Authorization");
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return new Response(JSON.stringify({ error: "Missing or invalid Authorization header" }), {
        status: 401, headers: { ...cors, "Content-Type": "application/json" },
      });
    }

    const token    = authHeader.slice(7); // Remove "Bearer "
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      return new Response(JSON.stringify({ error: "Invalid or expired token" }), {
        status: 401, headers: { ...cors, "Content-Type": "application/json" },
      });
    }

    // ── Rate limit ────────────────────────────────────────────────────────────
    if (!await checkCheckoutRateLimit(supabase, user.id)) {
      console.warn(`[stripe-checkout] Rate limit hit for user ${user.id}`);
      return new Response(JSON.stringify({ error: "Too many checkout attempts. Try again later." }), {
        status: 429, headers: { ...cors, "Content-Type": "application/json" },
      });
    }

    // ── Validate plan ─────────────────────────────────────────────────────────
    let body: { plan?: string };
    try {
      body = await req.json();
    } catch {
      return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
        status: 400, headers: { ...cors, "Content-Type": "application/json" },
      });
    }

    const plan = body.plan;
    if (!plan || !VALID_PLANS.includes(plan as Plan)) {
      return new Response(JSON.stringify({ error: "Invalid plan. Must be starter, professional, or enterprise." }), {
        status: 400, headers: { ...cors, "Content-Type": "application/json" },
      });
    }
    const validPlan = plan as Plan;
    const priceId   = PRICE_MAP[validPlan];

    if (!priceId) {
      return new Response(JSON.stringify({ error: `Price not configured for plan: ${validPlan}` }), {
        status: 503, headers: { ...cors, "Content-Type": "application/json" },
      });
    }

    // ── Get or create Stripe customer ─────────────────────────────────────────
    const { data: profile } = await supabase
      .from("profiles")
      .select("stripe_customer_id, email, full_name, plan")
      .eq("id", user.id)
      .single();

    let customerId = profile?.stripe_customer_id;

    if (!customerId) {
      const customerRes = await fetch("https://api.stripe.com/v1/customers", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${STRIPE_SECRET_KEY}`,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          email: profile?.email ?? user.email ?? "",
          name:  profile?.full_name ?? user.email ?? "",
          "metadata[supabase_user_id]": user.id,
        }).toString(),
      });
      const customer = await customerRes.json();
      if (customer.error) {
        throw new Error(`Failed to create Stripe customer: ${customer.error.message}`);
      }
      customerId = customer.id;

      // Only update stripe_customer_id — never overwrite existing plan or limits
      if (profile) {
        await supabase.from("profiles")
          .update({ stripe_customer_id: customerId })
          .eq("id", user.id);
      } else {
        // New user — insert full profile with defaults
        await supabase.from("profiles").insert({
          id:                 user.id,
          email:              user.email ?? "",
          stripe_customer_id: customerId,
          plan:               "starter",
          searches_this_month: 0,
          searches_limit:     SEARCH_LIMITS["starter"],
        });
      }
    }

    // ── Log checkout attempt ──────────────────────────────────────────────────
    await logCheckoutAttempt(supabase, user.id, validPlan);

    // ── Create Checkout Session ───────────────────────────────────────────────
    const sessionRes = await fetch("https://api.stripe.com/v1/checkout/sessions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${STRIPE_SECRET_KEY}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        customer:                          customerId,
        mode:                              "subscription",
        "line_items[0][price]":            priceId,
        "line_items[0][quantity]":         "1",
        success_url:                       `${APP_URL}/app?subscription=success&plan=${validPlan}`,
        cancel_url:                        `${APP_URL}/#pricing`,
        "subscription_data[metadata][supabase_user_id]": user.id,
        "subscription_data[metadata][plan]": validPlan,
      }).toString(),
    });

    const session = await sessionRes.json();
    if (session.error) {
      throw new Error(`Stripe error: ${session.error.message}`);
    }

    return new Response(JSON.stringify({ url: session.url }), {
      headers: { ...cors, "Content-Type": "application/json" },
    });

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[stripe-checkout] Error:", msg);
    return new Response(JSON.stringify({ error: "An error occurred. Please try again." }), {
      status: 500, headers: { ...cors, "Content-Type": "application/json" },
    });
  }
});
