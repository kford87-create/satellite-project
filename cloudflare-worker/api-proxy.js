/**
 * Cloudflare Worker — api.kestrelai.io
 *
 * Proxies all requests from api.kestrelai.io to the Supabase Edge Function,
 * keeping the Supabase project URL internal.
 *
 * Deploy:
 *   1. Install Wrangler: npm install -g wrangler
 *   2. wrangler login
 *   3. wrangler deploy
 *   4. In Cloudflare dashboard: Workers & Pages → api-proxy → Settings →
 *      Triggers → Add Custom Domain → api.kestrelai.io
 */

const SUPABASE_FUNCTIONS_URL = "https://obdsgqjkjjmmtbcfjhnn.supabase.co/functions/v1";

const ALLOWED_ORIGINS = [
  "https://kestrelai.io",
  "https://www.kestrelai.io",
  "https://satellite-detector-demo.vercel.app",
  "https://public-chi-sable.vercel.app",
];

// Headers leaked by the Supabase gateway — strip from all responses
const STRIPPED_HEADERS = [
  "sb-gateway-version",
  "sb-project-ref",
  "sb-request-id",
  "x-sb-edge-region",
  "x-served-by",
];

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const origin = request.headers.get("Origin");

    // Health check
    if (url.pathname === "/" || url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "ok", service: "Kestrel AI API" }), {
        headers: { "Content-Type": "application/json", ...responseHeaders(origin) },
      });
    }

    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: responseHeaders(origin),
      });
    }

    // Build upstream URL — strip leading slash from pathname
    const upstreamUrl = `${SUPABASE_FUNCTIONS_URL}${url.pathname}${url.search}`;

    // Forward the request
    const upstreamRequest = new Request(upstreamUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body,
      redirect: "follow",
    });

    try {
      const response = await fetch(upstreamRequest);

      // Build clean response headers: strip upstream leaks, apply ours
      const newHeaders = new Headers(response.headers);
      STRIPPED_HEADERS.forEach((h) => newHeaders.delete(h));
      newHeaders.delete("access-control-allow-origin"); // remove upstream wildcard
      Object.entries(responseHeaders(origin)).forEach(([k, v]) => newHeaders.set(k, v));

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders,
      });
    } catch (err) {
      return new Response(
        JSON.stringify({ error: "Upstream request failed", detail: err.message }),
        { status: 502, headers: { "Content-Type": "application/json", ...responseHeaders(origin) } }
      );
    }
  },
};

function responseHeaders(origin) {
  const headers = {
    // CORS
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Api-Key",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
    // Security
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
  };
  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Access-Control-Allow-Credentials"] = "true";
  }
  return headers;
}
