/**
 * Basecamp OAuth flow — run once to get tokens
 * Usage: node auth.js
 */

import http from "http";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TOKENS_FILE = join(__dirname, "tokens.json");

const CLIENT_ID     = "73fdac6b4f4a1a56d087f777d5dc26fa4f3adcc1";
const CLIENT_SECRET = "9f5e626ba9d6c7ef222771e5806f9d737986e8d0";
const REDIRECT_URI  = "http://localhost:3000/callback";

export function loadTokens() {
  if (!existsSync(TOKENS_FILE)) return null;
  try { return JSON.parse(readFileSync(TOKENS_FILE, "utf8")); }
  catch { return null; }
}

export function saveTokens(tokens) {
  writeFileSync(TOKENS_FILE, JSON.stringify(tokens, null, 2));
}

export async function refreshAccessToken(refreshToken) {
  const resp = await fetch("https://launchpad.37signals.com/authorization/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      type: "refresh",
      refresh_token: refreshToken,
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
      redirect_uri: REDIRECT_URI,
    }),
  });
  const data = await resp.json();
  if (!data.access_token) throw new Error("Token refresh failed: " + JSON.stringify(data));
  return data;
}

async function runOAuthFlow() {
  const authUrl = `https://launchpad.37signals.com/authorization/new?type=web_server&client_id=${CLIENT_ID}&redirect_uri=${encodeURIComponent(REDIRECT_URI)}`;

  console.log("\n🔐 Basecamp OAuth Setup");
  console.log("Opening browser for authorization...\n");
  console.log("If browser does not open, visit:\n" + authUrl);

  // Open browser
  const { exec } = await import("child_process");
  exec(`open "${authUrl}"`);

  // Start local callback server
  await new Promise((resolve, reject) => {
    const server = http.createServer(async (req, res) => {
      const url = new URL(req.url, "http://localhost:3000");
      const code = url.searchParams.get("code");
      if (!code) { res.end("No code received."); return; }

      try {
        // Exchange code for tokens
        const tokenResp = await fetch("https://launchpad.37signals.com/authorization/token", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams({
            type: "web_server",
            client_id: CLIENT_ID,
            client_secret: CLIENT_SECRET,
            redirect_uri: REDIRECT_URI,
            code,
          }),
        });
        const tokens = await tokenResp.json();
        if (!tokens.access_token) throw new Error(JSON.stringify(tokens));

        // Get account info
        const meResp = await fetch("https://launchpad.37signals.com/authorization.json", {
          headers: { Authorization: `Bearer ${tokens.access_token}` },
        });
        const me = await meResp.json();
        const account = me.accounts?.find(a => a.product === "bc3") || me.accounts?.[0];

        saveTokens({
          access_token: tokens.access_token,
          refresh_token: tokens.refresh_token,
          expires_in: tokens.expires_in,
          obtained_at: Date.now(),
          account_id: account?.id,
          account_name: account?.name,
        });

        res.end("<h2>✅ Authorized! You can close this tab and return to Claude.</h2>");
        console.log(`\n✅ Tokens saved. Account: ${account?.name} (${account?.id})`);
        server.close();
        resolve();
      } catch (err) {
        res.end("Error: " + err.message);
        reject(err);
      }
    });

    server.listen(3000, () => console.log("Waiting for Basecamp authorization..."));
    server.on("error", reject);
  });
}

// Run if called directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  runOAuthFlow().catch(console.error);
}
