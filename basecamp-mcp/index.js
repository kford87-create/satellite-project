/**
 * Basecamp MCP Server for Claude
 * Gives Claude direct access to your Basecamp projects, todos, messages, and campfire.
 *
 * First-time setup: node auth.js
 * Then restart Claude Code — Basecamp tools will appear automatically.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { loadTokens, saveTokens, refreshAccessToken } from "./auth.js";

const server = new McpServer({
  name: "basecamp",
  version: "1.0.0",
});

// ── Basecamp API client ────────────────────────────────────────────────────────

async function getToken() {
  let tokens = loadTokens();
  if (!tokens) throw new Error("Not authenticated. Run: node basecamp-mcp/auth.js");

  // Refresh if expired (Basecamp tokens last 2 weeks)
  const age = Date.now() - tokens.obtained_at;
  const TWO_WEEKS = 14 * 24 * 60 * 60 * 1000;
  if (age > TWO_WEEKS - 60000) {
    const fresh = await refreshAccessToken(tokens.refresh_token);
    tokens = { ...tokens, ...fresh, obtained_at: Date.now() };
    saveTokens(tokens);
  }

  return tokens;
}

async function bcFetch(path, options = {}) {
  const tokens = await getToken();
  const base = `https://3.basecampapi.com/${tokens.account_id}`;
  const url = path.startsWith("http") ? path : `${base}${path}`;

  const resp = await fetch(url, {
    ...options,
    headers: {
      Authorization: `Bearer ${tokens.access_token}`,
      "Content-Type": "application/json",
      "User-Agent": "Kestrel AI Claude MCP (kahlil.ford87@gmail.com)",
      ...options.headers,
    },
  });

  if (resp.status === 204) return null;
  if (!resp.ok) throw new Error(`Basecamp API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

// ── Category todolist mapping (KestrelAI project) ────────────────────────────
const KESTREL_PROJECT_ID = "46173187";
const CATEGORY_TODOLISTS = {
  marketing:     "9613992475",
  finance:       "9613992499",
  developer:     "9613992525",
  security:      "9613992551",
  "social media": "9613992563",
  "things to do": "9613981079",
};

// ── Tools ─────────────────────────────────────────────────────────────────────

server.tool("list_projects", "List all Basecamp projects", {}, async () => {
  const projects = await bcFetch("/projects.json");
  const list = projects.map(p => `• ${p.name} (ID: ${p.id}) — ${p.description || "no description"}`).join("\n");
  return { content: [{ type: "text", text: list || "No projects found." }] };
});

server.tool("get_todos", "Get todos from a Basecamp project", {
  project_id: z.string().describe("Basecamp project ID"),
  todolist_id: z.string().optional().describe("Specific todolist ID (optional — lists all todolists if omitted)"),
}, async ({ project_id, todolist_id }) => {
  if (todolist_id) {
    const todos = await bcFetch(`/buckets/${project_id}/todolists/${todolist_id}/todos.json`);
    const list = todos.map(t =>
      `[${t.completed ? "x" : " "}] ${t.content} (ID: ${t.id})${t.assignees?.length ? ` — ${t.assignees.map(a => a.name).join(", ")}` : ""}`
    ).join("\n");
    return { content: [{ type: "text", text: list || "No todos." }] };
  } else {
    const project = await bcFetch(`/projects/${project_id}.json`);
    const todoset = project.dock.find(d => d.name === "todoset");
    if (!todoset) return { content: [{ type: "text", text: "No todoset found." }] };
    const todolists = await bcFetch(todoset.url.replace(".json", "/todolists.json"));
    const list = todolists.map(tl => `📋 ${tl.name} (ID: ${tl.id}) — ${tl.completed_ratio}`).join("\n");
    return { content: [{ type: "text", text: list || "No todolists found." }] };
  }
});

server.tool("create_todo", "Create a new todo in Basecamp", {
  project_id: z.string().describe("Basecamp project ID"),
  todolist_id: z.string().describe("Todolist ID to add the todo to"),
  content: z.string().describe("The todo text"),
  due_on: z.string().optional().describe("Due date in YYYY-MM-DD format"),
  notify: z.boolean().optional().describe("Notify assignees (default false)"),
}, async ({ project_id, todolist_id, content, due_on, notify }) => {
  const body = { content, due_on, notify: notify ?? false };
  const todo = await bcFetch(`/buckets/${project_id}/todolists/${todolist_id}/todos.json`, {
    method: "POST",
    body: JSON.stringify(body),
  });
  return { content: [{ type: "text", text: `✅ Created: "${todo.content}" (ID: ${todo.id})` }] };
});

server.tool("create_category_todo", "Create a todo in a specific category (marketing, finance, developer, security, social media)", {
  category: z.enum(["marketing", "finance", "developer", "security", "social media"]).describe("Category: marketing, finance, developer, security, or social media"),
  content: z.string().describe("The todo text"),
  due_on: z.string().optional().describe("Due date in YYYY-MM-DD format"),
}, async ({ category, content, due_on }) => {
  const todolist_id = CATEGORY_TODOLISTS[category];
  if (!todolist_id) throw new Error(`Unknown category: ${category}`);
  const body = { content, due_on };
  const todo = await bcFetch(`/buckets/${KESTREL_PROJECT_ID}/todolists/${todolist_id}/todos.json`, {
    method: "POST",
    body: JSON.stringify(body),
  });
  return { content: [{ type: "text", text: `✅ [${category.toUpperCase()}] Created: "${todo.content}" (ID: ${todo.id})` }] };
});

server.tool("complete_todo", "Mark a Basecamp todo as complete", {
  project_id: z.string().describe("Basecamp project ID"),
  todo_id: z.string().describe("Todo ID to complete"),
}, async ({ project_id, todo_id }) => {
  await bcFetch(`/buckets/${project_id}/todos/${todo_id}/completion.json`, { method: "POST" });
  return { content: [{ type: "text", text: `✅ Todo ${todo_id} marked complete.` }] };
});

server.tool("get_messages", "Get messages from a Basecamp project message board", {
  project_id: z.string().describe("Basecamp project ID"),
}, async ({ project_id }) => {
  const project = await bcFetch(`/projects/${project_id}.json`);
  const boardDock = project.dock.find(d => d.name === "message_board");
  if (!boardDock) return { content: [{ type: "text", text: "No message board found." }] };
  const messages = await bcFetch(boardDock.url.replace(".json", "/messages.json"));
  const list = messages.map(m => `📌 ${m.subject} (ID: ${m.id})\n   ${m.creator.name} · ${new Date(m.created_at).toLocaleDateString()}`).join("\n\n");
  return { content: [{ type: "text", text: list || "No messages." }] };
});

server.tool("post_message", "Post a message to a Basecamp project message board", {
  project_id: z.string().describe("Basecamp project ID"),
  subject: z.string().describe("Message subject/title"),
  content: z.string().describe("Message body (plain text)"),
}, async ({ project_id, subject, content }) => {
  const project = await bcFetch(`/projects/${project_id}.json`);
  const boardDock = project.dock.find(d => d.name === "message_board");
  if (!boardDock) throw new Error("No message board found.");
  const msg = await bcFetch(boardDock.url.replace(".json", "/messages.json"), {
    method: "POST",
    body: JSON.stringify({ subject, content: `<p>${content}</p>` }),
  });
  return { content: [{ type: "text", text: `✅ Posted: "${msg.subject}" (ID: ${msg.id})` }] };
});

server.tool("campfire_post", "Send a message to a Basecamp project Campfire chat", {
  project_id: z.string().describe("Basecamp project ID"),
  content: z.string().describe("Chat message to send"),
}, async ({ project_id, content }) => {
  const project = await bcFetch(`/projects/${project_id}.json`);
  const chatDock = project.dock.find(d => d.name === "chat");
  if (!chatDock) throw new Error("No campfire found.");
  await bcFetch(chatDock.url.replace(".json", "/lines.json"), {
    method: "POST",
    body: JSON.stringify({ content }),
  });
  return { content: [{ type: "text", text: `✅ Sent to Campfire.` }] };
});

// ── Start server ──────────────────────────────────────────────────────────────

const transport = new StdioServerTransport();
await server.connect(transport);
