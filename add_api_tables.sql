-- ============================================================
-- supabase/migrations/add_api_tables.sql
--
-- Run this in Supabase Dashboard → SQL Editor
-- Adds tables needed for the Edge Function API layer
-- ============================================================

-- API request log — tracks every call to the detect-objects endpoint
-- Used for: usage analytics, billing, rate limiting, debugging
CREATE TABLE IF NOT EXISTS api_requests (
    id                  BIGSERIAL PRIMARY KEY,
    request_id          TEXT UNIQUE NOT NULL,
    client_id           TEXT NOT NULL DEFAULT 'anonymous',
    processing_time_ms  INTEGER,
    n_detections        INTEGER DEFAULT 0,
    success             BOOLEAN DEFAULT TRUE,
    error_message       TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- API clients — future SaaS customer accounts
CREATE TABLE IF NOT EXISTS api_clients (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    client_name     TEXT NOT NULL,
    plan            TEXT NOT NULL DEFAULT 'free',   -- 'free', 'pro', 'enterprise'
    monthly_quota   INTEGER DEFAULT 1000,           -- Requests per month
    requests_used   INTEGER DEFAULT 0,
    api_key_hash    TEXT,                           -- Hashed API key for reference
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_api_requests_client ON api_requests(client_id);
CREATE INDEX IF NOT EXISTS idx_api_requests_created ON api_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_api_clients_user ON api_clients(user_id);

-- Row Level Security — users can only see their own requests
ALTER TABLE api_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_clients ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own requests"
    ON api_requests FOR SELECT
    USING (client_id = auth.uid()::text);

CREATE POLICY "Users see own client record"
    ON api_clients FOR SELECT
    USING (user_id = auth.uid());

-- Auto-update updated_at on api_clients whenever a row is modified
-- Without this trigger, updated_at would stay frozen at the insert time
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_api_clients_updated_at
    BEFORE UPDATE ON api_clients
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Useful view: usage summary per client (for a future billing dashboard)
CREATE OR REPLACE VIEW client_usage_summary AS
SELECT
    client_id,
    COUNT(*) AS total_requests,
    COUNT(*) FILTER (WHERE success = TRUE) AS successful_requests,
    COUNT(*) FILTER (WHERE success = FALSE) AS failed_requests,
    AVG(processing_time_ms) AS avg_processing_ms,
    SUM(n_detections) AS total_detections,
    DATE_TRUNC('month', created_at) AS month
FROM api_requests
GROUP BY client_id, DATE_TRUNC('month', created_at)
ORDER BY month DESC;
