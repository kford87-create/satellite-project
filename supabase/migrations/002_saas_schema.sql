-- ═══════════════════════════════════════════════════════════════════════════
-- 002_saas_schema.sql
-- SatelliteVision SaaS — user profiles, search history, subscriptions
-- Run in Supabase Dashboard → SQL Editor
-- ═══════════════════════════════════════════════════════════════════════════

-- ── Profiles ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS profiles (
  id                  UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email               TEXT NOT NULL,
  full_name           TEXT,
  company_name        TEXT,
  phone               TEXT,
  job_title           TEXT,
  plan                TEXT NOT NULL DEFAULT 'starter',   -- starter | professional | enterprise
  searches_this_month INT  NOT NULL DEFAULT 0,
  searches_limit      INT  NOT NULL DEFAULT 200,
  stripe_customer_id  TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own profile"
  ON profiles FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON profiles FOR INSERT WITH CHECK (auth.uid() = id);

-- Service role bypass (for Edge Functions + admin)
CREATE POLICY "Service role full access to profiles"
  ON profiles FOR ALL USING (auth.role() = 'service_role');

-- ── Searches ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS searches (
  id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id             UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  address             TEXT,
  lat                 FLOAT,
  lng                 FLOAT,
  zoom                INT,
  imagery_source      TEXT,                   -- google_maps | upload | url
  image_storage_path  TEXT,                   -- Supabase Storage path
  detections          JSONB,                  -- full detection array
  detection_count     INT NOT NULL DEFAULT 0,
  processing_time_ms  INT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE searches ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own searches"
  ON searches FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own searches"
  ON searches FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Service role full access to searches"
  ON searches FOR ALL USING (auth.role() = 'service_role');

CREATE INDEX IF NOT EXISTS searches_user_id_idx ON searches (user_id);
CREATE INDEX IF NOT EXISTS searches_created_at_idx ON searches (created_at DESC);

-- ── Subscriptions ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS subscriptions (
  id                      UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id                 UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  plan                    TEXT,
  status                  TEXT,               -- active | canceled | past_due | trialing
  stripe_subscription_id  TEXT UNIQUE,
  stripe_customer_id      TEXT,
  current_period_start    TIMESTAMPTZ,
  current_period_end      TIMESTAMPTZ,
  created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own subscription"
  ON subscriptions FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Service role full access to subscriptions"
  ON subscriptions FOR ALL USING (auth.role() = 'service_role');

-- ── Storage bucket for satellite images ─────────────────────────────────────
INSERT INTO storage.buckets (id, name, public)
VALUES ('search-images', 'search-images', false)
ON CONFLICT (id) DO NOTHING;

CREATE POLICY "Users can upload own images"
  ON storage.objects FOR INSERT
  WITH CHECK (bucket_id = 'search-images' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can read own images"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'search-images' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Service role full access to images"
  ON storage.objects FOR ALL
  USING (bucket_id = 'search-images' AND auth.role() = 'service_role');

-- ── Updated_at trigger ───────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at();

CREATE TRIGGER set_subscriptions_updated_at
  BEFORE UPDATE ON subscriptions
  FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at();

-- ── Admin view (service role only) ──────────────────────────────────────────
CREATE OR REPLACE VIEW admin_user_overview AS
SELECT
  p.id,
  p.email,
  p.full_name,
  p.company_name,
  p.phone,
  p.job_title,
  p.plan,
  p.searches_this_month,
  p.searches_limit,
  p.created_at,
  COUNT(s.id)            AS total_searches,
  MAX(s.created_at)      AS last_search_at
FROM profiles p
LEFT JOIN searches s ON s.user_id = p.id
GROUP BY p.id;

-- ── Monthly reset function (call via cron or manually) ───────────────────────
CREATE OR REPLACE FUNCTION reset_monthly_search_counts()
RETURNS void AS $$
BEGIN
  UPDATE profiles SET searches_this_month = 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
