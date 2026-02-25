-- ═══════════════════════════════════════════════════════════════════════════
-- 003_security_hardening.sql
-- Security hardening: admin RBAC, constraints, audit log, search increment
-- Run in Supabase Dashboard → SQL Editor
-- ═══════════════════════════════════════════════════════════════════════════

-- ── 1. Admin flag on profiles (replaces hardcoded email check) ───────────────
ALTER TABLE profiles
  ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;

-- Set the initial admin
UPDATE profiles SET is_admin = TRUE
WHERE email = 'kahlil.ford87@gmail.com';

-- RLS: users can read their own is_admin flag
CREATE POLICY IF NOT EXISTS "Users can read own is_admin"
  ON profiles FOR SELECT USING (auth.uid() = id);

-- ── 2. Unique constraint on stripe_customer_id ───────────────────────────────
ALTER TABLE profiles
  ADD CONSTRAINT IF NOT EXISTS profiles_stripe_customer_id_unique
  UNIQUE (stripe_customer_id);

-- ── 3. Add check constraints ─────────────────────────────────────────────────
ALTER TABLE profiles
  ADD CONSTRAINT IF NOT EXISTS profiles_searches_limit_nonneg
    CHECK (searches_limit >= 0),
  ADD CONSTRAINT IF NOT EXISTS profiles_searches_this_month_nonneg
    CHECK (searches_this_month >= 0),
  ADD CONSTRAINT IF NOT EXISTS profiles_plan_valid
    CHECK (plan IN ('starter', 'professional', 'enterprise'));

-- ── 4. Atomic search count increment function ────────────────────────────────
CREATE OR REPLACE FUNCTION increment_search_count(user_id_input UUID)
RETURNS void AS $$
BEGIN
  UPDATE profiles
  SET searches_this_month = searches_this_month + 1,
      updated_at = NOW()
  WHERE id = user_id_input;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ── 5. Admin audit log ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS admin_audit_log (
  id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  admin_id   UUID REFERENCES auth.users(id),
  action     TEXT NOT NULL,
  target_id  UUID,
  details    JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE admin_audit_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admins can insert audit log"
  ON admin_audit_log FOR INSERT
  WITH CHECK (auth.uid() = admin_id);

CREATE POLICY "Service role full access to audit log"
  ON admin_audit_log FOR ALL
  USING (auth.role() = 'service_role');

CREATE INDEX IF NOT EXISTS admin_audit_log_created_at_idx
  ON admin_audit_log (created_at DESC);

-- ── 6. Checkout attempts table (rate limiting) ───────────────────────────────
CREATE TABLE IF NOT EXISTS checkout_attempts (
  id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id    UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  plan       TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE checkout_attempts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access to checkout_attempts"
  ON checkout_attempts FOR ALL
  USING (auth.role() = 'service_role');

CREATE INDEX IF NOT EXISTS checkout_attempts_user_created_idx
  ON checkout_attempts (user_id, created_at DESC);

-- Auto-clean checkout attempts older than 24 hours (keep table small)
CREATE OR REPLACE FUNCTION cleanup_old_checkout_attempts()
RETURNS void AS $$
BEGIN
  DELETE FROM checkout_attempts
  WHERE created_at < NOW() - INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
