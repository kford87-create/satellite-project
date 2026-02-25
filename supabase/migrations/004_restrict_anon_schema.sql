-- 004_restrict_anon_schema.sql
-- Restrict anonymous role access to the public schema.
-- Prevents full schema introspection via the PostgREST /rest/v1/ endpoint
-- while keeping RLS-protected tables accessible for the frontend.

-- 1. Revoke broad table access from anon
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon;

-- 2. Re-grant SELECT on tables the frontend needs (RLS is still enforced)
GRANT SELECT ON profiles, searches, subscriptions TO anon;

-- 3. Enable RLS on internal/ML tables and ensure no anon access
ALTER TABLE dataset_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE bootstrap_iterations ENABLE ROW LEVEL SECURITY;
ALTER TABLE fn_reports ENABLE ROW LEVEL SECURITY;

-- Deny all access to internal tables for anon (explicit deny policies)
CREATE POLICY "Deny anon access to dataset_images"
  ON dataset_images FOR ALL TO anon USING (false);

CREATE POLICY "Deny anon access to bootstrap_iterations"
  ON bootstrap_iterations FOR ALL TO anon USING (false);

CREATE POLICY "Deny anon access to fn_reports"
  ON fn_reports FOR ALL TO anon USING (false);
