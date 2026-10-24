-- Runs once on first container start.
-- `memory` (POSTGRES_USER) owns the schema and runs migrations.
-- `memory_app` is the runtime role: no BYPASSRLS, so row-level security applies.
CREATE ROLE memory_app LOGIN PASSWORD 'memory_app' NOSUPERUSER NOBYPASSRLS;
GRANT CONNECT ON DATABASE memory TO memory_app;
GRANT USAGE ON SCHEMA public TO memory_app;
CREATE EXTENSION IF NOT EXISTS vector;
