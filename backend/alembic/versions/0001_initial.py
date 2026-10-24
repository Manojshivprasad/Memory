"""initial schema: tenancy, projects/chats/turns, memories, graph, provenance, traces

Revision ID: 0001
Revises:
Create Date: 2026-09-28
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Tables scoped by row-level security on tenant_id.
RLS_TABLES = [
    "users",
    "projects",
    "chats",
    "turns",
    "memories",
    "memory_versions",
    "entities",
    "edges",
    "memory_entities",
    "episodes",
    "provenance",
    "retrieval_traces",
]

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- tenancy & auth (no RLS: read before a tenant is known) ----------
CREATE TABLE tenants (
    id          uuid PRIMARY KEY,
    name        text NOT NULL,
    settings    jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE dashboard_users (
    id            uuid PRIMARY KEY,
    tenant_id     uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email         text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    role          text NOT NULL DEFAULT 'admin',
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE api_keys (
    id          uuid PRIMARY KEY,
    tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name        text NOT NULL,
    prefix      text NOT NULL UNIQUE,
    key_hash    text NOT NULL,
    scopes      text[] NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz,
    revoked_at  timestamptz
);

-- ---------- tenant data ----------
CREATE TABLE users (
    id           uuid PRIMARY KEY,
    tenant_id    uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    external_id  text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, external_id)
);

CREATE TABLE projects (
    id                uuid PRIMARY KEY,
    tenant_id         uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id           uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name              text NOT NULL,
    description       text NOT NULL DEFAULT '',
    isolated          boolean NOT NULL DEFAULT false,
    brief             text NOT NULL DEFAULT '',
    brief_updated_at  timestamptz,
    last_sleep_at     timestamptz,
    settings          jsonb NOT NULL DEFAULT '{}',
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX projects_user_idx ON projects (tenant_id, user_id);

CREATE TABLE chats (
    id               uuid PRIMARY KEY,
    tenant_id        uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    project_id       uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title            text NOT NULL DEFAULT '',
    status           text NOT NULL DEFAULT 'open',      -- open | idle | closed
    turn_count       integer NOT NULL DEFAULT 0,
    last_active_at   timestamptz NOT NULL DEFAULT now(),
    consolidated_at  timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX chats_project_idx ON chats (tenant_id, project_id, last_active_at DESC);
CREATE INDEX chats_consolidate_idx ON chats (status, last_active_at);

-- Verbatim archive: the AI's exact record of every turn.
CREATE TABLE turns (
    id             uuid PRIMARY KEY,
    tenant_id      uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    project_id     uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chat_id        uuid NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    seq            integer NOT NULL,
    user_msg       text NOT NULL,
    assistant_msg  text NOT NULL,
    model          text,
    retrieved_ids  uuid[] NOT NULL DEFAULT '{}',
    salience       real,
    processed_at   timestamptz,
    created_at     timestamptz NOT NULL DEFAULT now(),
    tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', user_msg || ' ' || assistant_msg)) STORED
);
CREATE INDEX turns_chat_idx ON turns (chat_id, seq);
CREATE INDEX turns_project_idx ON turns (tenant_id, project_id, created_at DESC);
CREATE INDEX turns_tsv_idx ON turns USING gin (tsv);

CREATE TABLE memories (
    id                  uuid PRIMARY KEY,
    tenant_id           uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id             uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id          uuid REFERENCES projects(id) ON DELETE CASCADE,   -- NULL => user scope
    scope               text NOT NULL,                                     -- chat | project | user
    source_chat_id      uuid,
    source_turn_id      uuid,
    kind                text NOT NULL,        -- fact|preference|decision|goal|commitment|open_question|outcome|entity_relation|episode
    is_belief           boolean NOT NULL DEFAULT false,
    content             text NOT NULL,
    subj                text,
    pred                text,
    obj                 text,
    embedding           vector(384) NOT NULL,
    tsv                 tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    encoding_context    jsonb NOT NULL DEFAULT '{}',
    importance          real NOT NULL DEFAULT 0.5,
    confidence          real NOT NULL DEFAULT 0.7,
    source_reliability  real NOT NULL DEFAULT 0.9,
    salience            real NOT NULL DEFAULT 0.5,
    stability           real NOT NULL DEFAULT 10,       -- days (forgetting-curve S)
    last_recalled_at    timestamptz,
    recall_count        integer NOT NULL DEFAULT 0,
    utility             real NOT NULL DEFAULT 0,
    observed_count      integer NOT NULL DEFAULT 1,
    status              text NOT NULL DEFAULT 'active', -- active | weakening | archived
    valid_from          timestamptz NOT NULL DEFAULT now(),
    valid_until         timestamptz,
    superseded_by       uuid,
    thread_status       text,                           -- open | closed (commitments / open questions)
    due_at              timestamptz,
    version             integer NOT NULL DEFAULT 1,
    metadata            jsonb NOT NULL DEFAULT '{}',
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);
-- Per-project exact search: btree-filter then exact cosine (small sets, perfect recall).
CREATE INDEX memories_project_idx ON memories (tenant_id, project_id, status);
CREATE INDEX memories_user_scope_idx ON memories (tenant_id, user_id, scope, status);
CREATE INDEX memories_spo_idx ON memories (project_id, subj, pred) WHERE valid_until IS NULL;
CREATE INDEX memories_tsv_idx ON memories USING gin (tsv);
CREATE INDEX memories_source_turn_idx ON memories (source_turn_id);
CREATE INDEX memories_open_idx ON memories (project_id) WHERE thread_status = 'open';

CREATE TABLE memory_versions (
    id          bigserial PRIMARY KEY,
    tenant_id   uuid NOT NULL,
    memory_id   uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    version     integer NOT NULL,
    content     text NOT NULL,
    op          text NOT NULL,
    turn_id     uuid,
    reason      text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX memory_versions_mem_idx ON memory_versions (memory_id, version);

CREATE TABLE entities (
    id             uuid PRIMARY KEY,
    tenant_id      uuid NOT NULL,
    project_id     uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name           text NOT NULL,
    norm_name      text NOT NULL,
    type           text NOT NULL DEFAULT 'thing',
    aliases        text[] NOT NULL DEFAULT '{}',
    mention_count  integer NOT NULL DEFAULT 1,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (project_id, norm_name)
);

CREATE TABLE edges (
    id           uuid PRIMARY KEY,
    tenant_id    uuid NOT NULL,
    project_id   uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    src_entity   uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst_entity   uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation     text NOT NULL,
    memory_id    uuid REFERENCES memories(id) ON DELETE SET NULL,
    weight       real NOT NULL DEFAULT 0.5,
    valid_from   timestamptz NOT NULL DEFAULT now(),
    valid_until  timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (project_id, src_entity, dst_entity, relation)
);
CREATE INDEX edges_project_idx ON edges (project_id);

CREATE TABLE memory_entities (
    memory_id  uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id  uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    tenant_id  uuid NOT NULL,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX memory_entities_entity_idx ON memory_entities (entity_id);

CREATE TABLE episodes (
    id            uuid PRIMARY KEY,
    tenant_id     uuid NOT NULL,
    project_id    uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chat_id       uuid NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    memory_id     uuid REFERENCES memories(id) ON DELETE SET NULL,
    title         text NOT NULL,
    gist          text NOT NULL,
    decisions     jsonb NOT NULL DEFAULT '[]',
    open_threads  jsonb NOT NULL DEFAULT '[]',
    turn_from     integer NOT NULL,
    turn_to       integer NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX episodes_chat_idx ON episodes (chat_id, turn_to DESC);

CREATE TABLE provenance (
    id          bigserial PRIMARY KEY,
    tenant_id   uuid NOT NULL,
    memory_id   uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    op          text NOT NULL,
    turn_id     uuid,
    chat_id     uuid,
    speaker     text,
    evidence    text,
    method      text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX provenance_mem_idx ON provenance (memory_id, created_at);

CREATE TABLE retrieval_traces (
    id          uuid PRIMARY KEY,
    tenant_id   uuid NOT NULL,
    project_id  uuid,
    chat_id     uuid,
    query       text NOT NULL,
    gate        text NOT NULL,
    cache_hit   boolean NOT NULL DEFAULT false,
    agent_used  boolean NOT NULL DEFAULT false,
    candidates  jsonb NOT NULL DEFAULT '[]',
    selected    uuid[] NOT NULL DEFAULT '{}',
    latency     jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX retrieval_traces_chat_idx ON retrieval_traces (chat_id, created_at DESC);
"""


def upgrade() -> None:
    # asyncpg runs one statement per call
    for stmt in SCHEMA.split(";\n"):
        if stmt.strip():
            op.execute(stmt)
    for table in RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid) "
            f"WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)"
        )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO memory_app")
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO memory_app")


def downgrade() -> None:
    for table in [
        "retrieval_traces", "provenance", "episodes", "memory_entities", "edges", "entities",
        "memory_versions", "memories", "turns", "chats", "projects", "users",
        "api_keys", "dashboard_users", "tenants",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
