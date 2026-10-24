"""Cheap text utilities shared by the hot path and the write path (no models, no I/O)."""

import re

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+#-]*")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    let me more most my myself no nor not now of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves ok okay yes yeah please thanks thank hi hello hey
    also get got go going gonna want need like know think one thing things something anything tell us
    did does done do can could would should shall may might must""".split()
)


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.+#-]", " ", name.lower())).strip()


def words(text: str) -> list[str]:
    return [w.lower().strip(".-") for w in _WORD.findall(text)]


def stem(w: str) -> str:
    """Tiny suffix stripper: good enough for overlap scoring, not linguistics."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_tokens(text: str) -> list[str]:
    return [stem(w) for w in words(text) if w and w not in STOPWORDS and len(w) > 1]


def is_identifier(tok: str) -> bool:
    return any(c.isdigit() for c in tok) or "_" in tok or "." in tok


def tsquery_or(text: str, limit: int = 12, salient_only: bool = False) -> str:
    """Build an OR tsquery from content words; only [a-z0-9] survive, so it is injection-safe.

    ``salient_only`` keeps identifiers, proper nouns and longer words: the lexical channel is for
    exact names/codes, and common short words would make Postgres rank most of the project.
    """
    toks: list[str] = []
    for raw in _WORD.findall(text):
        w = raw.lower().strip(".-")
        if w in STOPWORDS:
            continue
        if salient_only and not (is_identifier(w) or raw[:1].isupper() or len(w) >= 5):
            continue
        w = re.sub(r"[^a-z0-9]", "", w)
        if len(w) > 1 and w not in toks:
            toks.append(w)
        if len(toks) >= limit:
            break
    return " | ".join(toks)


# Small technology taxonomy: lets the heuristic extractor produce *functional* predicates
# (a project has one primary database) so contradictions ("we switched to MySQL") resolve.
TECH_TAXONOMY: dict[str, tuple[str, str]] = {}
for _cat, _names in {
    "database": "postgres postgresql mysql mariadb mongodb sqlite dynamodb cassandra cockroachdb supabase firestore",
    "framework": "fastapi django flask express nestjs next.js nextjs react vue angular svelte spring rails laravel",
    "language": "python typescript javascript go golang rust java kotlin ruby php c# csharp swift",
    "cloud": "aws gcp azure vercel heroku netlify cloudflare",
    "cache": "redis memcached",
    "queue": "kafka rabbitmq sqs celery",
}.items():
    for _n in _names.split():
        TECH_TAXONOMY[_n] = (_cat, _n)
_CANON = {"postgresql": "postgres", "golang": "go", "nextjs": "next.js", "csharp": "c#"}


def canonical_entity(name: str) -> str:
    """Entity resolution key: known technologies collapse to one name (PostgreSQL == Postgres)."""
    norm = normalize(name)
    hit = TECH_TAXONOMY.get(norm)
    return _CANON.get(hit[1], hit[1]) if hit else norm


def detect_tech(text: str) -> list[tuple[str, str]]:
    """Return (category, canonical_name) for technologies mentioned in ``text``."""
    out: list[tuple[str, str]] = []
    for w in words(text):
        hit = TECH_TAXONOMY.get(w)
        if hit:
            item = (hit[0], _CANON.get(hit[1], hit[1]))
            if item not in out:
                out.append(item)
    return out


_FP_RULES = [
    (re.compile(r"\b(?:the )?user's\b", re.I), "my"),
    (re.compile(r"\b(?:the )?user is\b", re.I), "I am"),
    (re.compile(r"\b(?:the )?user was\b", re.I), "I was"),
    (re.compile(r"\b(?:the )?user has\b", re.I), "I have"),
    (re.compile(r"\b(?:the )?user does\b", re.I), "I do"),
    (re.compile(r"\b(?:the )?user\b", re.I), "I"),
    (re.compile(r"\bthe project's\b", re.I), "our project's"),
    (re.compile(r"\bthe project\b", re.I), "our project"),
]
_FP_VERB = re.compile(r"\bI ([a-z]+?)(ches|shes|sses|xes|s)\b")


def embedding_text(content: str) -> str:
    """Text used to EMBED a memory: first-person perspective ("The user lives in X" -> "I live in X").

    Users ask in the first person ("where do I live?"); embedding memories the same way raises the
    similarity of the right memory (measured 0.44 -> 0.60) without lifting unrelated ones. Display text
    is unchanged.
    """
    s = content
    for rx, rep in _FP_RULES:
        s = rx.sub(rep, s)

    def verb(m: re.Match) -> str:
        stem_, suf = m.group(1), m.group(2)
        if stem_ + suf in ("is", "was", "has", "does", "this", "us"):
            return m.group(0)
        return f"I {stem_}{suf[:-2] if suf in ('ches', 'shes', 'sses', 'xes') else ''}" if suf != "s" else f"I {stem_}"

    return _FP_VERB.sub(verb, s)


def first_sentence(text: str, max_chars: int = 160) -> str:
    s = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return s[:max_chars]


def split_clauses(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+|;\s*", text)
    return [p.strip() for p in parts if p and p.strip()]
