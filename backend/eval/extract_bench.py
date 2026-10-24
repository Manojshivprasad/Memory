"""Extraction-only model comparison on hard turns (one LLM call per turn per model).

    python -m eval.extract_bench openrouter/openai/gpt-4.1-nano openrouter/google/gemini-2.5-flash-lite ...

Scores: did the ADD ops capture each expected detail (keyword groups), and did anything get stored for turns
that should produce nothing.
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

CASES: list[tuple[str, str, list[list[str]]]] = [
    ("My novel is called The Glass Tide. Set in 1920s Cornwall.", "Evocative setting!", [["glass tide"], ["cornwall"]]),
    ("The protagonist is Elena, a lighthouse keeper. The antagonist is Mr. Voss, a smuggler.", "Great conflict.",
     [["elena"], ["voss"]]),
    ("Chapter 3 should end on a cliffhanger in the storm.", "A storm cliffhanger works well.", [["chapter 3"]]),
    ("In general, always keep your answers short and use bullet points, for every project.",
     "Will do: short, bulleted answers.", [["bullet", "short", "concise"]]),
    ("He gets distracted easily, so keep problems short, max 3 per set.", "Short sets of 3, got it.", [["3", "three"]]),
    ("His teacher Mrs. Rao said his board exam is in March.", "Noted.", [["rao"], ["march"]]),
    ("We definitely want to see the temples in Kyoto. We hate long bus rides.", "Kyoto temples, trains.",
     [["kyoto"], ["bus"]]),
    ("She loves pottery. Maybe a pottery class?", "A pottery class is a thoughtful gift.", [["pottery"]]),
    ("Alice owns the payments service and Bob handles DevOps.", "Noted.", [["alice"], ["bob"]]),
    ("My monthly budget is 60,000 INR and I'm saving for a car, target 8 lakh by December 2027.", "Let's plan.",
     [["60,000", "60000"], ["car"]]),
    # should store nothing
    ("haha nice, thanks", "Anytime!", []),
    ("what's the capital of Peru?", "Lima.", []),
]


async def bench(model: str) -> dict:
    os.environ["MEMORY_LLM_MODEL"] = model
    from app.config import settings

    settings.llm_model = model
    from app.write.extractor import extract_llm
    from app.write.ops import validate

    now = datetime.now(timezone.utc)
    hits = total = noise = 0
    misses = []
    t0 = time.perf_counter()
    for u, a, expect in CASES:
        ops = validate(await extract_llm(u, a, [], "", "", now) or [], set(), source=u)
        text = " ".join(o.content for o in ops if o.op == "ADD").lower()
        if not expect:
            noise += bool(text)
            continue
        for g in expect:
            total += 1
            if any(x in text for x in g):
                hits += 1
            else:
                misses.append(f"{g[0]} <- {u[:40]}")
    return {"model": model, "recall": round(hits / total, 3), "noise_turns": noise,
            "seconds": round(time.perf_counter() - t0, 1), "misses": misses}


async def main(models: list[str]) -> None:
    from app.gateway.providers import warm

    warm()
    for m in models:
        r = await bench(m)
        print(f"{r['model']:48} recall {r['recall']:.2f}  noise {r['noise_turns']}  {r['seconds']}s  misses={r['misses']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
