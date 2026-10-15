"""
Revision helpers. Imports the frozen code in ../ read-only and
writes ONLY under outputs/revision/.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import OUTPUTS  # noqa: E402

REV = OUTPUTS / "revision"
REV.mkdir(parents=True, exist_ok=True)


def metrics(matches: dict, sim_rows: list[dict], true_row: dict, no_pair: dict) -> dict:
    """Same definitions as eval_v4 / v4ml.evaluate / strategies_bench.match_metrics:
    precision = correct/matched, recall = correct/pairable deaths."""
    n_pairable = sum(1 for v in no_pair.values() if v == 0)
    correct = sum(1 for i, (j, _) in matches.items()
                  if true_row[int(sim_rows[i]["SYNTH_ID"])] == j)
    nm = len(matches)
    p = correct / nm if nm else 0.0
    r = correct / n_pairable
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"matched": nm, "correct": correct, "wrong": nm - correct,
            "n": len(sim_rows), "n_pairable": n_pairable,
            "precision": p, "recall": r, "f1": f1}


def greedy_best_score(pairs, scores, tau: float, pid: int = 7) -> dict:
    """Greedy 1:1: pairs sorted by descending score (ties: sim idx, cand id),
    accept if score >= tau and both sides unused."""
    order = sorted(range(len(pairs)),
                   key=lambda k: (-float(scores[k]), pairs[k][0], pairs[k][1]))
    used_s, used_c, acc = set(), set(), {}
    for k in order:
        s = float(scores[k])
        if s < tau:
            break
        i, c = pairs[k][0], pairs[k][1]
        if i in used_s or c in used_c:
            continue
        used_s.add(i); used_c.add(c)
        acc[i] = (c, pid)
    return acc


def fmt(m: dict) -> str:
    return (f"matched {m['matched']:,} wrong {m['wrong']:,} "
            f"P {m['precision']*100:.1f} R {m['recall']*100:.1f} F1 {m['f1']*100:.1f}")
