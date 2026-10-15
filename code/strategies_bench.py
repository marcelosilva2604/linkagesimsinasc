"""
Phase 3: benchmark strategies S1-S6.

Key-join strategies are implemented AS PUBLISHED (S1-S3 have no one-to-one
control: a death matches every birth sharing its key). Metrics therefore
distinguish pair-level precision from death-level recall, and report the
many-to-many rate the original pipelines silently carry.

  S1 labsus-style        DTNASC + municipality (occurrence, fallback natu)
  S2 fabianofilho-style  DTNASC + SEXO + PESO (no municipality)
  S3 GOPAD/JOSIF-style   single strict pass: DTNASC + SEXO + CODMUNNATU +
                         PESO + SEMAGESTAC + ESCMAE, all exact non-null
  S4 internal v1         v3.2 Pass 1 only (greedy 1:1)
  S5 internal v2         v3.2 Passes 1-2 (greedy 1:1)
  S6 Fellegi-Sunter      splink EM-trained probabilistic model (separate
                         module bench_splink.py), greedy 1:1 on match weight
  S7 v3.2 / S8 v4.2 come from strategies.py / v4ml.py.
"""

from __future__ import annotations

import polars as pl

from strategies import v32_core


def _key_join(sinasc_keys: pl.DataFrame, sim_keys: pl.DataFrame):
    """Inner-join on KEY. Returns pair frame (SIM_IDX, ROW)."""
    return sim_keys.join(sinasc_keys, on="KEY", how="inner").select(["SIM_IDX", "ROW"])


def _sinasc_key(sinasc: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    df = sinasc.select([pl.col(c).cast(pl.Utf8) for c in cols if c in sinasc.columns])
    df = df.with_row_index("ROW")
    key = pl.concat_str([pl.col(c) for c in cols if c in sinasc.columns], separator="_")
    return df.with_columns(key.alias("KEY")).drop_nulls("KEY").select(["ROW", "KEY"])


def _sim_key(sim_rows: list[dict], fields: list[str]) -> pl.DataFrame:
    out = []
    for i, r in enumerate(sim_rows):
        vals = [r.get(f) for f in fields]
        if None in vals or "" in vals:
            continue
        out.append({"SIM_IDX": i, "KEY": "_".join(str(v) for v in vals)})
    return pl.DataFrame(out, schema={"SIM_IDX": pl.Int64, "KEY": pl.Utf8})


def s1_labsus(sinasc: pl.DataFrame, sim_rows: list[dict]):
    """DTNASC + municipality; SIM uses occurrence municipality (fallback natu)."""
    sk = _sinasc_key(sinasc, ["DTNASC", "CODMUNNASC"])
    rows = []
    for i, r in enumerate(sim_rows):
        mun = r.get("CODMUNOCOR") or r.get("CODMUNNATU")
        if r.get("DTNASC") and mun:
            rows.append({"SIM_IDX": i, "KEY": f"{r['DTNASC']}_{mun}"})
    mk = pl.DataFrame(rows, schema={"SIM_IDX": pl.Int64, "KEY": pl.Utf8})
    return _key_join(sk, mk)


def s2_fabiano(sinasc: pl.DataFrame, sim_rows: list[dict]):
    sk = _sinasc_key(sinasc, ["DTNASC", "SEXO", "PESO"])
    mk = _sim_key(sim_rows, ["DTNASC", "SEXO", "PESO"])
    return _key_join(sk, mk)


def s3_strict(sinasc: pl.DataFrame, sim_rows: list[dict]):
    sk = _sinasc_key(sinasc, ["DTNASC", "SEXO", "CODMUNNASC", "PESO", "SEMAGESTAC", "ESCMAE"])
    rows = []
    for i, r in enumerate(sim_rows):
        vals = [r.get("DTNASC"), r.get("SEXO"), r.get("CODMUNNATU"),
                r.get("PESO"), r.get("SEMAGESTAC"), r.get("ESCMAE")]
        if None not in vals and "" not in vals:
            rows.append({"SIM_IDX": i, "KEY": "_".join(str(v) for v in vals)})
    mk = pl.DataFrame(rows, schema={"SIM_IDX": pl.Int64, "KEY": pl.Utf8})
    return _key_join(sk, mk)


def s4_v1(sinasc: pl.DataFrame, sim_rows: list[dict]):
    """v1 = Pass 1 only, greedy 1:1."""
    matches, _ = v32_core(sinasc, sim_rows)
    return {i: m for i, m in matches.items() if m[1] == 1}


def s5_v2(sinasc: pl.DataFrame, sim_rows: list[dict]):
    matches, _ = v32_core(sinasc, sim_rows)
    return {i: m for i, m in matches.items() if m[1] in (1, 2)}


def pair_metrics(pairs: pl.DataFrame, sim_rows: list[dict], true_row: dict,
                 no_pair: dict, name: str) -> dict:
    """Metrics for M:N join strategies (S1-S3)."""
    n = len(sim_rows)
    n_pairable = sum(1 for v in no_pair.values() if v == 0)
    n_pairs = len(pairs)

    truth = [true_row[int(r["SYNTH_ID"])] for r in sim_rows]
    correct_pairs = 0
    deaths_with_match: set[int] = set()
    deaths_hit: set[int] = set()
    per_death: dict[int, int] = {}
    for i, row_id in pairs.iter_rows():
        deaths_with_match.add(i)
        per_death[i] = per_death.get(i, 0) + 1
        if truth[i] == row_id:
            correct_pairs += 1
            deaths_hit.add(i)

    m2m = sum(1 for v in per_death.values() if v > 1)
    prec_pairs = correct_pairs / n_pairs if n_pairs else 0.0
    recall = len(deaths_hit) / n_pairable
    return {
        "strategy": name, "pairs": n_pairs,
        "deaths_matched": len(deaths_with_match),
        "match_rate": len(deaths_with_match) / n,
        "m2m_deaths": m2m, "m2m_rate": m2m / n,
        "precision_pairs": prec_pairs,
        "recall_pairable": recall,
        "f1": (2 * prec_pairs * recall / (prec_pairs + recall)
               if prec_pairs + recall else 0.0),
    }


def match_metrics(matches: dict, sim_rows: list[dict], true_row: dict,
                  no_pair: dict, name: str) -> dict:
    """Metrics for 1:1 strategies (S4, S5, S6...)."""
    n = len(sim_rows)
    n_pairable = sum(1 for v in no_pair.values() if v == 0)
    truth = [true_row[int(r["SYNTH_ID"])] for r in sim_rows]
    correct = sum(1 for i, (j, _) in matches.items() if truth[i] == j)
    nm = len(matches)
    prec = correct / nm if nm else 0.0
    rec = correct / n_pairable
    return {
        "strategy": name, "pairs": nm, "deaths_matched": nm,
        "match_rate": nm / n, "m2m_deaths": 0, "m2m_rate": 0.0,
        "precision_pairs": prec, "recall_pairable": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }
