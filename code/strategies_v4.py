"""
v4 linkage: candidate generation + staged assignment.

Architecture (ablation stages for the paper):
  v4.0  exact passes as v3.2 PLUS (a) cross-year index: December births of the
        previous year are indexed alongside the current year, fixing v3.2's
        asymmetry where those deaths could never match; (b) tolerant recovery
        pass for residuals (DTNASC +/-1 day or day-digit transposition, PESO
        +/-10 g or rounding to 50/100 g, IDADEMAE +/-1, CODMUNNATU/OCOR
        interchangeable), accepted only when the candidate is UNIQUE.
  v4.1  + XGBoost pair classifier scoring the candidates (trained on the
        synthetic reference of 2014, where the true pairs are known).
  v4.2  + globally optimal 1:1 assignment (Hungarian) within blocks instead
        of greedy first-match.

Row identity: current-year SINASC rows keep their index i; previous-December
rows are encoded as PREV_OFFSET + j (j = row position within the
December-filtered previous-year frame, same loader order as the generator).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

PREV_OFFSET = 100_000_000

SINASC_COLS = ["DTNASC", "SEXO", "CODMUNNASC", "CODESTAB", "PESO", "IDADEMAE"]


def sinasc_records(sinasc: pl.DataFrame, offset: int = 0) -> list[tuple]:
    """(row_id, dt, sexo, mun, estab, peso, idademae) tuples, all str-or-None."""
    present = [c for c in SINASC_COLS if c in sinasc.columns]
    df = sinasc.select([pl.col(c).cast(pl.Utf8) for c in present])
    out = []
    for i, row in enumerate(df.iter_rows(named=True)):
        out.append((
            offset + i,
            row.get("DTNASC"), row.get("SEXO"), row.get("CODMUNNASC"),
            row.get("CODESTAB"), row.get("PESO"), row.get("IDADEMAE"),
        ))
    return out


def build_indexes(records: list[tuple]):
    """Exact-key indexes (same key shapes as v3.2) over the given records."""
    idx_mun_peso: dict[str, list[int]] = {}
    idx_estab_peso: dict[str, list[int]] = {}
    idx_mun_idademae: dict[str, list[int]] = {}
    for rid, dt, sx, mun, est, peso, idm in records:
        if None not in (dt, sx, mun, peso):
            idx_mun_peso.setdefault(f"{dt}_{sx}_{mun}_{peso}", []).append(rid)
        if None not in (dt, est, peso):
            idx_estab_peso.setdefault(f"{dt}_{est}_{peso}", []).append(rid)
        if None not in (dt, sx, mun, idm):
            idx_mun_idademae.setdefault(f"{dt}_{sx}_{mun}_{idm}", []).append(rid)
    return idx_mun_peso, idx_estab_peso, idx_mun_idademae


def date_variants(dt: str | None) -> list[str]:
    if not dt:
        return []
    out = [dt]
    try:
        d = datetime.strptime(dt, "%d%m%Y")
        out.append((d + timedelta(days=1)).strftime("%d%m%Y"))
        out.append((d - timedelta(days=1)).strftime("%d%m%Y"))
    except ValueError:
        return out
    swapped = dt[1] + dt[0] + dt[2:]
    if swapped != dt:
        try:
            datetime.strptime(swapped, "%d%m%Y")
            out.append(swapped)
        except ValueError:
            pass
    return out


def peso_variants(peso: str | None) -> list[str]:
    if peso is None:
        return []
    try:
        w = int(float(peso))
    except ValueError:
        return [peso]
    vs = {str(w), str(w + 10), str(w - 10),
          str(int(round(w / 50.0) * 50)), str(int(round(w / 100.0) * 100))}
    return list(vs)


def idade_variants(idm: str | None) -> list[str]:
    if idm is None:
        return []
    try:
        a = int(float(idm))
    except ValueError:
        return [idm]
    return [str(a), str(a + 1), str(a - 1)]


def sexo_variants(sx: str | None) -> list[str]:
    return [sx] if sx in ("1", "2") else ["1", "2"]


def gen_candidates(r: dict, idx_mp, idx_ep, idx_mi) -> set[int]:
    """Tolerant candidate set for one SIM record via variant-key probing."""
    dts = date_variants(r.get("DTNASC"))
    sxs = sexo_variants(r.get("SEXO"))
    muns = [m for m in {r.get("CODMUNNATU"), r.get("CODMUNOCOR")} if m]
    pesos = peso_variants(r.get("PESO"))
    idms = idade_variants(r.get("IDADEMAE"))
    est = r.get("CODESTAB")

    cands: set[int] = set()
    for dt in dts:
        for peso in pesos:
            for sx in sxs:
                for mun in muns:
                    cands.update(idx_mp.get(f"{dt}_{sx}_{mun}_{peso}", ()))
            if est:
                cands.update(idx_ep.get(f"{dt}_{est}_{peso}", ()))
        for sx in sxs:
            for mun in muns:
                for idm in idms:
                    cands.update(idx_mi.get(f"{dt}_{sx}_{mun}_{idm}", ()))
    return cands


def v4_core(sinasc_cur: pl.DataFrame, sinasc_prev_dec: pl.DataFrame | None,
            sim_rows: list[dict], stage: str = "v4.0"):
    """Returns (matches, pass_counts, ambiguous) where matches maps
    sim_idx -> (row_id, pass_id); pass_id 1-4 as v3.2, 5 = tolerant unique.
    `ambiguous` maps sim_idx -> candidate row_id set (unresolved at v4.0)."""
    records = sinasc_records(sinasc_cur)
    if sinasc_prev_dec is not None and len(sinasc_prev_dec):
        records += sinasc_records(sinasc_prev_dec, offset=PREV_OFFSET)
    idx_mp, idx_ep, idx_mi = build_indexes(records)

    used: set[int] = set()
    matches: dict[int, tuple[int, int]] = {}

    def g(row, field):
        v = row.get(field)
        return "" if v is None else str(v)

    def try_key(sim_idx, key, index, pass_id) -> bool:
        if key in index:
            for j in index[key]:
                if j not in used:
                    used.add(j)
                    matches[sim_idx] = (j, pass_id)
                    return True
        return False

    # Exact passes, v3.2 order (now over the cross-year index)
    for i, r in enumerate(sim_rows):
        try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNNATU')}_{g(r,'PESO')}", idx_mp, 1)
    for i, r in enumerate(sim_rows):
        if i not in matches:
            try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNOCOR')}_{g(r,'PESO')}", idx_mp, 2)
    for i, r in enumerate(sim_rows):
        if i not in matches:
            try_key(i, f"{g(r,'DTNASC')}_{g(r,'CODESTAB')}_{g(r,'PESO')}", idx_ep, 3)
    for i, r in enumerate(sim_rows):
        if i not in matches:
            if not try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNNATU')}_{g(r,'IDADEMAE')}", idx_mi, 4):
                try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNOCOR')}_{g(r,'IDADEMAE')}", idx_mi, 4)

    # Pass 5: tolerant recovery, unique unused candidate only
    ambiguous: dict[int, set[int]] = {}
    for i, r in enumerate(sim_rows):
        if i in matches:
            continue
        cands = {c for c in gen_candidates(r, idx_mp, idx_ep, idx_mi) if c not in used}
        if len(cands) == 1:
            j = cands.pop()
            used.add(j)
            matches[i] = (j, 5)
        elif cands:
            ambiguous[i] = cands

    pass_counts = {p: sum(1 for _, pid in matches.values() if pid == p) for p in (1, 2, 3, 4, 5)}
    return matches, pass_counts, ambiguous
