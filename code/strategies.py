"""
linkage strategies operating on in-memory data.

v32_core is a faithful reimplementation of the 4-pass greedy linkage in
the v3.2 baseline pipeline (not included): same keys, same greedy
first-unused-candidate assignment, same pass order, same index built ONLY on
the current-year SINASC (including its cross-year asymmetry, which is part of
what the paper measures). It additionally records which SINASC row each SIM
record matched, so precision/recall can be computed against a known truth.

Strategies S1-S6 for the benchmark are added here as they are implemented.
"""

from __future__ import annotations

import polars as pl


def build_v32_indexes(sinasc: pl.DataFrame):
    """Key -> list of row indexes, replicated from v3.2 (cast to Utf8, '_' join).

    Null fields become the literal string 'None' in v3.2's f-string keys on the
    SIM side, but on the SINASC side pl.concat_str yields null for null inputs
    (key absent from the index). Replicate exactly: use Python f-strings on
    both sides so behaviour matches the SIM side, and skip keys where any
    component is null on the SINASC side (concat_str null semantics).
    """
    cols = ["DTNASC", "SEXO", "CODMUNNASC", "CODESTAB", "PESO", "IDADEMAE"]
    present = [c for c in cols if c in sinasc.columns]
    df = sinasc.select([pl.col(c).cast(pl.Utf8) for c in present])

    idx_mun_peso: dict[str, list[int]] = {}
    idx_estab_peso: dict[str, list[int]] = {}
    idx_mun_idademae: dict[str, list[int]] = {}

    for i, row in enumerate(df.iter_rows(named=True)):
        dt, sx = row.get("DTNASC"), row.get("SEXO")
        mun, est = row.get("CODMUNNASC"), row.get("CODESTAB")
        peso, idm = row.get("PESO"), row.get("IDADEMAE")
        # concat_str: null if any component is null -> key not present
        if None not in (dt, sx, mun, peso):
            idx_mun_peso.setdefault(f"{dt}_{sx}_{mun}_{peso}", []).append(i)
        if None not in (dt, est, peso):
            idx_estab_peso.setdefault(f"{dt}_{est}_{peso}", []).append(i)
        if None not in (dt, sx, mun, idm):
            idx_mun_idademae.setdefault(f"{dt}_{sx}_{mun}_{idm}", []).append(i)

    return idx_mun_peso, idx_estab_peso, idx_mun_idademae


def v32_core(sinasc: pl.DataFrame, sim_rows: list[dict]):
    """Run the 4 v3.2 passes. Returns (matches, pass_counts) where matches is
    {sim_idx: (sinasc_row_idx, pass_id)}."""
    idx_mp, idx_ep, idx_mi = build_v32_indexes(sinasc)

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

    # Pass 1: DTNASC + SEXO + CODMUNNATU + PESO
    for i, r in enumerate(sim_rows):
        try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNNATU')}_{g(r,'PESO')}", idx_mp, 1)
    # Pass 2: DTNASC + SEXO + CODMUNOCOR + PESO
    for i, r in enumerate(sim_rows):
        if i not in matches:
            try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNOCOR')}_{g(r,'PESO')}", idx_mp, 2)
    # Pass 3: DTNASC + CODESTAB + PESO
    for i, r in enumerate(sim_rows):
        if i not in matches:
            try_key(i, f"{g(r,'DTNASC')}_{g(r,'CODESTAB')}_{g(r,'PESO')}", idx_ep, 3)
    # Pass 4: DTNASC + SEXO + CODMUN + IDADEMAE (NATU then OCOR)
    for i, r in enumerate(sim_rows):
        if i not in matches:
            if not try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNNATU')}_{g(r,'IDADEMAE')}", idx_mi, 4):
                try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNOCOR')}_{g(r,'IDADEMAE')}", idx_mi, 4)

    pass_counts = {p: sum(1 for _, pid in matches.values() if pid == p) for p in (1, 2, 3, 4)}
    return matches, pass_counts


SIM_LINK_COLS = ["DTNASC", "SEXO", "CODMUNNATU", "CODMUNOCOR", "CODESTAB",
                 "PESO", "IDADEMAE", "DTOBITO", "IDADE_DIAS", "CAUSABAS"]


def prepare_sim_rows(df: pl.DataFrame) -> list[dict]:
    """Replicates v3.2's preparar_sim: select linkage columns, cast all to Utf8."""
    cols = [c for c in SIM_LINK_COLS if c in df.columns]
    df = df.select(cols)
    df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in df.columns])
    return df.to_dicts()
