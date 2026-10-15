"""
Synthetic reference with known pairs for SINASC-SIM linkage.

Builds a simulated "SIM neonatal" file for a given year by sampling true
SINASC records (stratified to the real SIM birthweight distribution, with the
measured share of deaths born in December of the previous year) and
perturbing the linkage fields with an error model in two parts:

- MISSINGNESS: measured directly per year from the real SIM neonatal file
  (outputs/calibration/error_model.json) - identical across tiers.
- DISCREPANCIES (typos, rounding, municipality swaps, age +/-1): not directly
  observable without identified data; the calibrated tier is tuned so that
  running the frozen v3.2 strategy on the synthetic set reproduces the real
  per-pass match profile. Optimistic = 0.5x, pessimistic = 2x those rates.

Outputs (per year, per tier) under outputs/synthetic/:
  synthetic_SIM_{year}_{tier}.csv  - simulated SIM records (linkage columns)
  truth_{year}_{tier}.csv          - synth_id -> true SINASC row index
                                     (row index in the current-year SINASC as
                                     loaded by common.load_sinasc;
                                     100_000_000 + row-in-December-subset when
                                     born in December of year-1; -2 when the
                                     birth was never registered - no pair)

Usage: python3 make_synthetic_reference.py <year> [tier] [seed]
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta

import polars as pl

from common import OUTPUTS, load_sinasc, load_sim_neonatal

# Discrepancy rates, calibrated tier (tuned against the real v3.2 pass profile)
DISCREPANCY = {
    "date_pm1": 0.005,       # DTNASC off by +/-1 day
    "date_transpose": 0.003, # day digits transposed
    "sexo_flip": 0.003,      # sex recorded as the opposite
    "peso_round100": 0.030,  # weight rounded to nearest 100 g
    "peso_round50": 0.010,   # weight rounded to nearest 50 g
    "peso_pm10": 0.010,      # weight off by +/-10 g
    "peso_gross": 0.015,     # gross weight typo (one digit wrong)
    "age_gross": 0.010,      # gross mother-age typo (one digit wrong)
    "mun_disc": 0.034,       # CODMUNNATU is not the SINASC birth municipality
    "ocor_away": 0.150,      # death occurred in another municipality (referral)
    "estab_disc": 0.200,  # neonatal transfer: death facility != birth facility     # CODESTAB differs from SINASC
    "age_pm1": 0.035,        # mother age off by +/-1 year
    "gest_pm1": 0.100,       # gestational weeks off by +/-1
    "escmae_shift": 0.050,   # schooling category off by one
}
# Correlated corruption: a "messy" record has all discrepancy rates multiplied,
# reproducing the multi-field failures that leave real deaths unmatched.
P_MESSY = 0.060
MESSY_MULT = 5.0
MESSY_MISS_MULT = 2.0   # messy records also have inflated missingness
P_IDM_MISS_GIVEN_PESO_MISS = 0.35  # correlated form quality: extra IDADEMAE
                                   # missingness when PESO ends up missing
# Deaths whose birth was never registered in SINASC (no true pair exists).
# Simulated from previous-year Jan-Nov records with the year rewritten, so the
# field combination is plausible but the exact record is absent from the index.
P_UNREG = 0.045
TIER_MULT = {"optimistic": 0.5, "calibrated": 1.0, "pessimistic": 2.0}


def load_missingness(year: int) -> dict:
    with open(OUTPUTS / "calibration" / "error_model.json") as f:
        em = json.load(f)
    # measured rates fall between sentinel years; take nearest sentinel
    sentinel = min(em.keys(), key=lambda y: abs(int(y) - year))
    sim = em[sentinel]["SIM_neonatal"]
    return {
        "peso": sim["PESO"]["missing"],
        "idademae": sim["IDADEMAE"]["missing"],
        "codestab": sim["CODESTAB"]["missing"],
        "codmunnatu": sim["CODMUNNATU"]["missing"],
        "sexo_indet": sim["SEXO"]["indeterminate_or_other"],
        "semagestac": sim["SEMAGESTAC"]["missing"],
        "escmae": sim["ESCMAE"]["missing"],
    }


def perturb_date(dt: str, rng: random.Random, r: dict) -> str:
    try:
        d = datetime.strptime(dt, "%d%m%Y")
    except ValueError:
        return dt
    u = rng.random()
    if u < r["date_pm1"]:
        d2 = d + timedelta(days=rng.choice([-1, 1]))
        return d2.strftime("%d%m%Y")
    if u < r["date_pm1"] + r["date_transpose"]:
        dd = dt[:2]
        swapped = dd[1] + dd[0]
        cand = swapped + dt[2:]
        try:
            datetime.strptime(cand, "%d%m%Y")
            return cand
        except ValueError:
            return dt
    return dt


def digit_typo(value: int, rng: random.Random) -> int:
    digits = list(str(value))
    pos = rng.randrange(len(digits))
    choices = [d for d in "0123456789" if d != digits[pos]]
    if pos == 0:
        choices = [d for d in choices if d != "0"]
    digits[pos] = rng.choice(choices)
    return int("".join(digits))


def perturb_weight(peso, rng: random.Random, r: dict, miss: float):
    if peso is None:
        return None
    if rng.random() < miss:
        return None
    u = rng.random()
    if u < r["peso_gross"]:
        return digit_typo(peso, rng)
    u = rng.random()
    if u < r["peso_round100"]:
        return int(round(peso / 100.0) * 100)
    if u < r["peso_round100"] + r["peso_round50"]:
        return int(round(peso / 50.0) * 50)
    if u < r["peso_round100"] + r["peso_round50"] + r["peso_pm10"]:
        return max(1, peso + rng.choice([-10, 10]))
    return peso


def main():
    year = int(sys.argv[1])
    tier = sys.argv[2] if len(sys.argv) > 2 else "calibrated"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    mult = TIER_MULT[tier]
    rng = random.Random(seed)

    r = {k: min(v * mult, 0.95) for k, v in DISCREPANCY.items()}
    miss = load_missingness(year)

    # --- real SIM empirical distributions -------------------------------
    sim_real = load_sim_neonatal(year)
    n_deaths = len(sim_real)
    dt_years = sim_real["DTNASC"].str.slice(4, 4)
    frac_prev = (dt_years != str(year)).sum() / n_deaths
    idade_dias_pool = [
        int(v) for v in sim_real["IDADE_DIAS"].cast(pl.Int64, strict=False).drop_nulls().to_list()
    ]
    causabas_pool = sim_real["CAUSABAS"].drop_nulls().to_list()
    peso_real = sim_real["PESO"].cast(pl.Int64, strict=False)
    peso_valid = peso_real.filter((peso_real >= 200) & (peso_real <= 8000))
    bins_real = (peso_valid // 250).value_counts()
    bin_dist = {int(b): c / len(peso_valid) for b, c in bins_real.iter_rows()}

    # --- SINASC pools ----------------------------------------------------
    cols = ["DTNASC", "SEXO", "CODMUNNASC", "CODESTAB", "PESO", "IDADEMAE", "CODMUNRES", "SEMAGESTAC", "ESCMAE"]
    sinasc = load_sinasc(year)
    cur = sinasc.select(
        [pl.col(c).cast(pl.Utf8) for c in cols if c in sinasc.columns]
    ).with_row_index("ROW")
    cur = cur.with_columns(pl.col("PESO").cast(pl.Int64, strict=False).alias("PESO_I"))

    prev_pool: list[dict] = []
    unreg_frame = None
    sinasc_prev = load_sinasc(year - 1)
    if sinasc_prev is not None:
        prev_all = sinasc_prev.select(
            [pl.col(c).cast(pl.Utf8) for c in cols if c in sinasc_prev.columns]
        ).with_columns(pl.col("PESO").cast(pl.Int64, strict=False).alias("PESO_I"))
        prev_pool = (prev_all.filter(pl.col("DTNASC").str.slice(2, 2) == "12")
                     .with_row_index("ROWP").to_dicts())
        # Jan-Nov of the previous year: source for "unregistered birth" deaths
        unreg_frame = prev_all.filter(pl.col("DTNASC").str.slice(2, 2) != "12")

    # referral municipalities: top 5 by births per UF
    top_mun = (
        cur.group_by("CODMUNNASC").len().sort("len", descending=True)
        .with_columns(pl.col("CODMUNNASC").str.slice(0, 2).alias("UF"))
    )
    referral: dict[str, list[str]] = {}
    for m, _, uf in top_mun.iter_rows():
        if m is not None and len(referral.get(uf, [])) < 5:
            referral.setdefault(uf, []).append(m)
    # establishment pool per municipality (for estab discrepancies)
    estab_pool: dict[str, list[str]] = {}
    for m, e in cur.select(["CODMUNNASC", "CODESTAB"]).unique().iter_rows():
        if m is not None and e is not None and len(estab_pool.get(m, [])) < 20:
            estab_pool.setdefault(m, []).append(e)

    # --- sampling --------------------------------------------------------
    p_unreg = min(P_UNREG * mult, 0.5)
    n_prev = round(n_deaths * frac_prev)
    n_unreg = round(n_deaths * p_unreg) if unreg_frame is not None else 0
    n_cur = n_deaths - n_prev - n_unreg

    cur_valid = cur.filter(pl.col("PESO_I").is_between(200, 8000))
    by_bin = {int(b): g for (b,), g in cur_valid.with_columns(
        (pl.col("PESO_I") // 250).alias("BIN")).group_by("BIN")}

    sampled: list[dict] = []
    for b, share in sorted(bin_dist.items()):
        want = round(share * n_cur)
        if b in by_bin and want > 0:
            g = by_bin[b]
            take = min(want, len(g))
            sampled.extend(g.sample(n=take, seed=rng.randrange(2**31)).to_dicts())
    while len(sampled) < n_cur:
        sampled.append(cur_valid.sample(n=1, seed=rng.randrange(2**31)).to_dicts()[0])
    sampled = sampled[:n_cur]

    prev_sampled = [prev_pool[rng.randrange(len(prev_pool))] for _ in range(n_prev)] if prev_pool else []

    unreg_sampled: list[dict] = []
    if n_unreg and unreg_frame is not None:
        for rec in unreg_frame.sample(n=n_unreg, seed=rng.randrange(2**31)).to_dicts():
            rec = dict(rec)
            rec["DTNASC"] = rec["DTNASC"][:4] + str(year)  # rewrite year: plausible, absent
            unreg_sampled.append(rec)

    # --- perturbation ----------------------------------------------------
    n_sampled, n_prev_s = len(sampled), len(prev_sampled)
    synth_rows, truth_rows = [], []
    for k, rec in enumerate(sampled + prev_sampled + unreg_sampled):
        is_prev = n_sampled <= k < n_sampled + n_prev_s
        is_unreg = k >= n_sampled + n_prev_s
        if rng.random() < P_MESSY:
            r_rec = {kk: min(vv * MESSY_MULT, 0.95) for kk, vv in r.items()}
            m_rec = {kk: min(vv * MESSY_MISS_MULT, 0.95) for kk, vv in miss.items()}
        else:
            r_rec = r
            m_rec = miss
        true_mun = rec.get("CODMUNNASC")
        uf = true_mun[:2] if true_mun else None

        dt = perturb_date(rec["DTNASC"], rng, r_rec)

        sexo = rec.get("SEXO")
        u = rng.random()
        if u < m_rec["sexo_indet"]:
            sexo = "0"
        elif u < m_rec["sexo_indet"] + r_rec["sexo_flip"] and sexo in ("1", "2"):
            sexo = "2" if sexo == "1" else "1"

        peso = perturb_weight(rec.get("PESO_I"), rng, r_rec, m_rec["peso"])

        if rng.random() < m_rec["codmunnatu"]:
            natu = None
        elif rng.random() < r_rec["mun_disc"]:
            munres = rec.get("CODMUNRES")
            if munres and munres != true_mun:
                natu = munres
            elif uf and referral.get(uf):
                natu = rng.choice(referral[uf])
            else:
                natu = true_mun
        else:
            natu = true_mun

        if rng.random() < r_rec["ocor_away"] and uf and referral.get(uf):
            ocor = rng.choice([m for m in referral[uf] if m != true_mun] or [true_mun])
        else:
            ocor = true_mun

        estab = rec.get("CODESTAB")
        if rng.random() < m_rec["codestab"]:
            estab = None
        elif rng.random() < r_rec["estab_disc"] and true_mun and estab_pool.get(true_mun):
            alt = [e for e in estab_pool[true_mun] if e != estab]
            if alt:
                estab = rng.choice(alt)

        idm = rec.get("IDADEMAE")
        p_idm_miss = m_rec["idademae"]
        if peso is None and rec.get("PESO_I") is not None:
            p_idm_miss = min(p_idm_miss + P_IDM_MISS_GIVEN_PESO_MISS, 0.95)
        if rng.random() < p_idm_miss:
            idm = None
        elif idm is not None:
            u2 = rng.random()
            if u2 < r_rec["age_gross"]:
                try:
                    idm = str(digit_typo(int(float(idm)), rng))
                except ValueError:
                    pass
            elif u2 < r_rec["age_gross"] + r_rec["age_pm1"]:
                try:
                    idm = str(int(float(idm)) + rng.choice([-1, 1]))
                except ValueError:
                    pass

        gest = rec.get("SEMAGESTAC")
        if rng.random() < m_rec["semagestac"]:
            gest = None
        elif gest is not None and rng.random() < r_rec["gest_pm1"]:
            try:
                gest = str(int(float(gest)) + rng.choice([-1, 1]))
            except ValueError:
                pass

        escmae = rec.get("ESCMAE")
        if rng.random() < m_rec["escmae"]:
            escmae = None
        elif escmae is not None and rng.random() < r_rec["escmae_shift"]:
            try:
                escmae = str(max(1, int(float(escmae)) + rng.choice([-1, 1])))
            except ValueError:
                pass

        idade_dias = rng.choice(idade_dias_pool) if idade_dias_pool else 0
        try:
            dtobito = (datetime.strptime(rec["DTNASC"], "%d%m%Y")
                       + timedelta(days=idade_dias)).strftime("%d%m%Y")
        except ValueError:
            dtobito = None

        synth_rows.append({
            "SYNTH_ID": k,
            "DTNASC": dt, "SEXO": sexo,
            "CODMUNNATU": natu, "CODMUNOCOR": ocor, "CODESTAB": estab,
            "PESO": None if peso is None else str(peso),
            "IDADEMAE": idm,
            "SEMAGESTAC": gest, "ESCMAE": escmae,
            "DTOBITO": dtobito, "IDADE_DIAS": str(idade_dias),
            "CAUSABAS": rng.choice(causabas_pool) if causabas_pool else None,
        })
        truth_rows.append({
            "SYNTH_ID": k,
            "TRUE_ROW": -2 if is_unreg else (
                100_000_000 + int(rec["ROWP"]) if is_prev else int(rec["ROW"])),
            "BORN_PREV_YEAR": int(is_prev),
            "NO_TRUE_PAIR": int(is_unreg),
        })

    out = OUTPUTS / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(synth_rows).write_csv(out / f"synthetic_SIM_{year}_{tier}.csv")
    pl.DataFrame(truth_rows).write_csv(out / f"truth_{year}_{tier}.csv")
    print(f"{year} {tier}: {len(synth_rows):,} synthetic deaths "
          f"({n_prev:,} born dec/{year-1}, {n_unreg:,} unregistered births, seed={seed})")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
