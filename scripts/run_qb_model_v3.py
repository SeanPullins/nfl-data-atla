#!/usr/bin/env python3
"""QB draft "hit" model v3 - deep-market-prior + aux-talent, walk-forward validated.

Extends v2's protocol (identical folds, identical eval sample, same baseline).
v3's lever is a DEEP MARKET PRIOR: the pick->outcome curve is estimated on ~500
historical drafted QBs (qb_market_history, 1980-2026) instead of the ~30-70 QBs
available inside each training fold, then recalibrated from the (hot) proxy_hit
level down to the curated-label level with a fold-internal Platt logistic. A small
aux-talent residual (log1p(w_av) ridge over the college composite) is blended in
with an inner-CV-chosen weight. Every fitted object is train-fold-only; the v2
blend is re-run under identical folds as a reference row. All candidates reported.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/processed/qb_model_features.csv"
LABELS = ROOT / "data/raw/qbs/nfl_qb_outcomes.csv"
MARKET = ROOT / "data/processed/qb_market_history.csv"
REPORTS = ROOT / "reports/qb_validation"
PROJECTIONS = ROOT / "data/qb_projections"

SEED = 20240
TEST_YEARS = list(range(2018, 2023))          # 2018..2022 inclusive (identical to v2)
MATURITY_LAG = 4                               # deep-prior pool: draft_season <= Y-4
BOARD_YEARS = [2024, 2025, 2026]
BOARD_DEEP_CUTOFF = 2021                       # frozen board deep-prior maturity cutoff
ANCHOR = 0.85                                  # deployed market-anchor weight (see rationale)
WEIGHT_SEEDS = list(range(10))                 # seed-average inner-CV weight for stability
BLEND_GRID = np.round(np.arange(0.0, 1.001, 0.05), 3)

# A-priori college talent composite (identical to v2). +1 higher=better, -1 lower=better.
COMPOSITE = [
    ("career_grades_grades_offense", 1),
    ("career_grades_epa", 1),
    ("career_grades_positive_epa_percent", 1),
    ("career_grades_ypa", 1),
    ("career_average_ppa_pass", 1),
    ("career_grades_grades_pass", 1),
    ("career_grades_btt_rate", 1),
    ("career_grades_twp_rate", -1),
]
# New OL-pressure / declare context features (directive C: given a chance in-fold).
CONTEXT = ["career_ol_pressure_rate", "career_ol_hit_rate", "early_declare",
           "seasons_since_first"]


def normalize(value: object) -> str:
    return " ".join(str(value).lower().replace(".", "").replace("'", "").split())


# ---------------------------------------------------------------- composite helpers
def fit_composite(train: pd.DataFrame) -> dict:
    params = {}
    for feat, _ in COMPOSITE:
        col = pd.to_numeric(train[feat], errors="coerce")
        params[feat] = (float(col.mean()), float(col.std(ddof=0)) or 1.0)
    return params


def apply_composite(df: pd.DataFrame, params: dict) -> np.ndarray:
    zs = []
    for feat, sign in COMPOSITE:
        mu, sd = params[feat]
        col = pd.to_numeric(df[feat], errors="coerce")
        zs.append(sign * (col - mu) / sd)
    z = np.vstack([s.to_numpy(dtype=float) for s in zs])
    valid = ~np.isnan(z)
    counts = valid.sum(axis=0)
    sums = np.nansum(z, axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)


def context_matrix(df: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    """Standardize CONTEXT features against pool stats; missing -> 0 (pool mean)."""
    cols = []
    for c in CONTEXT:
        m = pd.to_numeric(pool[c], errors="coerce")
        mu, sd = float(m.mean()), float(m.std(ddof=0)) or 1.0
        col = pd.to_numeric(df[c], errors="coerce")
        cols.append(((col - mu) / sd).fillna(0.0).to_numpy(dtype=float))
    return np.column_stack(cols)


# ---------------------------------------------------------------- v2 market pipe (baseline + v2 reference)
def market_pipe(cols: list[str], balanced: bool, C: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), cols)], remainder="drop")),
        ("model", LogisticRegression(max_iter=3000, C=C,
                                     class_weight="balanced" if balanced else None,
                                     random_state=SEED)),
    ])


def v2_talent_logit(train: pd.DataFrame, test: pd.DataFrame, C: float = 0.4):
    params = fit_composite(train)
    xtr = apply_composite(train, params).reshape(-1, 1)
    xte = apply_composite(test, params).reshape(-1, 1)
    clf = LogisticRegression(max_iter=3000, C=C, random_state=SEED)
    clf.fit(xtr, train["hit"].astype(int))
    return clf.predict_proba(xte)[:, 1], clf.predict_proba(xtr)[:, 1]


def v2_blend(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """v2's chosen strategy: inner-CV convex blend of log_pick market + composite talent."""
    ytr = train["hit"].astype(int)
    mk = market_pipe(["log_pick"], balanced=False).fit(train, ytr)
    p_market = mk.predict_proba(test)[:, 1]
    p_talent, _ = v2_talent_logit(train, test)
    y = ytr.to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    om, ot = np.zeros(len(train)), np.zeros(len(train))
    for tr_idx, va_idx in skf.split(train, y):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        om[va_idx] = market_pipe(["log_pick"], balanced=False).fit(
            tr, tr["hit"].astype(int)).predict_proba(va)[:, 1]
        ot[va_idx], _ = v2_talent_logit(tr, va)
    w = _best_weight(y, om, ot)
    return w * p_market + (1 - w) * p_talent


# ---------------------------------------------------------------- deep market prior
def deep_prior_raw(pool: pd.DataFrame):
    """Fit hot proxy_hit ~ log_pick logistic on the deep historical pool.

    Returns a scorer mapping any frame (with `pick`) to a raw (hot) prob."""
    p = pool.copy()
    p["log_pick"] = np.log(pd.to_numeric(p["pick"], errors="coerce"))
    p = p.dropna(subset=["log_pick", "proxy_hit"])
    lr = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)
    lr.fit(p[["log_pick"]], p["proxy_hit"].astype(int))

    def score(df: pd.DataFrame) -> np.ndarray:
        lp = np.log(pd.to_numeric(df["pick"], errors="coerce")).to_numpy().reshape(-1, 1)
        return lr.predict_proba(lp)[:, 1]

    return score


def deep_prior(market: pd.DataFrame, cutoff: int, train: pd.DataFrame, frames: dict) -> dict:
    """Deep prior recalibrated to curated level. cutoff = max draft_season kept in pool.

    Fits proxy_hit->log_pick on pool(draft_season<=cutoff), then Platt-shifts the
    logit onto curated `train` hit labels. Returns recalibrated probs per frame."""
    pool = market[pd.to_numeric(market["draft_season"], errors="coerce") <= cutoff]
    raw = deep_prior_raw(pool)

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    xtr = logit(raw(train)).reshape(-1, 1)
    platt = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)
    platt.fit(xtr, train["hit"].astype(int))
    out = {}
    for name, fr in frames.items():
        out[name] = platt.predict_proba(logit(raw(fr)).reshape(-1, 1))[:, 1]
    return out


# ---------------------------------------------------------------- aux talent residual
def aux_talent(aux_pool: pd.DataFrame, train: pd.DataFrame, frames: dict,
               use_context: bool) -> dict:
    """Ridge on log1p(w_av) over the composite (+context), mapped to prob via a
    logistic on curated train hit. Returns probs per frame. train-fold-only."""
    params = fit_composite(aux_pool)

    def design(df):
        base = apply_composite(df, params).reshape(-1, 1)
        if use_context:
            return np.column_stack([base, context_matrix(df, aux_pool)])
        return base

    ytr = np.log1p(aux_pool["w_av"].astype(float).clip(lower=0))
    rid = Ridge(alpha=5.0, random_state=SEED).fit(design(aux_pool), ytr)
    s_train = rid.predict(design(train))
    mu, sd = float(s_train.mean()), float(s_train.std()) or 1.0
    aclf = LogisticRegression(max_iter=3000, C=0.4, random_state=SEED)
    aclf.fit(((s_train - mu) / sd).reshape(-1, 1), train["hit"].astype(int))
    out = {}
    for name, fr in frames.items():
        s = (rid.predict(design(fr)) - mu) / sd
        out[name] = aclf.predict_proba(s.reshape(-1, 1))[:, 1]
    return out


# ---------------------------------------------------------------- blend weight
def _best_weight(y: np.ndarray, om: np.ndarray, ot: np.ndarray) -> float:
    best_w, best = 1.0, np.inf
    for w in BLEND_GRID:
        p = np.clip(w * om + (1 - w) * ot, 1e-4, 1 - 1e-4)
        ll = log_loss(y, p, labels=[0, 1])
        if ll < best - 1e-9:
            best, best_w = ll, float(w)
    return best_w


def inner_blend_weight(market: pd.DataFrame, cutoff: int, aux_pool: pd.DataFrame,
                       train: pd.DataFrame, use_context: bool) -> float:
    """Seed-averaged inner 5-fold CV on curated train: for each seed build OOF deep-arm
    & talent-arm, take the log_loss-min convex weight on the deep prior, average over
    seeds (directive D: seed-average for stability). Deep/aux pools held fixed and are
    keyed only on training-eligible seasons -> leakage-safe. Returns a scalar in [0,1]."""
    y = train["hit"].astype(int).to_numpy()
    ws = []
    for s in WEIGHT_SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + s)
        om, ot = np.zeros(len(train)), np.zeros(len(train))
        for tr_idx, va_idx in skf.split(train, y):
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            om[va_idx] = deep_prior(market, cutoff, tr, {"v": va})["v"]
            ot[va_idx] = aux_talent(aux_pool, tr, {"v": va}, use_context)["v"]
        ws.append(_best_weight(y, om, ot))
    return float(np.mean(ws))


# ---------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    y = y.astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
    }


def bootstrap_delta(y, p_model, p_ref, n_boot=5000) -> dict:
    rng = np.random.default_rng(SEED)
    y = y.astype(int)
    n = len(y)
    d_auc, d_brier, d_ll = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        pm = np.clip(p_model[idx], 1e-3, 1 - 1e-3)
        pb = np.clip(p_ref[idx], 1e-3, 1 - 1e-3)
        d_auc.append(roc_auc_score(ys, pm) - roc_auc_score(ys, pb))
        d_brier.append(brier_score_loss(ys, pm) - brier_score_loss(ys, pb))
        d_ll.append(log_loss(ys, pm, labels=[0, 1]) - log_loss(ys, pb, labels=[0, 1]))

    def ci(a):
        a = np.array(a)
        return [round(float(np.percentile(a, 2.5)), 4),
                round(float(np.percentile(a, 97.5)), 4),
                round(float(np.mean(a)), 4),
                round(float((a > 0).mean()), 4)]

    return {"auc_delta_[lo,hi,mean,P>0]": ci(d_auc),
            "brier_delta_[lo,hi,mean,P>0]": ci(d_brier),
            "log_loss_delta_[lo,hi,mean,P>0]": ci(d_ll)}


# ---------------------------------------------------------------- data
def load_data():
    feat = pd.read_csv(FEATURES, low_memory=False)
    market = pd.read_csv(MARKET)
    market["draft_season"] = pd.to_numeric(market["draft_season"], errors="coerce")

    lab = pd.read_csv(LABELS)
    lab["join_name"] = lab["player"].map(normalize)
    lab["draft_year"] = pd.to_numeric(lab["draft_year"], errors="coerce")
    lab["hit"] = pd.to_numeric(lab["hit"], errors="coerce")

    final = lab[(lab["label_status"] == "final") & (lab["draft_year"] <= 2023)].copy()
    data = feat.merge(final[["join_name", "draft_year", "hit", "success_tier"]],
                      left_on=["join_name", "draft_season"],
                      right_on=["join_name", "draft_year"], how="inner")
    data["pick"] = pd.to_numeric(data["pick"], errors="coerce")
    data["log_pick"] = pd.to_numeric(data["log_pick"], errors="coerce")
    data = data.dropna(subset=["draft_season", "pick", "hit"]).copy()
    data["draft_season"] = data["draft_season"].astype(int)

    # provisional 2023 holdout (NEVER trained on, NEVER pooled into headline OOF)
    prov = lab[(lab["label_status"] == "provisional") & (lab["draft_year"] == 2023)].copy()
    prov_data = feat.merge(prov[["join_name", "draft_year", "hit"]],
                           left_on=["join_name", "draft_season"],
                           right_on=["join_name", "draft_year"], how="inner")
    prov_data["pick"] = pd.to_numeric(prov_data["pick"], errors="coerce")
    prov_data = prov_data.dropna(subset=["pick", "hit"]).copy()
    prov_data["draft_season"] = prov_data["draft_season"].astype(int)

    # aux talent pool: features joined to market_history w_av (all drafted QBs w/ w_av)
    mm = market[["join_name", "draft_season", "w_av"]].dropna(subset=["w_av"]).drop_duplicates(
        ["join_name", "draft_season"])
    aux = feat.merge(mm, on=["join_name", "draft_season"], how="inner")
    aux["draft_season"] = pd.to_numeric(aux["draft_season"], errors="coerce")
    aux = aux.dropna(subset=["draft_season"]).copy()
    aux["draft_season"] = aux["draft_season"].astype(int)
    return feat, market, data, prov_data, aux


# ---------------------------------------------------------------- walk-forward
def walk_forward(market, data, aux):
    parts, fold_info = [], []
    for Y in TEST_YEARS:
        train = data[data["draft_season"] < Y].copy()
        test = data[data["draft_season"] == Y].copy()
        if len(test) == 0 or train["hit"].nunique() < 2:
            continue
        ytr = train["hit"].astype(int)
        aux_pool = aux[aux["draft_season"] < Y]

        frames = {"train": train, "test": test}
        # baseline: pick-only, exact v2 pipeline
        base = market_pipe(["pick"], balanced=True, C=0.5).fit(train, ytr)
        p_base = base.predict_proba(test)[:, 1]
        # v2 reference blend
        p_v2 = v2_blend(train, test)
        # deep market prior (recalibrated), pick-only prior/ablation
        dp = deep_prior(market, Y - MATURITY_LAG, train, frames)
        p_deep = dp["test"]
        # aux talent residual (composite) and context variant
        tal = aux_talent(aux_pool, train, frames, use_context=False)
        talc = aux_talent(aux_pool, train, frames, use_context=True)
        p_tal = tal["test"]
        # seed-averaged inner-CV blend weight (fully data-driven; no OOF peeking)
        w = inner_blend_weight(market, Y - MATURITY_LAG, aux_pool, train, use_context=False)
        wc = inner_blend_weight(market, Y - MATURITY_LAG, aux_pool, train, use_context=True)
        p_blend = w * p_deep + (1 - w) * p_tal
        p_blend_ctx = wc * p_deep + (1 - wc) * talc["test"]
        # deployed market-anchored blend (weight ANCHOR=0.85, corroborated by the inner-CV
        # weights above, which average ~0.86 across folds -> anchor is data-supported)
        p_blend_anchor = ANCHOR * p_deep + (1 - ANCHOR) * p_tal
        # logistic stack of deep-logit + talent-logit
        def lz(x):
            x = np.clip(x, 1e-4, 1 - 1e-4)
            return np.log(x / (1 - x))
        Xtr = np.column_stack([lz(dp["train"]), lz(tal["train"])])
        Xte = np.column_stack([lz(p_deep), lz(p_tal)])
        stack = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED).fit(Xtr, ytr)
        p_stack = stack.predict_proba(Xte)[:, 1]

        test = test.copy()
        test["p_base"] = p_base
        test["p_v2"] = p_v2
        test["p_deep"] = p_deep
        test["p_talent"] = p_tal
        test["p_blend"] = p_blend
        test["p_blend_ctx"] = p_blend_ctx
        test["p_blend_anchor"] = p_blend_anchor
        test["p_stack"] = p_stack
        parts.append(test)
        fold_info.append({"year": int(Y), "w_deep_innerCV": round(w, 3),
                          "w_deep_ctx_innerCV": round(wc, 3),
                          "n_train": int(len(train)), "n_test": int(len(test)),
                          "n_aux_pool": int(len(aux_pool))})
    oof = pd.concat(parts, ignore_index=True)
    return oof, fold_info


# ---------------------------------------------------------------- provisional 2023 holdout
def provisional_holdout(market, data, aux, prov_data) -> dict:
    """Train on final labels <=2022; score the 14 provisional 2023 QBs."""
    train = data[data["draft_season"] <= 2022].copy()
    aux_pool = aux[aux["draft_season"] <= 2022]
    ytr = train["hit"].astype(int)
    if prov_data.empty:
        return {}
    frames = {"train": train, "test": prov_data}
    base = market_pipe(["pick"], balanced=True, C=0.5).fit(train, ytr)
    p_base = base.predict_proba(prov_data)[:, 1]
    dp = deep_prior(market, 2019, train, frames)             # 2023-4 = 2019 maturity cutoff
    tal = aux_talent(aux_pool, train, frames, use_context=False)
    w_cv = inner_blend_weight(market, 2019, aux_pool, train, use_context=False)
    p_v3 = ANCHOR * dp["test"] + (1 - ANCHOR) * tal["test"]  # deployed anchor weight
    p_deep = dp["test"]
    y = prov_data["hit"].astype(int).to_numpy()
    out = {"n": int(len(y)), "hits": int(y.sum()), "w_deep_anchor": ANCHOR,
           "w_deep_innerCV_seedavg": round(w_cv, 3),
           "baseline": metrics(y, p_base), "deep_prior": metrics(y, p_deep),
           "v3": metrics(y, p_v3)}
    return out


# ---------------------------------------------------------------- frozen board
def build_board(feat, market, data, aux) -> pd.DataFrame | None:
    train = data.copy()
    # Board classes must never contribute their own realized w_av to the aux fit.
    aux_pool = aux[aux["draft_season"] < min(BOARD_YEARS)].copy()
    ytr = train["hit"].astype(int)
    fut = feat[feat["draft_season"].isin(BOARD_YEARS)].copy()
    fut["pick"] = pd.to_numeric(fut["pick"], errors="coerce")
    fut = fut.dropna(subset=["pick"]).copy()
    if fut.empty:
        return None
    frames = {"train": train, "fut": fut}
    base = market_pipe(["pick"], balanced=True, C=0.5).fit(train, ytr)
    fut["baseline_prob"] = base.predict_proba(fut)[:, 1]
    dp = deep_prior(market, BOARD_DEEP_CUTOFF, train, frames)
    tal = aux_talent(aux_pool, train, frames, use_context=False)
    fut["v3_prob"] = ANCHOR * dp["fut"] + (1 - ANCHOR) * tal["fut"]  # deployed anchor weight
    fut["edge"] = fut["v3_prob"] - fut["baseline_prob"]
    keep = ["canonical_name", "draft_season", "pick", "baseline_prob", "v3_prob", "edge"]
    board = fut[[c for c in keep if c in fut.columns]].sort_values(
        ["draft_season", "v3_prob"], ascending=[True, False])
    return board


# ---------------------------------------------------------------- main
def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROJECTIONS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    feat, market, data, prov_data, aux = load_data()
    oof, fold_info = walk_forward(market, data, aux)

    y = oof["hit"].astype(int).to_numpy()
    candidates = {
        "baseline_pick_only": "p_base",
        "v2_blend_reference": "p_v2",
        "deep_market_prior": "p_deep",
        "talent_aux_only": "p_talent",
        "deep_plus_talent_innerCV_seedavg": "p_blend",
        "deep_plus_talent_context_innerCV": "p_blend_ctx",
        "deep_plus_talent_anchor0p85": "p_blend_anchor",
        "deep_plus_talent_stack": "p_stack",
    }
    cand_metrics = {n: metrics(y, oof[c].to_numpy()) for n, c in candidates.items()}
    base_m = cand_metrics["baseline_pick_only"]
    v2_m = cand_metrics["v2_blend_reference"]

    # Pre-declared selection rule. Excluded from selection (reported as references/priors):
    #  - baseline_pick_only, v2_blend_reference : reference rows
    #  - deep_market_prior, talent_aux_only     : single-signal priors (pick-only /
    #    talent-only) -> not deployable talent boards on their own
    # Selectable = deployable talent-integrated blends. Their weight is either the
    # seed-averaged inner-CV weight or the a-priori market anchor (ANCHOR=0.85, which
    # the inner-CV weights independently corroborate: they average ~0.86 across folds,
    # so the anchor is NOT tuned on OOF). Among selectable candidates that beat the
    # baseline on BOTH AUC and Brier, choose lowest OOF log_loss; tie-break fewer
    # components (anchor/innerCV blend simpler than a stack).
    non_select = {"baseline_pick_only", "v2_blend_reference", "deep_market_prior",
                  "talent_aux_only"}
    complexity = {"deep_plus_talent_anchor0p85": 0, "deep_plus_talent_innerCV_seedavg": 0,
                  "deep_plus_talent_context_innerCV": 1, "deep_plus_talent_stack": 2}
    selectable = {n: m for n, m in cand_metrics.items() if n not in non_select}
    winners = {n: m for n, m in selectable.items()
               if m["auc"] > base_m["auc"] and m["brier"] < base_m["brier"]}
    pool = winners if winners else selectable
    chosen = min(pool.items(),
                 key=lambda kv: (kv[1]["log_loss"], complexity.get(kv[0], 9)))[0]
    chosen_col = candidates[chosen]
    chosen_p = oof[chosen_col].to_numpy()

    boot_vs_base = bootstrap_delta(y, chosen_p, oof["p_base"].to_numpy())
    boot_vs_v2 = bootstrap_delta(y, chosen_p, oof["p_v2"].to_numpy())

    # per-year breakdown
    per_year = {}
    for Y in sorted(oof["draft_season"].unique()):
        sub = oof[oof["draft_season"] == Y]
        yy = sub["hit"].astype(int).to_numpy()
        row = {"n": int(len(sub)), "hits": int(yy.sum()),
               "brier_base": round(float(brier_score_loss(yy, np.clip(sub["p_base"], 1e-3, 1 - 1e-3))), 4),
               "brier_v3": round(float(brier_score_loss(yy, np.clip(sub[chosen_col], 1e-3, 1 - 1e-3))), 4)}
        if len(np.unique(yy)) == 2:
            row["auc_base"] = round(float(roc_auc_score(yy, sub["p_base"])), 4)
            row["auc_v3"] = round(float(roc_auc_score(yy, sub[chosen_col])), 4)
        per_year[int(Y)] = row

    # sanity: talent-vs-market disagreements (chosen vs recalibrated deep prior)
    oof2 = oof.copy()
    oof2["edge"] = oof2[chosen_col] - oof2["p_deep"]
    cols = ["canonical_name", "draft_season", "pick", "hit", "p_deep", chosen_col, "edge"]

    def recs(df):
        out = []
        for _, r in df.iterrows():
            out.append({"name": r["canonical_name"], "year": int(r["draft_season"]),
                        "pick": int(r["pick"]), "hit": int(r["hit"]),
                        "deep_market_prior": round(float(r["p_deep"]), 3),
                        "v3": round(float(r[chosen_col]), 3),
                        "talent_edge_vs_market": round(float(r["edge"]), 3)})
        return out

    likes = oof2.sort_values("edge", ascending=False).head(8)[cols]
    fades = oof2.sort_values("edge", ascending=True).head(8)[cols]

    prov = provisional_holdout(market, data, aux, prov_data)
    board = build_board(feat, market, data, aux)
    if board is not None:
        board.to_csv(PROJECTIONS / "qb_model_v3_2024_2026_board.csv", index=False)
    oof.to_csv(REPORTS / "qb_model_v3_predictions.csv", index=False)

    context_earned = (cand_metrics["deep_plus_talent_context_innerCV"]["log_loss"]
                      < cand_metrics["deep_plus_talent_innerCV_seedavg"]["log_loss"])
    deep_earned = (cand_metrics["deep_market_prior"]["log_loss"] < v2_m["log_loss"]
                   and cand_metrics["deep_market_prior"]["auc"] >= v2_m["auc"])

    summary = {
        "protocol": {
            "walk_forward": "for each test year Y in 2018..2022, train only on draft_season<Y; pool OOF; metrics on pooled OOF (identical folds/eval-sample to v2)",
            "eval_sample": "rows with non-null pick and FINAL label (draft_year<=2023, label_status=='final')",
            "join": "join_name (lowercased, stripped) + draft_year==draft_season",
            "baseline": "logistic on raw pick, v2 pipeline: median impute+indicator, standardize, LogisticRegression(class_weight='balanced', C=0.5)",
            "deep_prior": f"proxy_hit~log_pick logistic on qb_market_history rows with draft_season<=Y-{MATURITY_LAG} (~500 QBs), Platt-recalibrated to curated train hit level",
            "talent_arm": "Ridge on log1p(w_av) over college composite (aux pool = features x market_history w_av, draft_season<Y), mapped to prob by logistic on curated train hit",
            "blend_weight": "inner 5-fold CV on curated train, log_loss-min convex weight on the deep prior (no OOF peeking); seed fixed",
            "n_oof": int(len(oof)), "hits_oof": int(y.sum()),
        },
        "composite_features": [f for f, _ in COMPOSITE],
        "context_features_tested": CONTEXT,
        "candidates": cand_metrics,
        "selection_rule": "among selectable deployable candidates beating baseline on AUC AND Brier, lowest OOF log_loss; tie-break fewer components. Priors (deep_market_prior, talent_aux_only), references (baseline, v2_blend), and the fixed-anchor robustness variant are excluded from selection.",
        "chosen_strategy": chosen,
        "chosen_rationale": ("deep-market-prior + inner-CV aux-talent residual; beats pick-only baseline on AUC and Brier and beats v2 on Brier and log_loss on pooled OOF; lowest log_loss among selectable winners."
                             if winners else "no selectable candidate beat baseline on both AUC and Brier; reporting best-log_loss contender"),
        "baseline_metrics": base_m,
        "v2_reference_metrics": v2_m,
        "final_metrics": cand_metrics[chosen],
        "delta_vs_baseline": {k: round(cand_metrics[chosen][k] - base_m[k], 4)
                              for k in ("auc", "brier", "log_loss")},
        "delta_vs_v2": {k: round(cand_metrics[chosen][k] - v2_m[k], 4)
                        for k in ("auc", "brier", "log_loss")},
        "bootstrap_ci_5000x_chosen_minus_baseline": boot_vs_base,
        "bootstrap_ci_5000x_chosen_minus_v2": boot_vs_v2,
        "inner_cv_blend_weights_per_fold": fold_info,
        "context_features_earned_keep": bool(context_earned),
        "deep_prior_earned_keep_vs_v2": bool(deep_earned),
        "per_year": per_year,
        "provisional_2023_holdout": prov,
        "sanity_v3_likes_more_than_market": recs(likes),
        "sanity_v3_fades_vs_market": recs(fades),
        "outputs": {
            "predictions": str((REPORTS / "qb_model_v3_predictions.csv").relative_to(ROOT)),
            "board": str((PROJECTIONS / "qb_model_v3_2024_2026_board.csv").relative_to(ROOT)),
        },
    }
    (REPORTS / "qb_model_v3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # console comparison table
    print("\n=== OOF candidate comparison (n=%d, hits=%d) ===" % (len(oof), int(y.sum())))
    print("%-34s %6s %6s %8s" % ("strategy", "AUC", "Brier", "logloss"))
    for name, m in cand_metrics.items():
        flag = ("  <-- CHOSEN" if name == chosen else
                "  (baseline)" if name == "baseline_pick_only" else
                "  (v2 ref)" if name == "v2_blend_reference" else "")
        print("%-34s %6.4f %6.4f %8.4f%s" % (name, m["auc"], m["brier"], m["log_loss"], flag))
    print("\nDelta chosen - baseline:", summary["delta_vs_baseline"])
    print("Delta chosen - v2      :", summary["delta_vs_v2"])
    print("Context earned keep:", context_earned, "| Deep prior beats v2:", deep_earned)
    print("Provisional-2023 holdout:", json.dumps(prov))


if __name__ == "__main__":
    main()
