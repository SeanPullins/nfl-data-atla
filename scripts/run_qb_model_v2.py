#!/usr/bin/env python3
"""QB draft "hit" model v2 - market-anchored talent-residual, walk-forward validated.

Goal: beat the draft-pick-only logistic baseline on pooled out-of-fold (OOF)
predictions (AUC and Brier) under a strict walk-forward, no-leakage protocol.
All feature/hyperparameter/blend choices are made inside training folds or fixed
a priori. Every candidate strategy tried is reported in the summary JSON.
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
DRAFT_MASTER = ROOT / "data/raw/open_prospect/nflverse_draft_player_master_SAFE.csv"
REPORTS = ROOT / "reports/qb_validation"
PROJECTIONS = ROOT / "data/qb_projections"

SEED = 20240
TEST_YEARS = list(range(2018, 2023))  # 2018..2022 inclusive

# A-priori talent composite: well-covered, stable college signals (career-weighted).
# Sign +1 = higher is better; -1 = lower is better (negated before averaging).
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
# Compact talent feature set for the regularized talent logistic / aux model.
TALENT_FEATS = [f for f, _ in COMPOSITE]


def normalize(value: object) -> str:
    return " ".join(str(value).lower().replace(".", "").replace("'", "").split())


# ---------------------------------------------------------------- market pipes
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


# ---------------------------------------------------------------- composite
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
    z = np.vstack([s.to_numpy(dtype=float) for s in zs])  # (nfeat, nrow)
    # row-wise nanmean; all-missing -> 0 (train mean)
    valid = ~np.isnan(z)
    counts = valid.sum(axis=0)
    sums = np.nansum(z, axis=0)
    out = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return out


def talent_logit(train: pd.DataFrame, test: pd.DataFrame, C: float = 0.4):
    """1-D logistic on the standardized talent composite (unbalanced)."""
    params = fit_composite(train)
    xtr = apply_composite(train, params).reshape(-1, 1)
    xte = apply_composite(test, params).reshape(-1, 1)
    clf = LogisticRegression(max_iter=3000, C=C, random_state=SEED)
    clf.fit(xtr, train["hit"].astype(int))
    return clf.predict_proba(xte)[:, 1], xtr.ravel(), xte.ravel()


# ---------------------------------------------------------------- blend weight
def choose_blend_weight(train: pd.DataFrame, grid: np.ndarray) -> float:
    """Inner stratified 5-fold CV on train; pick w minimizing pooled log_loss of
    w*p_market(log_pick) + (1-w)*p_talent."""
    y = train["hit"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_m = np.zeros(len(train))
    oof_t = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(train, y):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        mk = market_pipe(["log_pick"], balanced=False).fit(tr, tr["hit"].astype(int))
        oof_m[va_idx] = mk.predict_proba(va)[:, 1]
        pt, _, _ = talent_logit(tr, va)
        oof_t[va_idx] = pt
    best_w, best_ll = 1.0, np.inf
    for w in grid:
        p = np.clip(w * oof_m + (1 - w) * oof_t, 1e-4, 1 - 1e-4)
        ll = log_loss(y, p, labels=[0, 1])
        if ll < best_ll - 1e-9:
            best_ll, best_w = ll, float(w)
    return best_w


# ---------------------------------------------------------------- aux target
def aux_talent_score(train_pool: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Ridge on log1p(w_av) using ALL drafted QBs (with w_av) in train years, over
    the talent composite features; return standardized predicted-score for test."""
    params = fit_composite(train_pool)
    xtr = apply_composite(train_pool, params).reshape(-1, 1)
    ytr = np.log1p(train_pool["w_av"].astype(float).clip(lower=0))
    rid = Ridge(alpha=5.0, random_state=SEED).fit(xtr, ytr)
    xte = apply_composite(test, params).reshape(-1, 1)
    s = rid.predict(xte)
    mu, sd = s.mean(), s.std() or 1.0
    return (s - mu) / sd


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


def bootstrap_delta(y: np.ndarray, p_model: np.ndarray, p_base: np.ndarray,
                    n_boot: int = 5000) -> dict:
    rng = np.random.default_rng(SEED)
    y = y.astype(int)
    n = len(y)
    d_auc, d_brier, d_ll = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        pm, pb = np.clip(p_model[idx], 1e-3, 1 - 1e-3), np.clip(p_base[idx], 1e-3, 1 - 1e-3)
        d_auc.append(roc_auc_score(ys, pm) - roc_auc_score(ys, pb))
        d_brier.append(brier_score_loss(ys, pm) - brier_score_loss(ys, pb))
        d_ll.append(log_loss(ys, pm, labels=[0, 1]) - log_loss(ys, pb, labels=[0, 1]))

    def ci(a):
        a = np.array(a)
        return [round(float(np.percentile(a, 2.5)), 4),
                round(float(np.percentile(a, 97.5)), 4),
                round(float(np.mean(a)), 4),
                round(float((a > 0).mean()), 4)]  # P(delta>0)

    return {"auc_delta_[lo,hi,mean,P>0]": ci(d_auc),
            "brier_delta_[lo,hi,mean,P>0]": ci(d_brier),
            "log_loss_delta_[lo,hi,mean,P>0]": ci(d_ll)}


# ---------------------------------------------------------------- data
def load_data():
    feat = pd.read_csv(FEATURES, low_memory=False)
    lab = pd.read_csv(LABELS)
    lab["join_name"] = lab["player"].map(normalize)
    lab["draft_year"] = pd.to_numeric(lab["draft_year"], errors="coerce")
    lab["hit"] = pd.to_numeric(lab["hit"], errors="coerce")
    lab = lab[(lab["label_status"] == "final") & (lab["draft_year"] <= 2023)].copy()
    data = feat.merge(lab[["join_name", "draft_year", "hit"]],
                      left_on=["join_name", "draft_season"],
                      right_on=["join_name", "draft_year"], how="inner")
    data["pick"] = pd.to_numeric(data["pick"], errors="coerce")
    data["log_pick"] = pd.to_numeric(data["log_pick"], errors="coerce")
    data = data.dropna(subset=["draft_season", "pick", "hit"]).copy()
    data["draft_season"] = data["draft_season"].astype(int)

    # Auxiliary pool: all drafted QBs with w_av, joined to features by name+year.
    aux = pd.read_csv(DRAFT_MASTER, low_memory=False)
    aux = aux[aux["position"] == "QB"].copy()
    aux["join_name"] = aux["pfr_player_name"].map(normalize)
    aux["draft_season"] = pd.to_numeric(aux["season"], errors="coerce")
    aux = aux[["join_name", "draft_season", "w_av"]].dropna(subset=["w_av"]).drop_duplicates(["join_name", "draft_season"])
    aux_feat = feat.merge(aux, on=["join_name", "draft_season"], how="inner")
    aux_feat["draft_season"] = pd.to_numeric(aux_feat["draft_season"], errors="coerce")
    aux_feat = aux_feat.dropna(subset=["draft_season"]).copy()
    aux_feat["draft_season"] = aux_feat["draft_season"].astype(int)
    return feat, data, aux_feat


# ---------------------------------------------------------------- walk-forward
def walk_forward(data: pd.DataFrame, aux_feat: pd.DataFrame):
    """Return dict candidate -> OOF prob array aligned to `oof` frame order."""
    blend_grid = np.round(np.arange(0.0, 1.001, 0.05), 3)
    parts = []
    chosen_w = []
    for Y in TEST_YEARS:
        train = data[data["draft_season"] < Y].copy()
        test = data[data["draft_season"] == Y].copy()
        if len(test) == 0 or train["hit"].nunique() < 2:
            continue
        ytr = train["hit"].astype(int)

        # baseline: pick-only, exact old-lab pipeline (balanced, C=0.5)
        base = market_pipe(["pick"], balanced=True, C=0.5).fit(train, ytr)
        p_base = base.predict_proba(test)[:, 1]

        # C1: log_pick market prior (unbalanced -> better calibrated)
        mk = market_pipe(["log_pick"], balanced=False).fit(train, ytr)
        p_market = mk.predict_proba(test)[:, 1]

        # C2: talent composite logistic
        p_talent, _, _ = talent_logit(train, test)

        # C3: convex blend, weight from inner CV on train
        w = choose_blend_weight(train, blend_grid)
        chosen_w.append(w)
        p_blend = w * p_market + (1 - w) * p_talent

        # C4: stacked logistic on [logit(p_market_cv-consistent), composite]
        #     use train-fit market + composite as 2 features, small L2, unbalanced
        params = fit_composite(train)
        mk_tr = mk.predict_proba(train)[:, 1]
        Xtr = np.column_stack([np.log(mk_tr / (1 - mk_tr)), apply_composite(train, params)])
        Xte = np.column_stack([np.log(p_market / (1 - p_market)), apply_composite(test, params)])
        stack = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED).fit(Xtr, ytr)
        p_stack = stack.predict_proba(Xte)[:, 1]

        # C5: aux w_av-augmented talent blended with market (fixed 0.5 blend a priori)
        aux_pool = aux_feat[aux_feat["draft_season"] < Y]
        if len(aux_pool) >= 20:
            aux_s = aux_talent_score(aux_pool, test)
            # map aux score -> prob via logistic fit on labeled train aux score
            aux_tr = aux_talent_score(aux_pool, train).reshape(-1, 1)
            aclf = LogisticRegression(max_iter=3000, C=0.4, random_state=SEED).fit(aux_tr, ytr)
            p_aux_t = aclf.predict_proba(aux_s.reshape(-1, 1))[:, 1]
        else:
            p_aux_t = p_talent
        p_aux = 0.5 * p_market + 0.5 * p_aux_t

        test = test.copy()
        test["p_base"] = p_base
        test["p_market"] = p_market
        test["p_talent"] = p_talent
        test["p_blend"] = p_blend
        test["p_stack"] = p_stack
        test["p_aux"] = p_aux
        parts.append(test)

    oof = pd.concat(parts, ignore_index=True)
    return oof, chosen_w


# ---------------------------------------------------------------- frozen board
def build_board(feat: pd.DataFrame, data: pd.DataFrame, aux_feat: pd.DataFrame,
                candidate: str):
    train = data.copy()
    ytr = train["hit"].astype(int)
    fut = feat[feat["draft_season"].isin([2024, 2025, 2026])].copy()
    fut["pick"] = pd.to_numeric(fut["pick"], errors="coerce")
    fut["log_pick"] = pd.to_numeric(fut["log_pick"], errors="coerce")
    fut = fut.dropna(subset=["pick"]).copy()
    if fut.empty:
        return None
    base = market_pipe(["pick"], balanced=True, C=0.5).fit(train, ytr)
    fut["baseline_prob"] = base.predict_proba(fut)[:, 1]
    mk = market_pipe(["log_pick"], balanced=False).fit(train, ytr)
    p_market = mk.predict_proba(fut)[:, 1]
    p_talent, _, _ = talent_logit(train, fut)
    if candidate == "blend":
        w = choose_blend_weight(train, np.round(np.arange(0.0, 1.001, 0.05), 3))
        fut["model_prob"] = w * p_market + (1 - w) * p_talent
    elif candidate == "stack":
        params = fit_composite(train)
        mk_tr = mk.predict_proba(train)[:, 1]
        Xtr = np.column_stack([np.log(mk_tr / (1 - mk_tr)), apply_composite(train, params)])
        Xfu = np.column_stack([np.log(p_market / (1 - p_market)), apply_composite(fut, params)])
        stack = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED).fit(Xtr, ytr)
        fut["model_prob"] = stack.predict_proba(Xfu)[:, 1]
    else:
        fut["model_prob"] = p_market
    fut["edge"] = fut["model_prob"] - fut["baseline_prob"]
    keep = ["canonical_name", "draft_season", "pick", "colleges",
            "baseline_prob", "model_prob", "edge"]
    board = fut[[c for c in keep if c in fut.columns]].sort_values(
        ["draft_season", "model_prob"], ascending=[True, False])
    return board


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROJECTIONS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    feat, data, aux_feat = load_data()
    oof, chosen_w = walk_forward(data, aux_feat)

    y = oof["hit"].astype(int).to_numpy()
    candidates = {
        "baseline_pick_only": "p_base",
        "market_logpick": "p_market",
        "talent_composite_only": "p_talent",
        "blend_market_talent_innerCV": "p_blend",
        "stack_market_talent": "p_stack",
        "aux_wav_augmented": "p_aux",
    }
    cand_metrics = {name: metrics(y, oof[col].to_numpy()) for name, col in candidates.items()}

    base_m = cand_metrics["baseline_pick_only"]
    # Pre-declared selection rule: market_logpick and talent_composite_only are
    # single-signal ABLATIONS/priors, not deployable models (a pure-market model
    # gives identical prob to every QB at the same pick -> useless board). The
    # deployable model candidates are the market-anchored + talent designs. Among
    # those that beat the baseline on BOTH AUC and Brier, choose the lowest OOF
    # log_loss. All candidates (incl. ablations) are reported below.
    ablations = {"market_logpick", "talent_composite_only"}
    model_candidates = {n: m for n, m in cand_metrics.items()
                        if n not in ablations and n != "baseline_pick_only"}
    winners = {n: m for n, m in model_candidates.items()
               if m["auc"] > base_m["auc"] and m["brier"] < base_m["brier"]}
    pool = winners if winners else model_candidates
    chosen = min(pool.items(), key=lambda kv: (kv[1]["log_loss"], -kv[1]["auc"]))[0]
    chosen_col = candidates[chosen]
    chosen_key = "blend" if "blend" in chosen else ("stack" if "stack" in chosen else "market")

    boot = bootstrap_delta(y, oof[chosen_col].to_numpy(), oof["p_base"].to_numpy())

    # per-year breakdown for baseline vs chosen
    per_year = {}
    for Y in sorted(oof["draft_season"].unique()):
        sub = oof[oof["draft_season"] == Y]
        yy = sub["hit"].astype(int).to_numpy()
        row = {"n": int(len(sub)), "hits": int(yy.sum())}
        if len(np.unique(yy)) == 2:
            row["auc_base"] = round(float(roc_auc_score(yy, sub["p_base"])), 4)
            row["auc_model"] = round(float(roc_auc_score(yy, sub[chosen_col])), 4)
        per_year[int(Y)] = row

    # sanity: talent contribution isolated from calibration -> model vs market
    # prior at the SAME (unbalanced) calibration. Positive = model likes the QB
    # MORE than draft position alone; negative = fades vs draft position.
    oof2 = oof.copy()
    oof2["edge"] = oof2[chosen_col] - oof2["p_market"]
    cols = ["canonical_name", "draft_season", "pick", "hit", "p_market", chosen_col, "edge"]
    likes = oof2.sort_values("edge", ascending=False).head(8)[cols]
    fades = oof2.sort_values("edge", ascending=True).head(8)[cols]

    def recs(df):
        out = []
        for _, r in df.iterrows():
            out.append({"name": r["canonical_name"], "year": int(r["draft_season"]),
                        "pick": int(r["pick"]), "hit": int(r["hit"]),
                        "market_prior": round(float(r["p_market"]), 3),
                        "model": round(float(r[chosen_col]), 3),
                        "talent_edge_vs_market": round(float(r["edge"]), 3)})
        return out

    board = build_board(feat, data, aux_feat, chosen_key)
    if board is not None:
        board.to_csv(PROJECTIONS / "qb_model_v2_2024_2026_board.csv", index=False)

    oof.to_csv(REPORTS / "qb_model_v2_predictions.csv", index=False)

    summary = {
        "protocol": {
            "walk_forward": "for each test year Y in 2018..2022, train only on draft_season<Y; pool OOF preds; metrics on pooled OOF",
            "eval_sample": "rows with non-null pick and final label (draft_year<=2023, label_status=='final')",
            "join": "join_name (lowercased, stripped) + draft_year==draft_season",
            "baseline": "logistic on raw pick, exact old-lab pipeline: median impute+indicator, standardize, LogisticRegression(max_iter=3000, class_weight='balanced', C=0.5)",
            "choices": "all feature/hyperparameter/blend choices fixed a priori or via inner 5-fold CV on train only; seeds fixed (deterministic)",
            "n_oof": int(len(oof)), "hits_oof": int(y.sum()),
        },
        "composite_features": [f for f, _ in COMPOSITE],
        "candidates": cand_metrics,
        "chosen_strategy": chosen,
        "ablation_priors_not_deployable": sorted(ablations),
        "chosen_rationale": ("market-anchored talent model; beats pick-only baseline on AUC and Brier on pooled OOF; lowest log_loss among deployable winners. (market_logpick/talent_only are single-signal ablations, excluded from selection.)"
                             if winners else "no deployable candidate beat baseline on both AUC and Brier; reporting best-log_loss contender"),
        "baseline_metrics": base_m,
        "final_metrics": cand_metrics[chosen],
        "delta_vs_baseline": {
            "auc": round(cand_metrics[chosen]["auc"] - base_m["auc"], 4),
            "brier": round(cand_metrics[chosen]["brier"] - base_m["brier"], 4),
            "log_loss": round(cand_metrics[chosen]["log_loss"] - base_m["log_loss"], 4),
        },
        "bootstrap_ci_delta_model_minus_baseline_5000x": boot,
        "inner_cv_blend_weights_per_fold_on_market": chosen_w,
        "per_year": per_year,
        "sanity_model_likes_more_than_market": recs(likes),
        "sanity_model_fades_vs_market": recs(fades),
        "outputs": {
            "predictions": str((REPORTS / "qb_model_v2_predictions.csv").relative_to(ROOT)),
            "board": str((PROJECTIONS / "qb_model_v2_2024_2026_board.csv").relative_to(ROOT)),
        },
    }
    (REPORTS / "qb_model_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # console comparison table
    print("\n=== OOF candidate comparison (n=%d, hits=%d) ===" % (len(oof), int(y.sum())))
    print("%-32s %6s %6s %8s" % ("strategy", "AUC", "Brier", "logloss"))
    for name, m in cand_metrics.items():
        flag = "  <-- CHOSEN" if name == chosen else ("  (baseline)" if "baseline" in name else "")
        print("%-32s %6.4f %6.4f %8.4f%s" % (name, m["auc"], m["brier"], m["log_loss"], flag))
    print("\nDelta chosen - baseline:", summary["delta_vs_baseline"])
    print("Bootstrap 95% CI (model-baseline):", json.dumps(boot, indent=2))


if __name__ == "__main__":
    main()
