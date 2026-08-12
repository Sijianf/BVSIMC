"""Analysis utilities for real-data CV results (nested HP selection).

Raw data: trials x outer folds x HP combos x inner CV folds.
Load, summarize, tabulate (LaTeX), and plot method comparisons.
Import into a notebook or script; nothing runs at import time.

No ground-truth feature set exists for real data, so there are no
selection *metrics* (recall/precision vs known features). Selection
*stability* across folds is analyzed instead via get_feature_frequency().

Typical usage
-------------
    import utils_real as ur

    collected = ur.collect(PATH_OUTPUT, methods=["bissgl", "sgimc", "imc"],
                           val_metric="aupr", n_orig_U=9000, n_orig_V=33,
                           filters={"bissgl": {"xi": 1}})
    ur.overview(collected)
    disp, latex = ur.comparison_table(collected)
    ur.fig_comparison(collected, metrics=("aupr", "auc", "f1"), ylim=(0, 1.02))
    ur.make_report(collected, figdir=FIG_DIR, tabledir=TAB_DIR)
"""

import os
import glob
import gzip
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ====== score vector column names (matches get_metrics return order) ======
SCORE_COLS = ["aupr", "auc", "f1", "accuracy", "recall", "specificity", "precision"]

# ====== hyperparam columns per method (real-data schema) ======
HYPERPARAM_COLS = {
    "bissgl": ["xi", "eta", "tilde_lambda0", "tilde_lambda1",
               "lambda0", "lambda1", "K"],
    "sgimc":  ["C_lasso", "C_group", "C_ridge", "rank"],
    "imc":    ["C_lasso", "C_group", "C_ridge", "lamb", "rank"],
    "drimc":  ["lamU", "lamV", "cc", "iterpara", "numLat"],
    "nrlmf":  ["cfix", "K1", "K2", "num_factors", "lambda_d", "lambda_t",
               "alpha", "beta", "theta", "max_iter"],
}

# ====== display config (kept consistent with the simulation utils) ======
METHOD_LABEL = {"bissgl": "BVSIMC", "sgimc": "SGIMC", "imc": "IMC",
                "drimc": "DRIMC", "nrlmf": "NRLMF"}
METHOD_ORDER = ["bissgl", "sgimc", "imc", "drimc", "nrlmf"]

METRIC_LABEL = {"aupr": "AUPR", "auc": "AUC", "f1": "F1", "accuracy": "Accuracy",
                "recall": "Recall", "specificity": "Specificity",
                "precision": "Precision"}

_LATEX_PARAM = {"xi": r"\xi", "tilde_lambda0": r"\tilde{\lambda}_0",
                "lambda0": r"\lambda_0", "K": "K", "cc": "c", "cfix": "c",
                "lamU": r"\lambda_U", "lamV": r"\lambda_V", "rank": "r"}


def get_method_label(method, filters=None):
    """Display label, appending filtered hyperparameters in LaTeX.
    get_method_label('bissgl', {'xi': 1}) -> 'BVSIMC ($\\xi=1$)'."""
    base = METHOD_LABEL.get(method, method)
    if not filters:
        return base
    parts = [f"${_LATEX_PARAM.get(k, k)}={v}$" for k, v in filters.items()]
    return f"{base} ({', '.join(parts)})"


# modern, colourblind-safe palette (Okabe-Ito); BVSIMC salient, IMC muted baseline
STYLE = {
    "bissgl": dict(color="#D55E00", marker="o"),   # vermillion — proposed method
    "sgimc":  dict(color="#0072B2", marker="s"),   # blue
    "drimc":  dict(color="#009E73", marker="^"),   # bluish green
    "nrlmf":  dict(color="#CC79A7", marker="D"),   # reddish purple
    "imc":    dict(color="#9AA0A6", marker="v"),   # neutral grey — baseline
}

_INK = "#2b2b2b"
_MUTED = "#6b7280"
_GRID = "#e8eaed"
RC = {
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": _INK,
    "axes.edgecolor": "#c7cbd1",
    "axes.labelcolor": _INK,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
}


# ====== load results — single / task / local files ======
def load_results(output_dir, method_name):
    """Load real-data CV results for a method, concatenating any of:
        results_<method>.gz
        results_<method>_task*.gz      (HPC array)
        results_<method>_local_*.gz    (local notebooks)
    Returns one DataFrame; val_score / test_score become numpy arrays.
    One row per (trial, fold, HP combo, inner CV fold).
    """
    patterns = [
        f"results_{method_name}.gz",
        f"results_{method_name}_task*.gz",
        f"results_{method_name}_local_*.gz",
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(output_dir, pat))))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No results for '{method_name}' in {output_dir} "
            f"(looked for {patterns})"
        )

    rows = []
    for fname in files:
        with gzip.open(fname, "rb") as fin:
            rows.extend(pickle.load(fin))

    df = pd.DataFrame(rows)
    df["val_score"] = df["val_score"].apply(np.asarray)
    df["test_score"] = df["test_score"].apply(np.asarray)
    return df


# ====== expand a column of score vectors into named columns ======
def expand_scores(df, col, names=SCORE_COLS):
    """Expand a column of score vectors into individual named columns.
    e.g. 'val_score' -> 'val_score_aupr', 'val_score_auc', ..."""
    return pd.DataFrame(
        np.stack(df[col].values), index=df.index,
        columns=[f"{col}_{n}" for n in names],
    )


# ====== main summarization pipeline (per-trial averaging) ======
def summarize_results(df, hyperparam_cols, val_metric="aupr",
                      n_orig_U=None, n_orig_V=None):
    """Summarize real-application CV results with nested HP selection.

    Pipeline
    --------
    1. Average val_score across inner CV folds -> per (trial, fold, HP combo)
    2. Select best HP combo per (trial, fold) by val_metric
    3. Retrieve test_score for the best combo (identical across inner CV
       folds for the same combo) -> per (trial, fold)
    4. Average test_score across folds per trial -> per trial
    5. Mean +/- std across trials -> single-row final summary

    Parameters
    ----------
    df             : pd.DataFrame  raw results from load_results()
    hyperparam_cols: list[str]     HP column names for this method
    val_metric     : str           score metric for HP selection
    n_orig_U       : int or None   original row-side feature count (before
                                   identity augmentation); if given, d1_orig
                                   is recomputed excluding augmented indices
    n_orig_V       : int or None   original column-side feature count

    Returns
    -------
    dict with keys: raw, val_avg, best_params, test_per_fold,
                    test_per_trial, final
    """
    fold_cols = ["trial", "fold"]
    group_cols = fold_cols + hyperparam_cols

    # --- Step 1: average val_score across inner CV folds ---
    val_avg = df.groupby(group_cols, as_index=False).agg(
        val_score=("val_score", lambda x: np.mean(np.stack(x), axis=0)),
        val_d1=("val_d1", "mean"),
        val_d2=("val_d2", "mean"),
        n_inner_cv=("cv", "count"),  # diagnostic: should equal the inner-CV count
    )
    val_avg = pd.concat(
        [val_avg.drop(columns=["val_score"]), expand_scores(val_avg, "val_score")],
        axis=1,
    )
    val_metric_col = f"val_score_{val_metric}"

    # --- Step 2: select best HP combo per (trial, fold) ---
    best_idx = val_avg.groupby(fold_cols)[val_metric_col].idxmax()
    best_params = val_avg.loc[
        best_idx, fold_cols + hyperparam_cols + [val_metric_col, "n_inner_cv"]
    ].reset_index(drop=True)

    # --- Step 3: retrieve test_score for the best combo per (trial, fold) ---
    df_best = df.merge(
        best_params[fold_cols + hyperparam_cols],
        on=fold_cols + hyperparam_cols,
        how="inner",
    )

    # feature-selection index dicts may or may not exist
    base_cols = fold_cols + hyperparam_cols + ["test_score", "test_d1", "test_d2"]
    sel_cols_available = [
        c for c in ["test_sel_W", "test_sel_H", "test_sel_A", "test_sel_B"]
        if c in df_best.columns
    ]
    keep_cols = base_cols + sel_cols_available

    test_per_fold = df_best.groupby(fold_cols, as_index=False).first()[keep_cols]
    test_per_fold = pd.concat(
        [test_per_fold.drop(columns=["test_score"]),
         expand_scores(test_per_fold, "test_score")],
        axis=1,
    )

    # --- Step 3b: recompute d1/d2 excluding augmented identity features ---
    d1_sel_col = next(
        (c for c in ["test_sel_A", "test_sel_W"] if c in test_per_fold.columns), None
    )
    d2_sel_col = next(
        (c for c in ["test_sel_B", "test_sel_H"] if c in test_per_fold.columns), None
    )

    def _count_orig(d, n_orig):
        keys = d.keys() if isinstance(d, dict) else d
        return sum(1 for k in keys if int(k) < n_orig)

    if n_orig_U is not None and d1_sel_col is not None:
        test_per_fold["test_d1_orig"] = test_per_fold[d1_sel_col].apply(
            lambda d: _count_orig(d, n_orig_U)
        )
    if n_orig_V is not None and d2_sel_col is not None:
        test_per_fold["test_d2_orig"] = test_per_fold[d2_sel_col].apply(
            lambda d: _count_orig(d, n_orig_V)
        )

    # --- Step 4: average test_score across folds per trial ---
    test_score_cols = [f"test_score_{n}" for n in SCORE_COLS]
    has_d1_orig = "test_d1_orig" in test_per_fold.columns
    has_d2_orig = "test_d2_orig" in test_per_fold.columns

    agg_dict = {
        **{c: (c, "mean") for c in test_score_cols},
        "test_d1_mean": ("test_d1", "mean"),
        "test_d2_mean": ("test_d2", "mean"),
        "n_folds": ("fold", "count"),
    }
    if has_d1_orig:
        agg_dict["test_d1_orig_mean"] = ("test_d1_orig", "mean")
    if has_d2_orig:
        agg_dict["test_d2_orig_mean"] = ("test_d2_orig", "mean")

    test_per_trial = test_per_fold.groupby("trial", as_index=False).agg(**agg_dict)

    # --- Step 5: mean +/- std across trials ---
    final = {}
    for c in test_score_cols:
        vals = test_per_trial[c].values
        final[f"{c}_mean"] = vals.mean()
        final[f"{c}_std"] = vals.std(ddof=1)
    final["test_d1_mean"] = test_per_trial["test_d1_mean"].mean()
    final["test_d1_std"] = test_per_trial["test_d1_mean"].std(ddof=1)
    final["test_d2_mean"] = test_per_trial["test_d2_mean"].mean()
    final["test_d2_std"] = test_per_trial["test_d2_mean"].std(ddof=1)
    if has_d1_orig:
        final["test_d1_orig_mean"] = test_per_trial["test_d1_orig_mean"].mean()
        final["test_d1_orig_std"] = test_per_trial["test_d1_orig_mean"].std(ddof=1)
    if has_d2_orig:
        final["test_d2_orig_mean"] = test_per_trial["test_d2_orig_mean"].mean()
        final["test_d2_orig_std"] = test_per_trial["test_d2_orig_mean"].std(ddof=1)
    final["n_trials"] = len(test_per_trial)
    final["n_folds_per_trial"] = int(test_per_trial["n_folds"].iloc[0])
    final = pd.DataFrame([final])

    return {
        "raw": df,
        "val_avg": val_avg,
        "best_params": best_params,
        "test_per_fold": test_per_fold,
        "test_per_trial": test_per_trial,
        "final": final,
    }


# ====== convenience wrapper per method ======
def summarize_method(output_dir, method_name, val_metric="aupr", filters=None,
                     n_orig_U=None, n_orig_V=None):
    """Load + summarize a method in one call.

    Parameters
    ----------
    output_dir  : str   path to result files
    method_name : str   one of HYPERPARAM_COLS keys
    val_metric  : str   score metric for HP selection (default: 'aupr')
    filters     : dict  optional column filters, e.g. {"xi": 1}
    n_orig_U    : int   original row-side feature count (pre-augmentation)
    n_orig_V    : int   original column-side feature count

    Example
    -------
    res = summarize_method(PATH, "bissgl", filters={"xi": 1},
                           n_orig_U=9000, n_orig_V=33)
    print(res["final"])
    """
    hyperparam_cols = HYPERPARAM_COLS[method_name]
    df = load_results(output_dir, method_name)

    if filters is not None:
        for col, val in filters.items():
            df = df[df[col] == val]
        df = df.reset_index(drop=True)

    return summarize_results(df, hyperparam_cols, val_metric=val_metric,
                             n_orig_U=n_orig_U, n_orig_V=n_orig_V)


# ====== load + summarize every method present ======
def collect(output_dir, methods=METHOD_ORDER, val_metric="aupr",
            n_orig_U=None, n_orig_V=None, filters=None):
    """Return {method: summarize result dict} for every method whose result
    files are found in output_dir.

    `filters` is optional and per-method: {"bissgl": {"xi": 1}, ...}.
    """
    filters = filters or {}
    out = {}
    for m in methods:
        try:
            out[m] = summarize_method(output_dir, m, val_metric=val_metric,
                                      filters=filters.get(m),
                                      n_orig_U=n_orig_U, n_orig_V=n_orig_V)
            if filters.get(m):
                out[m]["_filters"] = dict(filters[m])
        except (FileNotFoundError, KeyError):
            pass
    return out


def add_variant(collected, output_dir, method, filters, key=None, label=None,
                style=None, val_metric="aupr", n_orig_U=None, n_orig_V=None):
    """Add a hyperparameter-restricted variant of `method` as its own entry,
    e.g. add_variant(collected, PATH, "bissgl", {"xi": 1}) -> key "bissgl_xi1".
    The variant inherits the base method's colour/marker (hatched in bars)."""
    key = key or (method + "_" + "_".join(f"{k}{v}" for k, v in filters.items()))
    summ = summarize_method(output_dir, method, val_metric=val_metric,
                            filters=filters,
                            n_orig_U=n_orig_U, n_orig_V=n_orig_V)
    summ["_base"] = method
    summ["_filters"] = dict(filters)
    summ["_label"] = label
    summ["_style"] = style
    collected[key] = summ
    return key


def _label_for(key, collected):
    """Display/LaTeX label for a key (base method or add_variant key)."""
    summ = collected.get(key, {})
    if summ.get("_label"):
        return summ["_label"]
    return get_method_label(summ.get("_base", key), summ.get("_filters"))


def _style_for(key, collected):
    summ = collected.get(key, {})
    if summ.get("_style"):
        return summ["_style"]
    base = summ.get("_base", key)
    return STYLE.get(base, dict(color="#7f7f7f", marker="o"))


# ====== interactive analysis helpers ======
def overview(collected, methods=None):
    """One-row-per-method sanity table: trials, outer folds, and whether
    feature-selection dicts are available."""
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    rows = []
    for m in methods:
        res = collected[m]
        tpf = res["test_per_fold"]
        rows.append({
            "method": _label_for(m, collected),
            "key": m,
            "trials": tpf["trial"].nunique(),
            "outer_folds": len(tpf),
            "sel_dicts": any(c.startswith("test_sel_") for c in tpf.columns),
        })
    return pd.DataFrame(rows)


def combined_per_fold(collected, methods=None):
    """All methods' per-(trial, fold) test scores stacked into one tidy frame
    with a `method` column — the handle for custom analysis / plotting / tests."""
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    score_cols = [f"test_score_{n}" for n in SCORE_COLS]
    frames = []
    for m in methods:
        cols = ["trial", "fold"] + score_cols + ["test_d1", "test_d2"]
        tpf = collected[m]["test_per_fold"]
        df = tpf[[c for c in cols if c in tpf.columns]].copy()
        df.insert(0, "method", _label_for(m, collected))
        df.insert(1, "method_key", m)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pairwise_wilcoxon(collected, reference="bissgl", metric="aupr", methods=None):
    """Paired Wilcoxon signed-rank of `reference` vs each other method, paired
    on (trial, fold) — valid because all methods share the same CV splits.
    With 5 trials x 10 folds this pairs 50 observations per comparison; note
    folds within a trial share training data, so treat p-values as indicative."""
    from scipy.stats import wilcoxon
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    if reference not in collected:
        raise KeyError(f"reference '{reference}' not in collected results")
    col = f"test_score_{metric}"
    keys = ["trial", "fold"]
    ref = collected[reference]["test_per_fold"][keys + [col]]
    rows = []
    for m in methods:
        if m == reference:
            continue
        oth = collected[m]["test_per_fold"][keys + [col]]
        mg = ref.merge(oth, on=keys, suffixes=("_ref", "_oth"))
        a, b = mg[f"{col}_ref"].values, mg[f"{col}_oth"].values
        try:
            _, p = wilcoxon(a, b)
        except ValueError:      # zero differences etc.
            p = np.nan
        rows.append({"vs": _label_for(m, collected),
                     "n_pairs": len(mg),
                     "median_diff": float(np.median(a - b)),
                     "ref_wins_frac": float(np.mean(a > b)),
                     "p_value": p})
    return pd.DataFrame(rows).sort_values("vs").reset_index(drop=True)


# ====== LaTeX helpers ======
def _fmt(m, s, prec):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return f"{m:.{prec}f}"
    return f"{m:.{prec}f} ({s:.{prec}f})"


def _booktabs(header_rows, body_rows, colfmt, caption, label):
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule"]
    lines += header_rows
    lines.append("\\midrule")
    lines += body_rows
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ====== comparison table: rows = metrics, cols = methods ======
def comparison_table(collected, metrics=SCORE_COLS, methods=None, prec=3,
                     bold_best=True, include_d=False, caption=None, label=None):
    """One table for the whole real-data experiment: mean +/- std per method
    for each metric (best per row in bold). Returns (display_df, latex_str).

    include_d=True appends d1/d2 rows (selected feature counts; the *_orig
    variant excluding augmented identity features is used when available).
    """
    methods = methods or [m for m in METHOD_ORDER if m in collected]

    def _mean_std(m, metric):
        f = collected[m]["final"]
        return (f[f"test_score_{metric}_mean"].iloc[0],
                f[f"test_score_{metric}_std"].iloc[0])

    disp = pd.DataFrame(index=[METRIC_LABEL.get(mt, mt) for mt in metrics])
    disp.index.name = "Metric"
    body = []
    for mt in metrics:
        stats = {m: _mean_std(m, mt) for m in methods}
        best = (max(stats, key=lambda m: stats[m][0]) if bold_best else None)
        cells = []
        for m in methods:
            txt = _fmt(*stats[m], prec)
            disp.loc[METRIC_LABEL.get(mt, mt), _label_for(m, collected)] = txt
            cells.append(f"\\textbf{{{txt}}}" if m == best else txt)
        body.append(" & ".join([METRIC_LABEL.get(mt, mt)] + cells) + " \\\\")

    if include_d:
        any_orig = False
        for d_key, d_name in [("test_d1", "$d_1$"), ("test_d2", "$d_2$")]:
            cells = []
            for m in methods:
                f = collected[m]["final"]
                base = (f"{d_key}_orig" if f"{d_key}_orig_mean" in f.columns
                        else d_key)
                any_orig = any_orig or base.endswith("_orig")
                mean = f[f"{base}_mean"].iloc[0]
                std = (f[f"{base}_std"].iloc[0]
                       if f"{base}_std" in f.columns else None)
                cells.append(_fmt(mean, std, prec=1))
            for m, txt in zip(methods, cells):
                disp.loc[d_name, _label_for(m, collected)] = txt
            body.append(" & ".join([d_name] + cells) + " \\\\")
        d_note = (" $d_1$/$d_2$ exclude identity-augmented features"
                  " where selection info is available." if any_orig else "")
    else:
        d_note = ""

    # subtitle from the reference method's final frame
    f0 = collected[methods[0]]["final"]
    how = (f"mean (sd) over {int(f0['n_trials'].iloc[0])} trials of "
           f"{int(f0['n_folds_per_trial'].iloc[0])}-fold CV")

    header = " & ".join(["Metric"] + [_label_for(m, collected)
                                      for m in methods]) + " \\\\"
    colfmt = "l" + "c" * len(methods)
    caption = caption or (f"Test performance on real data ({how}); "
                          f"best per row in bold.{d_note}")
    latex = _booktabs([header], body, colfmt, caption, label or "tab:realdata")
    return disp, latex


# ====== shared axis styling ======
def _style_axis(ax):
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 4))
        ax.spines[side].set_color("#c7cbd1")
    ax.grid(axis="y", linestyle="-", linewidth=0.9, color=_GRID)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5, width=1.0, colors=_MUTED, labelcolor=_INK,
                   labelsize=10)


# ====== figure: grouped bars, mean +/- std per method for each metric ======
def fig_comparison(collected, metrics=("aupr", "auc", "f1"), methods=None,
                   ylim=None, figsize=None, savepath=None):
    """Grouped bar chart: one group per metric, one bar per method, with
    std error bars. The real-data analog of the simulation curves (no x-axis
    sweep exists, so bars replace curves)."""
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    metrics = list(metrics)
    figsize = figsize or (1.1 + 1.35 * len(metrics) * len(methods) / 5, 4.3)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=figsize)
        n_m = len(methods)
        width = 0.8 / n_m
        x = np.arange(len(metrics))

        for i, m in enumerate(methods):
            f = collected[m]["final"]
            means = [f[f"test_score_{mt}_mean"].iloc[0] for mt in metrics]
            stds = [f[f"test_score_{mt}_std"].iloc[0] for mt in metrics]
            style = _style_for(m, collected)
            hatch = "///" if collected[m].get("_filters") and "_base" in collected[m] else None
            ax.bar(x + (i - (n_m - 1) / 2) * width, means, width * 0.92,
                   yerr=stds, capsize=2.5, label=_label_for(m, collected),
                   color=style["color"], alpha=0.9, hatch=hatch,
                   edgecolor="white", linewidth=0.6,
                   error_kw=dict(ecolor=_INK, elinewidth=1.0, alpha=0.7))

        _style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABEL.get(mt, mt) for mt in metrics],
                           fontsize=11)
        ax.set_ylabel("Score", fontsize=12, labelpad=7)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14),
                  ncol=min(len(methods), 5), frameon=False,
                  handlelength=1.4, columnspacing=1.6, fontsize=10)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight",
                        facecolor="white")
        return fig


# ====== diagnostics ======
def check_completeness(results, method_name, expected_trials=None,
                       expected_folds=None):
    """Print diagnostic info to verify results are complete.

    If expected_trials / expected_folds are None, the expected grid is derived
    from the trial and fold IDs actually present (so 1-indexed IDs are fine);
    pass ints to check against a fixed 0-indexed grid instead.
    """
    df = results["raw"]
    tpf = results["test_per_fold"]

    hyperparam_cols = HYPERPARAM_COLS[method_name]
    hp_combos = df.groupby(hyperparam_cols).ngroups
    inner_counts = df.groupby(["trial", "fold"] + hyperparam_cols)["cv"].count()

    if expected_trials is None:
        trial_ids = sorted(df["trial"].unique())
    else:
        trial_ids = list(range(expected_trials))
    if expected_folds is None:
        fold_ids = sorted(df["fold"].unique())
    else:
        fold_ids = list(range(expected_folds))
    expected_outer = len(trial_ids) * len(fold_ids)

    print(f"\n{'=' * 60}")
    print(f"  Diagnostics: {method_name}")
    print(f"{'=' * 60}")
    print(f"  Raw result rows:            {len(df)}")
    print(f"  HP combos:                  {hp_combos}")
    print(f"  Inner CV per combo (min/max): "
          f"{inner_counts.min()}/{inner_counts.max()}")
    print(f"  Outer folds with results:   {len(tpf)} / {expected_outer} expected")

    all_pairs = set((t, f) for t in trial_ids for f in fold_ids)
    present_pairs = set(zip(tpf["trial"], tpf["fold"]))
    missing = all_pairs - present_pairs
    if missing:
        print(f"  *** MISSING folds: {sorted(missing)}")
    else:
        print(f"  All {expected_outer} outer folds present")
    print(f"{'=' * 60}\n")


# ====== pretty-print single method ======
def print_summary(results, method_label="Method"):
    """Print a compact summary of one method's final results."""
    final = results["final"]
    n_trials = int(final["n_trials"].iloc[0])
    n_folds = int(final["n_folds_per_trial"].iloc[0])
    subtitle = f"{n_trials} trials x {n_folds}-fold CV"

    print(f"\n{'=' * 60}")
    print(f"  {method_label}  --  {subtitle}")
    print(f"{'=' * 60}")
    for metric in SCORE_COLS:
        m = final[f"test_score_{metric}_mean"].iloc[0]
        s = final[f"test_score_{metric}_std"].iloc[0]
        print(f"  {metric:>12s}:  {m:.4f} +/- {s:.4f}")

    d1_total = final["test_d1_mean"].iloc[0]
    d2_total = final["test_d2_mean"].iloc[0]
    if "test_d1_orig_mean" in final.columns:
        d1_orig = final["test_d1_orig_mean"].iloc[0]
        print(f"  {'d1 (orig)':>12s}:  {d1_orig:.1f}  "
              f"(total incl. augmented: {d1_total:.1f})")
    else:
        print(f"  {'d1 (mean)':>12s}:  {d1_total:.1f}")
    if "test_d2_orig_mean" in final.columns:
        d2_orig = final["test_d2_orig_mean"].iloc[0]
        print(f"  {'d2 (orig)':>12s}:  {d2_orig:.1f}  "
              f"(total incl. augmented: {d2_total:.1f})")
    else:
        print(f"  {'d2 (mean)':>12s}:  {d2_total:.1f}")
    print(f"{'=' * 60}\n")


# ====== feature selection frequency analysis (stability across folds) ======
def get_feature_frequency(results, sel_col, feature_names=None, n_orig=None):
    """Count selection frequency and aggregate importance (L2 norm) across
    outer folds. `sel_col` holds dicts of {feature_idx: L2_row_norm}
    (BiSSGL: test_sel_A / test_sel_B, SGIMC: test_sel_W / test_sel_H).

    n_orig, if given, excludes identity-augmented indices (idx >= n_orig).
    Returns a DataFrame sorted by count (desc) with columns:
    feature_idx, [feature_name], count, frequency, norm_mean, norm_std.
    """
    tpf = results["test_per_fold"]
    if sel_col not in tpf.columns:
        available = [c for c in tpf.columns if c.startswith("test_sel_")]
        raise KeyError(f"Column '{sel_col}' not found in test_per_fold. "
                       f"Available selection columns: {available}")

    n_folds = len(tpf)
    norms_by_idx = defaultdict(list)
    for sel in tpf[sel_col]:
        if isinstance(sel, dict):
            items = ((int(k), float(v)) for k, v in sel.items())
        else:  # backward compatibility: plain list of indices
            items = ((int(k), np.nan) for k in sel)
        for idx, norm in items:
            if n_orig is not None and idx >= n_orig:
                continue
            norms_by_idx[idx].append(norm)

    rows = []
    for feat_idx in sorted(norms_by_idx):
        norms = norms_by_idx[feat_idx]
        row = {
            "feature_idx": feat_idx,
            "count": len(norms),
            "frequency": len(norms) / n_folds,
            "norm_mean": float(np.nanmean(norms)),
            "norm_std": (float(np.nanstd(norms, ddof=1))
                         if len(norms) > 1 else 0.0),
        }
        if feature_names is not None:
            row["feature_name"] = feature_names[feat_idx]
        rows.append(row)

    freq_df = pd.DataFrame(rows)
    if len(freq_df) == 0:
        return freq_df

    col_order = ["feature_idx"]
    if feature_names is not None:
        col_order.append("feature_name")
    col_order += ["count", "frequency", "norm_mean", "norm_std"]
    # sort by frequency, breaking ties by mean importance when selected
    return (freq_df[col_order]
            .sort_values(["count", "norm_mean"], ascending=[False, False])
            .reset_index(drop=True))


def sel_norm_matrix(results, sel_col, feature_names=None, n_orig=None,
                    fill=np.nan):
    """Expand a selection column into a (outer fold x feature) norm matrix.

    Unlike score vectors, the test_sel_* columns hold ragged dicts
    {feature_idx: L2_row_norm} whose keys differ per fold, so expand_scores
    (which needs fixed-length, fixed-order vectors) cannot be used. This
    unions the keys instead: one row per (trial, fold), one column per
    feature ever selected, `fill` (default NaN) where a feature was not
    selected in that fold. Use fill=0.0 to treat non-selection as zero norm.

    Returns a DataFrame indexed by (trial, fold), columns sorted by
    feature index (renamed via feature_names if given).
    """
    tpf = results["test_per_fold"]
    if sel_col not in tpf.columns:
        available = [c for c in tpf.columns if c.startswith("test_sel_")]
        raise KeyError(f"Column '{sel_col}' not found in test_per_fold. "
                       f"Available selection columns: {available}")

    records = []
    for sel in tpf[sel_col]:
        if isinstance(sel, dict):
            rec = {int(k): float(v) for k, v in sel.items()}
        else:  # backward compatibility: plain list of indices
            rec = {int(k): 1.0 for k in sel}
        if n_orig is not None:
            rec = {k: v for k, v in rec.items() if k < n_orig}
        records.append(rec)

    mat = pd.DataFrame(records,
                       index=pd.MultiIndex.from_frame(tpf[["trial", "fold"]]))
    mat = mat[sorted(mat.columns)]
    if not np.isnan(fill):
        mat = mat.fillna(fill)
    if feature_names is not None:
        mat.columns = [feature_names[i] for i in mat.columns]
    return mat


# ====== grid-pooled selection frequency (Komodromos et al. 2022 style) ======
def _infer_hyperparam_cols(results):
    """Recover the hyperparameter columns from results['best_params']
    (everything that is not an id/diagnostic/score column)."""
    skip = {"trial", "fold", "n_inner_cv"}
    return [c for c in results["best_params"].columns
            if c not in skip and not c.startswith("val_score")]


def _fits_frame(results, sel_col, hyperparam_cols=None):
    """One row per fit = (trial, fold, HP combo), with its test selection
    dict. test_sel_* is identical across inner CV folds for the same combo
    (like test_score), so the first inner-CV row is taken per fit."""
    raw = results["raw"]
    if sel_col not in raw.columns:
        available = [c for c in raw.columns if c.startswith("test_sel_")]
        raise KeyError(f"Column '{sel_col}' not found in raw results. "
                       f"Available selection columns: {available}")
    hyperparam_cols = hyperparam_cols or _infer_hyperparam_cols(results)
    fit_cols = ["trial", "fold"] + hyperparam_cols
    fits = (raw.groupby(fit_cols, as_index=False, sort=False)
               .agg(**{sel_col: (sel_col, "first")}))
    return fits, hyperparam_cols


def _sel_items(sel, n_orig):
    """Normalize one selection entry to (idx, norm) pairs."""
    if isinstance(sel, dict):
        items = ((int(k), float(v)) for k, v in sel.items())
    else:  # backward compatibility: plain list of indices
        items = ((int(k), np.nan) for k in sel)
    if n_orig is not None:
        items = ((i, v) for i, v in items if i < n_orig)
    return items


def get_feature_frequency_grid(results, sel_col, hyperparam_cols=None,
                               feature_names=None, n_orig=None):
    """Selection frequency pooled over the full hyperparameter grid AND all
    outer folds/trials, mirroring the 'selection proportion' of Komodromos
    et al. (2022, Bioinformatics 38:3918), whose denominator was
    (14 lambda values x 10 folds) = 140 fits.

    Here: frequency = (# fits selecting the feature) / n_fits, where a fit
    is one (trial, outer fold, HP combo) and n_fits = trials x folds x grid.

    CAVEAT: pooling over the grid is only meaningful if selection is not
    too sensitive to the hyperparameters (as Komodromos et al. verified for
    lambda). When the spike penalty strongly drives the support -- as for
    tilde_lambda0 in BiSSGL -- the pooled number partly reflects the grid
    choice; pair this with get_feature_frequency() (frequency at the
    CV-selected HP, data-stability only) and/or the stratified view from
    get_feature_frequency_by_hp() when reporting.

    Returns a DataFrame sorted by count desc with columns:
    feature_idx, [feature_name], count, frequency, norm_mean, norm_std,
    where norms aggregate over the fits in which the feature was selected.
    """
    fits, _ = _fits_frame(results, sel_col, hyperparam_cols)
    n_fits = len(fits)

    norms_by_idx = defaultdict(list)
    for sel in fits[sel_col]:
        for idx, norm in _sel_items(sel, n_orig):
            norms_by_idx[idx].append(norm)

    rows = []
    for feat_idx in sorted(norms_by_idx):
        norms = norms_by_idx[feat_idx]
        row = {
            "feature_idx": feat_idx,
            "count": len(norms),
            "frequency": len(norms) / n_fits,
            "norm_mean": float(np.nanmean(norms)),
            "norm_std": (float(np.nanstd(norms, ddof=1))
                         if len(norms) > 1 else 0.0),
        }
        if feature_names is not None:
            row["feature_name"] = feature_names[feat_idx]
        rows.append(row)

    freq_df = pd.DataFrame(rows)
    if len(freq_df) == 0:
        return freq_df
    col_order = ["feature_idx"]
    if feature_names is not None:
        col_order.append("feature_name")
    col_order += ["count", "frequency", "norm_mean", "norm_std"]
    return (freq_df[col_order]
            .sort_values(["count", "norm_mean"], ascending=[False, False])
            .reset_index(drop=True))


def get_feature_frequency_by_hp(results, sel_col, stratify_by,
                                hyperparam_cols=None, feature_names=None,
                                n_orig=None, top_n=None, include_norm=False):
    """Selection frequency stratified by one hyperparameter: one column per
    value of `stratify_by` (e.g. 'tilde_lambda0'), frequency computed over
    the fits (trials x folds x remaining grid) within that stratum.

    This is the honest companion to get_feature_frequency_grid() when
    selection is hyperparameter-sensitive: consistent rows across columns
    support a stability claim; rows that light up only at small penalties
    reveal grid dependence that pooling would average away.

    n_orig, if given, excludes identity-augmented indices (idx >= n_orig)
    from counts and denominators, as in the other frequency functions.

    Returns a DataFrame indexed by feature (name if feature_names given,
    else index), columns = sorted values of `stratify_by`, plus a 'pooled'
    column (and 'norm_mean' if include_norm=True); rows sorted by pooled
    frequency desc with ties broken by mean norm when selected -- the same
    order as get_feature_frequency_grid() -- truncated to top_n.
    """
    fits, hyperparam_cols = _fits_frame(results, sel_col, hyperparam_cols)
    if stratify_by not in hyperparam_cols:
        raise KeyError(f"'{stratify_by}' is not a hyperparameter column "
                       f"({hyperparam_cols})")

    counts = defaultdict(lambda: defaultdict(int))   # idx -> stratum -> count
    norms_by_idx = defaultdict(list)                 # idx -> pooled norms
    n_by_stratum = defaultdict(int)
    for stratum, sel in zip(fits[stratify_by], fits[sel_col]):
        n_by_stratum[stratum] += 1
        for idx, norm in _sel_items(sel, n_orig):
            counts[idx][stratum] += 1
            norms_by_idx[idx].append(norm)

    strata = sorted(n_by_stratum)
    n_fits = sum(n_by_stratum.values())
    rows, index = [], []
    for feat_idx in sorted(counts):
        by_s = counts[feat_idx]
        row = {s: by_s.get(s, 0) / n_by_stratum[s] for s in strata}
        row["pooled"] = sum(by_s.values()) / n_fits
        row["norm_mean"] = float(np.nanmean(norms_by_idx[feat_idx]))
        rows.append(row)
        index.append(feature_names[feat_idx] if feature_names is not None
                     else feat_idx)

    out = pd.DataFrame(rows, index=index,
                       columns=strata + ["pooled", "norm_mean"])
    out.index.name = "feature"
    out = out.sort_values(["pooled", "norm_mean"], ascending=[False, False])
    if not include_norm:
        out = out.drop(columns=["norm_mean"])
    return out.head(top_n) if top_n else out


def feature_selection_table(results, sel_col, hyperparam_cols=None,
                            feature_names=None, n_orig=None, top_n=15,
                            prec=2, side_label="row-side features",
                            caption=None, label=None, savepath=None):
    """Publication table of the most frequently selected features, reporting
    BOTH selection proportions side by side:

      - Freq. (grid):        pooled over trials x folds x full HP grid
                             (Komodromos et al. 2022 style; rank order)
      - Freq. (selected HP): across outer folds at the CV-selected HP combo
                             only (pure data-resampling stability)
      - ||row||_2:           mean (sd) coefficient-row norm over the fits in
                             which the feature was selected (grid pooling)

    Returns (display_df, latex_str); writes the .tex if savepath is given.
    """
    grid = get_feature_frequency_grid(results, sel_col, hyperparam_cols,
                                      feature_names=feature_names,
                                      n_orig=n_orig)
    sel = get_feature_frequency(results, sel_col,
                                feature_names=feature_names, n_orig=n_orig)
    if len(grid) == 0:
        raise ValueError(f"No features ever selected in '{sel_col}'.")

    sel_freq = dict(zip(sel["feature_idx"], sel["frequency"])) if len(sel) \
        else {}
    n_fits = int(round(grid["count"].iloc[0] / grid["frequency"].iloc[0]))
    n_outer = len(results["test_per_fold"])

    top = grid.head(top_n)
    name_col = "feature_name" if feature_names is not None else "feature_idx"

    disp = pd.DataFrame({
        "Feature": top[name_col].values,
        "Freq. (grid)": top["frequency"].round(prec + 1).values,
        "Freq. (selected HP)": [round(sel_freq.get(i, 0.0), prec + 1)
                                for i in top["feature_idx"]],
        "Norm": [_fmt(m, s, prec) for m, s in
                 zip(top["norm_mean"], top["norm_std"])],
    })

    header = ("Feature & Freq. (grid) & Freq. (sel. HP) & "
              "$\\|A_{j\\cdot}\\|_2$ \\\\")
    body = []
    for _, r in disp.iterrows():
        feat_tex = str(r["Feature"]).replace("_", "\\_")
        body.append(f"{feat_tex} & {r['Freq. (grid)']:.{prec}f} & "
                    f"{r['Freq. (selected HP)']:.{prec}f} & {r['Norm']} \\\\")
    caption = caption or (
        f"Most frequently selected {side_label}. Freq.\\ (grid): proportion "
        f"of the {n_fits} fits (trials $\\times$ outer folds $\\times$ "
        f"hyperparameter grid) selecting the feature; Freq.\\ (sel.\\ HP): "
        f"proportion of the {n_outer} outer folds at the CV-selected "
        f"hyperparameters; norm is the mean (sd) coefficient-row $\\ell_2$ "
        f"norm when selected."
    )
    latex = _booktabs([header], body, "lccc", caption,
                      label or "tab:selection_freq")
    if savepath:
        with open(savepath, "w") as f:
            f.write(latex + "\n")
    return disp, latex


def print_top_features(freq_df, top_n=20, label="Features"):
    """Print the top-N most frequently selected features with importance."""
    if len(freq_df) == 0:
        print(f"No selected features to report for {label}.")
        return
    n_total = int(round(freq_df["count"].iloc[0] / freq_df["frequency"].iloc[0]))
    print(f"\n{'=' * 70}")
    print(f"  Top {top_n} {label}  (out of {n_total} fits)")
    print(f"{'=' * 70}")
    has_name = "feature_name" in freq_df.columns
    has_norm = ("norm_mean" in freq_df.columns
                and not freq_df["norm_mean"].isna().all())
    for _, row in freq_df.head(top_n).iterrows():
        name_str = f"  {row['feature_name']}" if has_name else ""
        norm_str = (f"  ||r||={row['norm_mean']:.3f}+/-{row['norm_std']:.3f}"
                    if has_norm else "")
        # note: iterrows() casts all-numeric rows to float, so re-int here
        print(f"  idx {int(row['feature_idx']):>5d}{name_str:30s}  "
              f"count={int(row['count']):>3d}  "
              f"freq={row['frequency']:.0%}{norm_str}")
    print(f"{'=' * 70}\n")


# ====== figure: selection-stability heatmap (outer folds x top features) ======
def fig_sel_heatmap(results, sel_col, top_n=30, feature_names=None, n_orig=None,
                    method_label=None, cmap="Oranges", orient="horizontal",
                    figsize=None, savepath=None):
    """Heatmap of coefficient-row L2 norms across outer folds, restricted to
    the top_n features by selection frequency (ties broken by mean norm — same
    order as print_top_features). Empty cells = not selected in that fold.
    Thin lines separate trials.

    orient : "horizontal" -> folds on y, features on x (wide; good for few
                             features, e.g. the drug/row side)
             "vertical"   -> features on y, folds on x (tall; good for many
                             features, e.g. the disease/target side)

    Example
    -------
    fig_sel_heatmap(collected["bissgl"], "test_sel_B", top_n=40,
                    orient="vertical", method_label="BVSIMC, disease side")
    """
    if orient not in ("horizontal", "vertical"):
        raise ValueError("orient must be 'horizontal' or 'vertical'")

    freq = get_feature_frequency(results, sel_col, feature_names=feature_names,
                                 n_orig=n_orig)
    if len(freq) == 0:
        return None
    name_col = "feature_name" if feature_names is not None else "feature_idx"
    top = freq.head(top_n)
    mat = sel_norm_matrix(results, sel_col, feature_names=feature_names,
                          n_orig=n_orig)
    mat = mat[top[name_col].tolist()]                # order = freq, then norm

    n_folds, n_feat = mat.shape
    trials = mat.index.get_level_values("trial").values
    bounds = np.flatnonzero(np.diff(trials)) + 1     # trial boundaries
    edges = np.r_[0, bounds, n_folds]
    trial_starts = np.r_[0, bounds].astype(int)
    fold_ticks = (edges[:-1] + edges[1:]) / 2
    fold_labels = [f"trial {t}" for t in trials[trial_starts]]

    # data grid is (fold, feature); transpose for the vertical layout
    grid = mat.values if orient == "horizontal" else mat.values.T
    masked = np.ma.masked_invalid(grid)
    cbar_label = r"$\|A_{j\cdot}\|_2$ (coefficient-row norm)"
    title = "Selected features across outer folds"
    if method_label:
        title += f" — {method_label}"
    else:
        title = None

    with plt.rc_context(RC):
        if orient == "horizontal":
            figsize = figsize or (max(6.0, 0.22 * n_feat + 1.8),
                                  max(3.6, 0.09 * n_folds + 1.6))
        else:
            figsize = figsize or (max(4.5, 0.10 * n_folds + 1.8),
                                  max(4.0, 0.20 * n_feat + 1.6))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(masked, cmap=cmap, edgecolors="white",
                           linewidth=0.4, vmin=0)

        if orient == "horizontal":
            ax.invert_yaxis()                        # fold 0 at the top
            for b in bounds:
                ax.axhline(b, color=_INK, linewidth=0.9)
            ax.set_yticks(fold_ticks)
            ax.set_yticklabels(fold_labels, fontsize=9)
            ax.set_xticks(np.arange(n_feat) + 0.5)
            ax.set_xticklabels(mat.columns, rotation=90, fontsize=8)
        else:                                        # vertical
            ax.invert_yaxis()                        # top feature at the top
            for b in bounds:
                ax.axvline(b, color=_INK, linewidth=0.9)
            ax.set_xticks(fold_ticks)
            ax.set_xticklabels(fold_labels, fontsize=9, rotation=90)
            ax.set_yticks(np.arange(n_feat) + 0.5)
            ax.set_yticklabels(mat.columns, fontsize=8)

        ax.tick_params(length=0, colors=_MUTED, labelcolor=_INK)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=12, pad=10, color=_INK, loc="left")

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight",
                        facecolor="white")
        return fig



def make_tables(collected, savedir, methods=None, metrics=SCORE_COLS,
                include_d=True, prec=3, verbose=True):
    """Write the comparison table (.tex) to `savedir`.
    Returns {"comparison": (display_df, latex_str)}."""
    os.makedirs(savedir, exist_ok=True)
    disp, latex = comparison_table(collected, metrics=metrics, methods=methods,
                                   include_d=include_d, prec=prec)
    with open(os.path.join(savedir, "table_comparison.tex"), "w") as f:
        f.write(latex + "\n")
    if verbose:
        print(f"Table written to {savedir}/table_comparison.tex")
    return {"comparison": (disp, latex)}


def make_figures(collected, savedir, methods=None,
                 metrics=("aupr", "auc", "f1"), ylim=None,
                 formats=("pdf", "png"), verbose=True):
    """Write the comparison bar figure to `savedir` in each format."""
    os.makedirs(savedir, exist_ok=True)
    for ext in formats:
        fig_comparison(collected, metrics=metrics, methods=methods, ylim=ylim,
                       savepath=os.path.join(savedir, f"fig_comparison.{ext}"))
    plt.close("all")
    if verbose:
        print(f"Figures written to {savedir}/ ({', '.join(formats)})")


def make_report(collected, savedir=None, figdir=None, tabledir=None,
                methods=None, table_metrics=SCORE_COLS,
                fig_metrics_=("aupr", "auc", "f1"), include_d=True,
                ylim=None, formats=("pdf", "png"), prec=3):
    """Export the comparison table and figure in one call.

    Destinations
    ------------
    - make_report(collected, savedir="report")            -> everything in one dir
    - make_report(collected, figdir=..., tabledir=...)    -> separate dirs
    Explicit figdir/tabledir override savedir when both are given.

    Returns the {name: (display_df, latex_str)} dict from make_tables.
    """
    figdir = figdir or savedir
    tabledir = tabledir or savedir
    if figdir is None or tabledir is None:
        raise ValueError("Provide savedir, or both figdir and tabledir.")

    tables = make_tables(collected, tabledir, methods=methods,
                         metrics=table_metrics, include_d=include_d, prec=prec)
    make_figures(collected, figdir, methods=methods, metrics=fig_metrics_,
                 ylim=ylim, formats=formats)
    return tables