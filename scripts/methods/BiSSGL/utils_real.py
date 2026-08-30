"""Analysis utilities for real-data CV results (nested HP selection).

Raw data: trials x outer folds x HP combos x inner CV folds.
Load, summarize, tabulate (LaTeX), and plot method comparisons.
Import into a notebook or script; nothing runs at import time.

No ground-truth feature set exists for real data, so there are no
selection *metrics* (recall/precision vs known features). Selection
*stability* across folds is analyzed instead in utils_selection.py.

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


def make_tables(collected, savedir, methods=None, metrics=SCORE_COLS,
                include_d=True, prec=3, table_name="table_comparison",
                caption=None, label=None, verbose=True):
    """Write the comparison table (.tex) to `savedir`.

    table_name : base filename WITHOUT extension (default 'table_comparison');
                 a trailing '.tex' is stripped if given
    caption, label : passed to comparison_table; label defaults to
                 'tab:<table_name>' so LaTeX \\ref names follow the filename
    Returns {"comparison": (display_df, latex_str)}."""
    os.makedirs(savedir, exist_ok=True)
    table_name = str(table_name)
    if table_name.endswith(".tex"):
        table_name = table_name[:-4]
    disp, latex = comparison_table(collected, metrics=metrics, methods=methods,
                                   include_d=include_d, prec=prec,
                                   caption=caption,
                                   label=label or f"tab:{table_name}")
    path = os.path.join(savedir, f"{table_name}.tex")
    with open(path, "w") as f:
        f.write(latex + "\n")
    if verbose:
        print(f"Table written to {path}")
    return {"comparison": (disp, latex)}


def make_figures(collected, savedir, methods=None,
                 metrics=("aupr", "auc", "f1"), ylim=None,
                 formats=("pdf", "png"), fig_name="fig_comparison",
                 verbose=True):
    """Write the comparison bar figure to `savedir` in each format.

    fig_name : base filename WITHOUT extension (default 'fig_comparison');
               an extension matching one of `formats` is stripped if given."""
    os.makedirs(savedir, exist_ok=True)
    fig_name = str(fig_name)
    root, ext = os.path.splitext(fig_name)
    if ext.lstrip(".") in formats:
        fig_name = root
    for ext in formats:
        fig_comparison(collected, metrics=metrics, methods=methods, ylim=ylim,
                       savepath=os.path.join(savedir, f"{fig_name}.{ext}"))
    plt.close("all")
    if verbose:
        print(f"Figures written to {savedir}, as {fig_name}.* "
              f"({', '.join(formats)})")


def make_report(collected, savedir=None, figdir=None, tabledir=None,
                methods=None, table_metrics=SCORE_COLS,
                fig_metrics_=("aupr", "auc", "f1"), include_d=True,
                ylim=None, formats=("pdf", "png"), prec=3,
                table_name="table_comparison", fig_name="fig_comparison",
                caption=None, label=None, name=None):
    """Export the comparison table and figure in one call.

    Destinations
    ------------
    - make_report(collected, savedir="report")            -> everything in one dir
    - make_report(collected, figdir=..., tabledir=...)    -> separate dirs
    Explicit figdir/tabledir override savedir when both are given.

    Output names
    ------------
    name       : shorthand suffix applied to both defaults, e.g. name="tb"
                 -> table_comparison_tb.tex, fig_comparison_tb.pdf/png
    table_name : full base name for the .tex (overrides `name`)
    fig_name   : full base name for the figure files (overrides `name`)
    caption    : custom LaTeX caption for the table
    label      : LaTeX label (default 'tab:<table_name>')

    Examples
    --------
    make_report(collected, savedir=OUT, name="tb_main")
    make_report(collected, savedir=OUT, table_name="table_tb_performance",
                fig_name="fig_tb_performance",
                caption="Predictive performance on the tuberculosis data.",
                label="tab:tb_performance")

    Returns the {name: (display_df, latex_str)} dict from make_tables.
    """
    figdir = figdir or savedir
    tabledir = tabledir or savedir
    if figdir is None or tabledir is None:
        raise ValueError("Provide savedir, or both figdir and tabledir.")
    if name:
        if table_name == "table_comparison":
            table_name = f"table_comparison_{name}"
        if fig_name == "fig_comparison":
            fig_name = f"fig_comparison_{name}"

    tables = make_tables(collected, tabledir, methods=methods,
                         metrics=table_metrics, include_d=include_d, prec=prec,
                         table_name=table_name, caption=caption, label=label)
    make_figures(collected, figdir, methods=methods, metrics=fig_metrics_,
                 ylim=ylim, formats=formats, fig_name=fig_name)
    return tables