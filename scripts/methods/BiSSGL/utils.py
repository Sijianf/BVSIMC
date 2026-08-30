"""Analysis utilities for masked-CV simulation results.

Load, summarize (nested CV), tabulate (LaTeX), and plot method-comparison
results. Import into a notebook or script; nothing runs at import time.

Typical usage
-------------
    import utils as u

    collected = u.collect(PATH_OUTPUT, methods=["bissgl", "sgimc", "imc"],
                          val_metric="auc", select_per="fold",
                          feature_range=(50, 400), drop_auc_half=True)
    u.overview(collected)
    u.fig_metrics(collected, metrics=("aupr", "auc", "f1"), ylim=(0, 1.02))
    u.make_report(collected, savedir="report")
"""

import os
import glob
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# ====== score vector column names (matches get_metrics return order) ======
SCORE_COLS = ["aupr", "auc", "f1", "accuracy", "recall", "specificity", "precision"]

# ====== selection-metric names (keys inside test_selacc_A / test_selacc_B) ======
SELACC_COLS = ["recall", "precision", "f1", "accuracy", "specificity"]

# ====== hyperparam columns per method (masked / pairwise schema) ======
HYPERPARAM_COLS = {
    "bissgl": ["xi", "eta", "tilde_lambda0", "tilde_lambda1", "lambda0", "lambda1", "K"],
    "sgimc":  ["C_lasso", "C_group", "C_ridge", "rank"],
    "imc":    ["C_lasso", "C_group", "C_ridge", "lamb", "rank"],
    "drimc":  ["lamU", "lamV", "cc", "iterpara", "numLat"],
    "nrlmf":  ["c", "K1", "K2", "r", "lambda_d", "lambda_t",
               "alpha", "beta", "theta", "max_iter"],
}


# ====== load results — single / task / local files ======
def load_results(output_dir, method_name):
    """Load masked-CV results for a method, concatenating any of:
        results_<method>.gz
        results_<method>_task*.gz      (HPC array)
        results_<method>_local_*.gz    (local notebooks)
    Returns one DataFrame; val_score / test_score become numpy arrays.
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
    return pd.DataFrame(
        np.stack(df[col].values), index=df.index,
        columns=[f"{col}_{n}" for n in names],
    )


# ====== expand a column of selection-metric dicts into named columns ======
def expand_selacc(df, col, prefix):
    if col not in df.columns:
        return pd.DataFrame(index=df.index)
    data = {}
    for m in SELACC_COLS:
        data[f"{prefix}_{m}"] = df[col].apply(
            lambda d: d.get(m, np.nan) if isinstance(d, dict) else np.nan
        )
    return pd.DataFrame(data, index=df.index)


# ====== main summarization (nested CV) ======
def summarize_masked(df, hyperparam_cols, val_metric="aupr", select_per="fold"):
    """Nested-CV summary for the masked / pairwise results.

    Pipeline
    --------
    1. average inner-CV val_score per (feature, repeat, fold, hyperparams)
    2. select best hyperparams per outer fold (select_per='fold', nested CV) or
       once per dataset (select_per='dataset', pooling inner CV across folds)
    3. take the outer test_score for the chosen hyperparams (one per fold)
    4. average over outer folds -> one score per (feature, repeat)
    5. mean & std over repeats -> per (feature)

    Returns dict: val_avg, best_params, test_per_fold, test_per_rep, final.
    """
    id_cols   = ["feature_id", "n_features", "repeat_id"]
    fold_col  = ["fold"]
    group     = id_cols + fold_col + hyperparam_cols
    val_metric_col = f"val_score_{val_metric}"

    # --- Step 1: average inner-CV val_score per (feature, repeat, fold, HP) ---
    val_avg = df.groupby(group, as_index=False).agg(
        val_score=("val_score", lambda x: np.mean(np.stack(x), axis=0)),
        val_d1=("val_d1", "mean"),
        val_d2=("val_d2", "mean"),
    )
    val_avg = pd.concat(
        [val_avg.drop(columns=["val_score"]), expand_scores(val_avg, "val_score")],
        axis=1,
    )

    # --- Step 2: select best hyperparams ---
    if select_per == "fold":
        sel_cols = id_cols + fold_col
        best_idx = val_avg.groupby(sel_cols)[val_metric_col].idxmax()
        best_params = val_avg.loc[
            best_idx, sel_cols + hyperparam_cols + [val_metric_col]
        ].reset_index(drop=True)
        merge_cols = sel_cols + hyperparam_cols
    elif select_per == "dataset":
        # average val across folds too, pick one HP per (feature, repeat)
        sel_cols = id_cols
        across_fold = (
            val_avg.groupby(id_cols + hyperparam_cols, as_index=False)[val_metric_col]
            .mean()
        )
        best_idx = across_fold.groupby(sel_cols)[val_metric_col].idxmax()
        best_params = across_fold.loc[
            best_idx, sel_cols + hyperparam_cols + [val_metric_col]
        ].reset_index(drop=True)
        merge_cols = sel_cols + hyperparam_cols
    else:
        raise ValueError("select_per must be 'fold' or 'dataset'")

    # --- Step 3: outer test_score for the chosen HP (one row per id+fold) ---
    df_best = df.merge(best_params[merge_cols], on=merge_cols, how="inner")
    dedup_cols = id_cols + fold_col
    keep = (
        [c for c in df_best.columns
         if c in dedup_cols + hyperparam_cols
         or c.startswith("test_")]
    )
    test_per_fold = (
        df_best.groupby(dedup_cols, as_index=False).first()[keep]
    )
    test_per_fold = pd.concat(
        [
            test_per_fold.drop(columns=["test_score"]),
            expand_scores(test_per_fold, "test_score"),
            expand_selacc(test_per_fold, "test_selacc_A", "selA"),
            expand_selacc(test_per_fold, "test_selacc_B", "selB"),
        ],
        axis=1,
    )

    test_score_cols = [f"test_score_{n}" for n in SCORE_COLS]
    sel_cols_present = [
        c for c in test_per_fold.columns
        if c.startswith("selA_") or c.startswith("selB_")
    ]
    metric_cols = test_score_cols + ["test_d1", "test_d2"] + sel_cols_present

    # --- Step 4: average over outer folds -> per (feature, repeat) ---
    test_per_rep = (
        test_per_fold.groupby(id_cols, as_index=False)[metric_cols].mean()
    )

    # --- Step 5: mean & std over repeats -> per (feature) ---
    agg = {f"{c}_mean": (c, "mean") for c in test_score_cols}
    agg.update({f"{c}_std": (c, lambda x: x.std(ddof=1)) for c in test_score_cols})
    agg.update({f"{c}_mean": (c, "mean") for c in (["test_d1", "test_d2"] + sel_cols_present)})
    agg["n_repeats"] = ("repeat_id", "count")

    final = (
        test_per_rep.groupby(["feature_id", "n_features"], as_index=False)
        .agg(**agg)
        .sort_values("n_features")
        .reset_index(drop=True)
    )

    return {
        "val_avg":       val_avg,
        "best_params":   best_params,
        "test_per_fold": test_per_fold,
        "test_per_rep":  test_per_rep,
        "final":         final,
    }


# ====== convenience wrapper per method ======
def summarize_method(output_dir, method_name, val_metric="aupr",
                     select_per="fold", filters=None,
                     feature_range=None, drop_auc_half=False):
    """Load + summarize a method in one call.

    Example
    -------
    res = summarize_method(
        output_dir  = ".../outputs/results/simulations/masked_ones",
        method_name = "bissgl",
        val_metric  = "aupr",
    )
    print(res["final"])
    """
    hyperparam_cols = HYPERPARAM_COLS[method_name]
    df = load_results(output_dir, method_name)

    # keep only feature sizes in range (e.g. drop 450+ where a method didn't finish)
    if feature_range is not None:
        lo, hi = feature_range
        df = df[(df["n_features"] >= lo) & (df["n_features"] <= hi)]

    # drop non-working runs: test AUC == 0.5 (degenerate / constant predictions)
    if drop_auc_half:
        test_auc = df["test_score"].apply(lambda v: float(np.asarray(v)[1]))
        df = df[~np.isclose(test_auc, 0.5)]

    df = df.reset_index(drop=True)

    if filters is not None:
        for col, val in filters.items():
            df = df[df[col] == val]
        df = df.reset_index(drop=True)

    return summarize_masked(df, hyperparam_cols, val_metric=val_metric,
                            select_per=select_per)

# ====== display config ======
METHOD_LABEL = {"bissgl": "BVSIMC", "sgimc": "SGIMC", "imc": "IMC",
                "drimc": "DRIMC", "nrlmf": "NRLMF"}
METHOD_ORDER = ["bissgl", "sgimc", "imc", "drimc", "nrlmf"]
SELECTION_METHODS = ["bissgl", "sgimc"]   # only these do variable selection

METRIC_LABEL = {"aupr": "AUPR", "auc": "AUC", "f1": "F1", "accuracy": "Accuracy",
                "recall": "Recall", "specificity": "Specificity",
                "precision": "Precision"}

# LaTeX-friendly names for hyperparameters used in variant labels
_LATEX_PARAM = {"xi": r"\xi", "tilde_lambda0": r"\tilde{\lambda}_0",
                "lambda0": r"\lambda_0", "K": "K", "cc": "c", "c": "c",
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


# ====== load + summarize every method present ======
def collect(output_dir, methods=METHOD_ORDER, val_metric="aupr", select_per="fold",
            feature_range=None, drop_auc_half=False, filters=None):
    """Return {method: summarize_masked result dict} for every method whose
    result files are found in output_dir.

    filters : optional per-method row filters applied before summarization.
        Either a single dict applied to every method, e.g. {"K": 25}, or a
        nested {method: dict} to filter methods individually, e.g.
        {"bissgl": {"xi": 1}, "sgimc": {"C_ridge": 1.0}}. A method absent from
        a nested mapping is summarized unfiltered.
    """
    # normalize: distinguish a flat dict (apply to all) from {method: dict}
    per_method = bool(filters) and all(
        (m in filters) and isinstance(filters[m], dict) for m in filters
    ) and any(m in methods for m in filters)

    out = {}
    for m in methods:
        if not filters:
            f = None
        elif per_method:
            f = filters.get(m)
        else:
            f = filters                      # same flat dict for every method
        try:
            out[m] = summarize_method(output_dir, m, val_metric=val_metric,
                                      select_per=select_per, filters=f,
                                      feature_range=feature_range,
                                      drop_auc_half=drop_auc_half)
        except (FileNotFoundError, KeyError, ValueError):
            # missing files, unknown method, or a filter that removed all rows
            pass
    return out


def add_variant(collected, output_dir, method, filters, key=None, label=None,
                style=None, val_metric="aupr", select_per="fold",
                feature_range=None, drop_auc_half=False):
    """Add a hyperparameter-restricted variant of `method` as its own curve.

    Example: a second BVSIMC curve using only xi=1 rows --
        add_variant(collected, out_dir, "bissgl", {"xi": 1})   # key "bissgl_xi1"
    Then include the key in `methods=[...]` for any table/figure. By default the
    variant inherits the base method's colour/marker and is drawn dashed.
    """
    key = key or (method + "_" + "_".join(f"{k}{v}" for k, v in filters.items()))
    summ = summarize_method(output_dir, method, val_metric=val_metric,
                            select_per=select_per, filters=filters,
                            feature_range=feature_range, drop_auc_half=drop_auc_half)
    summ["_base"] = method
    summ["_filters"] = dict(filters)
    summ["_label"] = label
    summ["_style"] = style
    collected[key] = summ
    return key


def _with_sel_avg(tpr):
    """Add sel_<metric> = mean over the A and B feature blocks, when present."""
    df = tpr.copy()
    for mt in SELACC_COLS:
        a, b = f"selA_{mt}", f"selB_{mt}"
        if a in df.columns and b in df.columns:
            df[f"sel_{mt}"] = df[[a, b]].mean(axis=1)
    return df


def feature_stats(collected, method, col):
    """mean / std / n of `col` per feature size, from test_per_rep."""
    tpr = _with_sel_avg(collected[method]["test_per_rep"])
    g = tpr.groupby(["feature_id", "n_features"])[col]
    out = (g.agg(mean="mean", std=lambda x: x.std(ddof=1), n="count")
             .reset_index().sort_values("n_features").reset_index(drop=True))
    return out


def _label_for(key, collected):
    """Display/LaTeX label for a curve key (base method or add_variant key)."""
    summ = collected.get(key, {})
    if summ.get("_label"):
        return summ["_label"]
    return get_method_label(summ.get("_base", key), summ.get("_filters"))


def _style_for(key, collected):
    """Line style for a curve: base method colour+marker, solid for the base
    method and dashed for a filtered variant (overridable via add_variant)."""
    summ = collected.get(key, {})
    if summ.get("_style"):
        return summ["_style"]
    base = summ.get("_base", key)
    st = STYLE.get(base, dict(color="#7f7f7f", marker="o"))
    ls = "--" if summ.get("_filters") else "-"
    return dict(color=st["color"], marker=st["marker"], linestyle=ls)


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


# ====== test-metric table: rows = n_features, cols = methods ======
def test_metric_table(collected, metric="aupr", methods=None, prec=3,
                      bold_best=True, show_selection=False, sel_prec=1,
                      caption=None, label=None):
    """Table of a test metric, rows = n_features, cols = methods.

    show_selection : if True, append the number of selected features for each
        variable-selection method (SELECTION_METHODS present in `collected`)
        as extra "<label> $d_1$" / "$d_2$" columns, formatted mean (sd) over
        repeats. d1 = row/drug-side, d2 = column/target-side. Methods that do
        not select variables contribute no selection columns.
    """
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    col = f"test_score_{metric}"

    stats = {m: feature_stats(collected, m, col) for m in methods}
    feats = stats[methods[0]]["n_features"].values

    # selection-count stats for the selecting methods (only if requested)
    sel_methods = ([m for m in methods
                    if collected.get(m, {}).get("_base", m) in SELECTION_METHODS
                    and "test_d1" in collected[m]["test_per_rep"].columns]
                   if show_selection else [])
    d_stats = {(m, d): feature_stats(collected, m, d)
               for m in sel_methods for d in ("test_d1", "test_d2")}

    # display DataFrame (plain strings) + bold-best per row for LaTeX
    disp = pd.DataFrame(index=feats)
    disp.index.name = "n\\_features"
    body = []
    for i, nf in enumerate(feats):
        row_means = {m: stats[m]["mean"].values[i] for m in methods}
        best = max(row_means, key=row_means.get) if bold_best else None
        cells = []
        for m in methods:
            txt = _fmt(stats[m]["mean"].values[i], stats[m]["std"].values[i], prec)
            disp.loc[nf, _label_for(m, collected)] = txt
            cells.append(f"\\textbf{{{txt}}}" if m == best else txt)
        # selection-count cells (not bolded — different quantity, not "better")
        for m in sel_methods:
            for d, sym in (("test_d1", "d_1"), ("test_d2", "d_2")):
                s = d_stats[(m, d)]
                txt = _fmt(s["mean"].values[i], s["std"].values[i], sel_prec)
                disp.loc[nf, f"{_label_for(m, collected)} ${sym}$"] = txt
                cells.append(txt)
        body.append(" & ".join([str(int(nf))] + cells) + " \\\\")

    # headers
    metric_hdr = [_label_for(m, collected) for m in methods]
    sel_hdr = [f"{_label_for(m, collected)} ${sym}$"
               for m in sel_methods for sym in ("d_1", "d_2")]
    header = " & ".join(["\\#features"] + metric_hdr + sel_hdr) + " \\\\"
    colfmt = "l" + "c" * (len(methods) + len(sel_hdr))
    sel_note = (" $d_1$/$d_2$ = number of selected row-/column-side features."
                if sel_methods else "")
    caption = caption or (f"Test {METRIC_LABEL.get(metric, metric)} "
                          f"(mean (sd) over 10 repeats); best per row in bold.{sel_note}")
    latex = _booktabs([header], body, colfmt, caption, label or f"tab:{metric}")
    return disp, latex


# ====== selection table: BVSIMC & SGIMC, grouped Recall/Precision/F1 ======
def selection_table(collected, methods=SELECTION_METHODS,
                    metrics=("recall", "precision", "f1"), prec=3,
                    caption=None, label=None):
    methods = [m for m in methods if m in collected
               and any(c.startswith("selA_") for c in
                       collected[m]["test_per_rep"].columns)]
    if not methods:
        return None, None

    feats = feature_stats(collected, methods[0], f"sel_{metrics[0]}")["n_features"].values
    stats = {(m, mt): feature_stats(collected, m, f"sel_{mt}")
             for m in methods for mt in metrics}

    body = []
    for i, nf in enumerate(feats):
        cells = [str(int(nf))]
        for m in methods:
            for mt in metrics:
                s = stats[(m, mt)]
                cells.append(_fmt(s["mean"].values[i], s["std"].values[i], prec))
        body.append(" & ".join(cells) + " \\\\")

    # two-row header with method groups
    g = len(metrics)
    top = ["\\#features"]
    cmid = []
    start = 2
    for m in methods:
        top.append(f"\\multicolumn{{{g}}}{{c}}{{{_label_for(m, collected)}}}")
        cmid.append(f"\\cmidrule(lr){{{start}-{start + g - 1}}}")
        start += g
    header1 = " & ".join(top) + " \\\\"
    header2 = " & ".join([""] + [METRIC_LABEL[mt].replace("F1", "F1")
                                 for m in methods for mt in metrics]) + " \\\\"
    header = [header1, " ".join(cmid), header2]
    colfmt = "l" + "c" * (g * len(methods))
    caption = caption or ("Variable-selection performance vs. the 25 informative "
                          "features (mean (sd) over 10 repeats; averaged over "
                          "the row- and column-feature blocks).")
    latex = _booktabs(header, body, colfmt, caption, label or "tab:selection")

    # plain display frame
    disp = pd.DataFrame(index=feats)
    disp.index.name = "n_features"
    for m in methods:
        for mt in metrics:
            s = stats[(m, mt)]
            disp[f"{_label_for(m, collected)}-{METRIC_LABEL[mt]}"] = [
                _fmt(s["mean"].values[i], s["std"].values[i], prec)
                for i in range(len(feats))
            ]
    return disp, latex


# ====== shared panel styling / drawing ======
def _style_axis(ax):
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 4))
        ax.spines[side].set_color("#c7cbd1")
    ax.grid(axis="y", linestyle="-", linewidth=0.9, color=_GRID)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5, width=1.0, colors=_MUTED, labelcolor=_INK, labelsize=10)


def _panels(collected, methods, metrics, col_fn, ylabel_fn, ylim=None,
            figsize=None, savepath=None):
    metrics = list(metrics)
    figsize = figsize or (4.9 * len(metrics), 4.3)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
        for j, mt in enumerate(metrics):
            ax = axes[0, j]
            for m in methods:
                st = feature_stats(collected, m, col_fn(mt))
                x = st["n_features"].values
                y = st["mean"].values
                e = st["std"].values
                style = _style_for(m, collected)
                c = style["color"]
                ax.fill_between(x, y - e, y + e, color=c, alpha=0.12,
                                linewidth=0, zorder=1)
                ax.plot(x, y, label=_label_for(m, collected), color=c,
                        marker=style["marker"], linestyle=style["linestyle"],
                        linewidth=2.2, markersize=3.5, 
                        # markerfacecolor="white",
                        markeredgecolor=c, markeredgewidth=1.8,
                        solid_capstyle="round", clip_on=False, zorder=3)
            _style_axis(ax)
            ax.set_xlabel("Number of features", fontsize=12, labelpad=7)
            ax.set_ylabel(ylabel_fn(mt), fontsize=12, labelpad=7)
            ax.set_title(chr(65 + j), loc="left", fontsize=13,
                         fontweight="bold", pad=8, color=_INK)
            ax.margins(x=0.03)
            if ylim is not None:
                ax.set_ylim(*ylim)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=len(methods),
                   frameon=False, handlelength=1.7, columnspacing=1.9, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
        return fig


# ====== figure: metric(s) vs n_features ======
def fig_metrics(collected, metrics=("aupr", "auc", "f1"), methods=None,
                ylim=None, figsize=None, savepath=None):
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    return _panels(collected, methods, metrics,
                   col_fn=lambda mt: f"test_score_{mt}",
                   ylabel_fn=lambda mt: METRIC_LABEL.get(mt, mt),
                   ylim=ylim, figsize=figsize, savepath=savepath)


# ====== single-panel: one metric vs n_features (old plot_vs_n_features analog) ==
def plot_vs_n_features(collected, metric="aupr", methods=None, ylim=None,
                       legend_loc="best", figsize=(6.2, 4.6), savepath=None):
    """One metric vs number of features on a single axes. `methods` may include
    add_variant() keys (e.g. 'bissgl_xi1') to overlay a dashed variant curve."""
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=figsize)
        for m in methods:
            st = feature_stats(collected, m, f"test_score_{metric}")
            x, y, e = st["n_features"].values, st["mean"].values, st["std"].values
            style = _style_for(m, collected)
            c = style["color"]
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.12, linewidth=0, zorder=1)
            ax.plot(x, y, label=_label_for(m, collected), color=c,
                    marker=style["marker"], linestyle=style["linestyle"],
                    linewidth=2.2, markersize=3.5, markerfacecolor="white",
                    markeredgecolor=c, markeredgewidth=1.8,
                    solid_capstyle="round", clip_on=False, zorder=3)
        _style_axis(ax)
        ax.set_xlabel("Number of features", fontsize=12, labelpad=7)
        ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=12, labelpad=7)
        ax.margins(x=0.03)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc=legend_loc, fontsize=10, frameon=False, handlelength=2.2)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
        return fig


# ====== figure: selection metric(s) vs n_features (BVSIMC & SGIMC) ======
def fig_selection(collected, metrics=("recall", "precision", "f1"),
                  methods=SELECTION_METHODS, ylim=(-0.02, 1.03),
                  figsize=None, savepath=None):
    methods = [m for m in methods if m in collected
               and any(c.startswith("selA_") for c in
                       collected[m]["test_per_rep"].columns)]
    if not methods:
        return None
    return _panels(collected, methods, metrics,
                   col_fn=lambda mt: f"sel_{mt}",
                   ylabel_fn=lambda mt: f"Selection {METRIC_LABEL.get(mt, mt)}",
                   ylim=ylim, figsize=figsize, savepath=savepath)


# ====== interactive analysis helpers ======
def overview(collected, methods=None):
    """One-row-per-method sanity table: feature sizes, dataset count, repeats,
    and whether selection metrics are available."""
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    rows = []
    for m in methods:
        tpr = collected[m]["test_per_rep"]
        rows.append({
            "method": _label_for(m, collected),
            "key": m,
            "feature_sizes": tpr["n_features"].nunique(),
            "datasets": len(tpr),
            "repeats/size": int(tpr.groupby("n_features")["repeat_id"].nunique().median()),
            "selection": any(c.startswith("selA_") for c in tpr.columns),
        })
    return pd.DataFrame(rows)


def combined_per_rep(collected, methods=None):
    """All methods' per-(feature, repeat) scores stacked into one tidy frame,
    with a `method` column — the handle for custom analysis / plotting / tests."""
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    frames = []
    for m in methods:
        df = _with_sel_avg(collected[m]["test_per_rep"]).copy()
        df.insert(0, "method", _label_for(m, collected))
        df.insert(1, "method_key", m)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pairwise_wilcoxon(collected, reference="bissgl", metric="aupr", methods=None):
    """Paired Wilcoxon signed-rank of `reference` vs each other method, per
    feature size (paired across repeats, since all methods share the same
    datasets/folds). Returns a long DataFrame with median difference and p-value.
    Note: only ~10 repeats per feature size, so treat p-values as indicative."""
    from scipy.stats import wilcoxon
    methods = methods or [m for m in METHOD_ORDER if m in collected]
    if reference not in collected:
        raise KeyError(f"reference '{reference}' not in collected results")
    col = f"test_score_{metric}"
    keys = ["feature_id", "n_features", "repeat_id"]
    ref = collected[reference]["test_per_rep"][keys + [col]]
    rows = []
    for m in methods:
        if m == reference:
            continue
        oth = collected[m]["test_per_rep"][keys + [col]]
        mg = ref.merge(oth, on=keys, suffixes=("_ref", "_oth"))
        for nf, sub in mg.groupby("n_features"):
            a, b = sub[f"{col}_ref"].values, sub[f"{col}_oth"].values
            try:
                _, p = wilcoxon(a, b)
            except ValueError:      # zero differences etc.
                p = np.nan
            rows.append({"n_features": nf, "vs": _label_for(m, collected),
                         "median_diff": float(np.median(a - b)),
                         "ref_wins_frac": float(np.mean(a > b)),
                         "p_value": p})
    return (pd.DataFrame(rows)
            .sort_values(["vs", "n_features"]).reset_index(drop=True))


# ====== report helpers: tables and figures, separately or together ======
def _sel_methods_present(collected, methods):
    return [m for m in SELECTION_METHODS
            if (methods is None or m in methods) and m in collected]


def make_tables(collected, savedir, methods=None,
                table_metrics=("aupr", "auc", "f1"),
                sel_metrics=("recall", "precision", "f1"),
                prec=3, verbose=True):
    """Write all LaTeX tables (.tex) to `savedir`.

    Returns {name: (display_df, latex_str)} so the same tables can also be
    inspected inline in a notebook, e.g. display(tables["aupr"][0]).
    """
    os.makedirs(savedir, exist_ok=True)
    tables = {}

    for metric in table_metrics:
        disp, latex = test_metric_table(collected, metric=metric,
                                        methods=methods, prec=prec)
        tables[metric] = (disp, latex)
        with open(os.path.join(savedir, f"table_{metric}.tex"), "w") as f:
            f.write(latex + "\n")

    sel_methods = _sel_methods_present(collected, methods)
    if sel_methods:
        disp_sel, latex_sel = selection_table(collected, methods=sel_methods,
                                              metrics=sel_metrics, prec=prec)
        if latex_sel:
            tables["selection"] = (disp_sel, latex_sel)
            with open(os.path.join(savedir, "table_selection.tex"), "w") as f:
                f.write(latex_sel + "\n")

    if verbose:
        print(f"Tables ({', '.join(tables)}) written to {savedir}/")
    return tables


def make_figures(collected, savedir, methods=None,
                 metrics=("aupr", "auc", "f1"),
                 sel_metrics=("recall", "precision", "f1"),
                 ylim=None, sel_ylim=(-0.02, 1.03),
                 formats=("pdf", "png"), verbose=True):
    """Write the metric and selection figures to `savedir` in each format.

    `ylim` applies to the test-metric panels (e.g. (0, 1.02));
    `sel_ylim` to the selection panels.
    """
    os.makedirs(savedir, exist_ok=True)
    sel_methods = _sel_methods_present(collected, methods)

    for ext in formats:
        fig_metrics(collected, metrics=metrics, methods=methods, ylim=ylim,
                    savepath=os.path.join(savedir, f"fig_metrics.{ext}"))
        if sel_methods:
            fig_selection(collected, metrics=sel_metrics, methods=sel_methods,
                          ylim=sel_ylim,
                          savepath=os.path.join(savedir, f"fig_selection.{ext}"))
    plt.close("all")
    if verbose:
        print(f"Figures written to {savedir}/ ({', '.join(formats)})")


def make_report(collected, savedir=None, figdir=None, tabledir=None,
                methods=None,
                table_metrics=("aupr", "auc", "f1"),
                fig_metrics_=("aupr", "auc", "f1"),
                sel_metrics=("recall", "precision", "f1"),
                ylim=None, sel_ylim=(-0.02, 1.03),
                formats=("pdf", "png"), prec=3):
    """Export all tables and figures in one call.

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
                         table_metrics=table_metrics,
                         sel_metrics=sel_metrics, prec=prec)
    make_figures(collected, figdir, methods=methods,
                 metrics=fig_metrics_, sel_metrics=sel_metrics,
                 ylim=ylim, sel_ylim=sel_ylim, formats=formats)
    return tables