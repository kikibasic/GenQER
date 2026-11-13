"""Recompute the paper's statistics from the per-query files in results/per_query/.

  python scripts/reproduce_tables.py

Needs only numpy, pandas and scipy (no index, no model, no API). Each block recomputes one
table of results/tables/ and reports the largest deviation from the stored file:

  Table III   per-dataset differences, bootstrap intervals, permutation tests, macro-average
  Table IV    passage-count control: tests at matched counts, Holm correction
  Sec. V-B    all-strategy integration against the two selectors
  Sec. V-B    degradation rates and McNemar test; regret against the per-query oracle
  Sec. V-E    integration against the median / mean / best single strategy on seven datasets
  Held-out    TREC-NEWS and Robust04

The permutation tests use 10,000 sign flips with a fixed seed; their last digits depend on
the order of the per-query vectors, so each block keeps the order used for the paper.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as C


def read(name):
    return pd.read_csv(os.path.join(C.PER_QUERY, name), dtype={"qid": str})


def compare(name, got, key):
    """Largest absolute deviation between a recomputed table and the stored one."""
    stored = pd.read_csv(os.path.join(C.TABLES, name))
    a = got.set_index(key).sort_index()
    b = stored.set_index(key).sort_index()
    cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    dev = float(np.nanmax(np.abs(a[cols].values.astype(float) - b.loc[a.index, cols].values.astype(float))))
    print(f"  -> {name}: {len(a)} rows, {len(cols)} numeric columns, largest deviation {dev:.4f}")


def series():
    sv = read("strategy_validation9_perquery.csv").set_index(["dataset", "qid"])
    fp = read("fusion_pool_perquery.csv")
    f9 = read("fusion_all9_perquery.csv")
    pick = lambda df, col, val: df[df[col] == val].set_index(["dataset", "qid"])["ndcg10"]
    return sv, {
        "all8": pick(fp, "method", "ctxpool_all8"),
        "all6": pick(fp, "method", "ctxpool_all6"),
        "rrf": pick(f9, "method", "rrf_all9"),
        "combsum": pick(f9, "method", "combsum_all9"),
        "obs_selector": pick(read("review_postretrieval_perquery.csv"), "method", "obs_top2_rrf_qpp_max"),
        "pred_selector": read("router_selection_accuracy9_perquery.csv").set_index(["dataset", "qid"])["ndcg10"],
    }


def table3(sv, s):
    print("Table III: differences against strict by dataset")
    base = sv["strict"]
    methods = ["all8", "rrf", "combsum"]
    rows, parts = [], {m: [] for m in methods}
    scopes = [(ds, base[base.index.get_level_values(0) == ds].sort_index().index)
              for ds in C.MAIN_DATASETS] + [("POOLED", base.sort_index().index)]
    for scope, idx in scopes:
        row = {"scope": scope, "n": len(idx), "strict_ndcg10": round(base.loc[idx].mean(), 4)}
        for m in methods:
            g = s[m].reindex(idx)
            d = g.values - base.loc[idx].values
            lo, hi = C.boot_ci(d)
            row.update({f"{m}_ndcg10": round(g.mean(), 4), f"{m}_delta": round(d.mean(), 4),
                        f"{m}_ci_lo": round(lo, 4), f"{m}_ci_hi": round(hi, 4),
                        f"{m}_p": round(C.paired_perm_p(g.values, base.loc[idx].values), 4)})
            if scope != "POOLED":
                parts[m].append(d)
        rows.append(row)
    row = {"scope": "MACRO", "n": len(C.MAIN_DATASETS),
           "strict_ndcg10": round(np.mean([r["strict_ndcg10"] for r in rows[:5]]), 4)}
    for m in methods:
        deltas = [d.mean() for d in parts[m]]
        lo, hi = C.macro_boot_ci(parts[m])
        row.update({f"{m}_ndcg10": round(np.mean([r[f"{m}_ndcg10"] for r in rows[:5]]), 4),
                    f"{m}_delta": round(float(np.mean(deltas)), 4),
                    f"{m}_ci_lo": round(lo, 4), f"{m}_ci_hi": round(hi, 4),
                    f"{m}_p": round(C.signflip_p(deltas), 4)})   # exact test over five datasets
    rows.append(row)
    out = pd.DataFrame(rows)
    print(out[["scope", "n", "strict_ndcg10", "all8_delta", "all8_ci_lo", "all8_ci_hi", "all8_p",
               "rrf_delta", "rrf_p", "combsum_delta", "combsum_p"]].to_string(index=False))
    compare("review_per_dataset.csv", out, "scope")


def table4(sv, s):
    print("\nTable IV: passage-count control, tests at matched counts")
    cc = read("count_control_perquery.csv")
    w = cc.pivot_table(index=["dataset", "qid"], columns="cond", values="ndcg10").reindex(sv.index)
    w["pool_5_avg"] = w[["pool_5_0", "pool_5_1", "pool_5_2"]].mean(axis=1)
    w["ref_strict_5"] = sv["strict"]
    w["ref_all8_40"] = s["all8"].reindex(w.index)
    pairs = [("pool_5_avg", "ref_strict_5"), ("mix_2_5_0", "single_strict_10"),
             ("pool_10_0", "single_strict_10"), ("mix_4_5_0", "single_strict_20"),
             ("pool_20_0", "single_strict_20"), ("single_strict_10", "ref_strict_5"),
             ("single_strict_20", "ref_strict_5"), ("ref_all8_40", "single_strict_40"),
             ("single_strict_40", "ref_strict_5"), ("single_strict_40", "single_strict_20")]
    test = lambda g, cond, ref, scope: {
        "scope": scope, "cond": cond, "base": ref, "n": len(g),
        "ndcg10": round(g[cond].mean(), 4), "base_ndcg10": round(g[ref].mean(), 4),
        "diff": round(g[cond].mean() - g[ref].mean(), 4),
        "p": round(C.paired_perm_p(g[cond].values, g[ref].values), 4)}
    rows = [test(w, cond, ref, "POOLED") for cond, ref in pairs]
    for cond, ref in [("pool_20_0", "single_strict_20"), ("ref_all8_40", "single_strict_40")]:
        rows += [test(w.loc[ds], cond, ref, ds) for ds in C.MAIN_DATASETS]
    out = pd.DataFrame(rows)
    # Holm correction over the four matched-count tests of the eight-strategy pool
    family = out[(out.scope == "POOLED") & out.cond.isin(["pool_5_avg", "pool_10_0", "pool_20_0", "ref_all8_40"])
                 & (out.base != "ref_strict_5") | (out.scope == "POOLED") & (out.cond == "pool_5_avg")].index
    out["holm3_p"] = np.nan
    out.loc[family, "holm3_p"] = np.round(C.holm(out.loc[family, "p"].values), 4)
    print(out[out.scope == "POOLED"].to_string(index=False))
    out["key"] = out.scope + "|" + out.cond + "|" + out.base
    stored_name = "count_control_samecount.csv"
    stored = pd.read_csv(os.path.join(C.TABLES, stored_name))
    stored["key"] = stored.scope + "|" + stored.cond + "|" + stored.base
    a, b = out.set_index("key").sort_index(), stored.set_index("key").sort_index()
    cols = ["ndcg10", "base_ndcg10", "diff", "p", "holm3_p"]
    dev = float(np.nanmax(np.abs(a[cols].values - b.loc[a.index, cols].values)))
    print(f"  -> {stored_name}: {len(a)} rows, largest deviation {dev:.4f}")

    rates = {c: float((w[c] < sv["n0"]).mean()) for c in ("single_strict_20", "pool_20_0")}
    print(f"  degradation rate at 20 passages: strict {rates['single_strict_20']:.3f}, "
          f"eight-strategy pool {rates['pool_20_0']:.3f}")


def selection_vs_integration(sv, s):
    print("\nAll-strategy integration against the per-query selectors")
    ser = {"obs_selector": s["obs_selector"], "pred_selector": s["pred_selector"],
           "all8": s["all8"], "strict": sv["strict"]}
    pairs = [("all8", "obs_selector"), ("all8", "pred_selector"),
             ("obs_selector", "strict"), ("pred_selector", "strict"), ("all8", "strict")]
    rows = []
    for scope in C.MAIN_DATASETS + ["POOLED"]:
        idx = ser["strict"].sort_index().index
        if scope != "POOLED":
            idx = idx[idx.get_level_values(0) == scope]
        for a, b in pairs:
            x, y = ser[a].reindex(idx), ser[b].reindex(idx)
            d = x.values - y.values
            lo, hi = C.boot_ci(d)
            rows.append({"scope": scope, "n": len(idx), "method": a, "baseline": b,
                         "ndcg10_method": round(x.mean(), 4), "ndcg10_baseline": round(y.mean(), 4),
                         "delta": round(d.mean(), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "perm_p": round(C.paired_perm_p(x.values, y.values), 4),
                         "method_wins": int((d > 0).sum()), "baseline_wins": int((d < 0).sum()),
                         "ties": int((d == 0).sum())})
    out = pd.DataFrame(rows)
    print(out[out.scope == "POOLED"].drop(columns=["scope", "n"]).to_string(index=False))
    out["key"] = out.scope + "|" + out.method + "|" + out.baseline
    stored = pd.read_csv(os.path.join(C.TABLES, "review_selection_vs_integration.csv"))
    stored["key"] = stored.scope + "|" + stored.method + "|" + stored.baseline
    a, b = out.set_index("key").sort_index(), stored.set_index("key").sort_index()
    cols = ["ndcg10_method", "ndcg10_baseline", "delta", "ci_lo", "ci_hi", "perm_p",
            "method_wins", "baseline_wins", "ties"]
    dev = float(np.abs(a[cols].values.astype(float) - b.loc[a.index, cols].values.astype(float)).max())
    print(f"  -> review_selection_vs_integration.csv: {len(a)} rows, largest deviation {dev:.4f}")


def degradation_and_regret(sv, s):
    print("\nDegradation (nDCG@10 below no expansion) and regret against the per-query oracle")
    n0, strict, all8 = sv["n0"], sv["strict"], s["all8"].reindex(sv.index)
    ws, wa = strict < n0, all8 < n0
    b, c = int((ws & ~wa).sum()), int((~ws & wa).sum())
    print(f"  degraded: strict {ws.mean():.3f}, all-strategy integration {wa.mean():.3f}; "
          f"only strict {b}, only integration {c}; exact McNemar p = {C.mcnemar_exact(b, c):.2g}")
    print(f"  degraded by at least 0.05: strict {((n0 - strict) >= 0.05).mean():.3f}, "
          f"integration {((n0 - all8) >= 0.05).mean():.3f}")
    opts = sv[C.OPTIONS9]
    oracle = opts.max(axis=1)
    ties = (opts.values == oracle.values[:, None]).sum(axis=1)
    print(f"  oracle nDCG@10 {oracle.mean():.4f}; options tied at the maximum {ties.mean():.1f} on average; "
          f"accuracy of a uniformly random choice {np.mean(ties / 9):.3f}")
    for name, g in (("strict", strict), ("predictive selector", s["pred_selector"].reindex(sv.index)),
                    ("observational selector", s["obs_selector"].reindex(sv.index)),
                    ("all-strategy integration", all8)):
        print(f"  regret of {name}: {oracle.mean() - g.mean():.4f}")
    p6 = C.paired_perm_p(s["all6"].reindex(sv.index).values, strict.values)
    print(f"  six-strategy integration: {s['all6'].reindex(sv.index).mean() - strict.mean():+.4f} against strict "
          f"(uncorrected p = {p6:.4f}; Holm-corrected values are in holm_corrected_pooled.csv)")


def across_datasets():
    print("\nIntegration against the single strategies on seven datasets")
    df = read("corpus_strategy_perquery.csv")
    cols = [C.CSV_NAME[x] for x in C.STRATEGIES]
    rows = []
    for ds in C.MAIN_DATASETS + C.HELDOUT_DATASETS:
        g = df[df.dataset == ds]
        means, all8 = g[cols].mean(), g["all8"].mean()
        rows.append({"dataset": ds, "n": len(g), "all8": round(all8, 4),
                     "best_single": means.idxmax(), "best_single_ndcg10": round(means.max(), 4),
                     "median_single": round(float(np.median(means.values)), 4),
                     "mean_single": round(float(means.mean()), 4),
                     "sd_single": round(float(means.std(ddof=0)), 4),
                     "delta_vs_best": round(all8 - means.max(), 4),
                     "delta_vs_median": round(all8 - float(np.median(means.values)), 4),
                     "delta_vs_mean": round(all8 - float(means.mean()), 4),
                     "n_strategies_beaten": int((all8 > means).sum())})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    compare("review_median_comparison.csv", out, "dataset")
    for col in ("delta_vs_best", "delta_vs_median", "delta_vs_mean"):
        r, p = stats.pearsonr(out["sd_single"], out[col])
        print(f"  correlation of {col} with the SD of the eight strategy means: r = {r:.3f}, p = {p:.3f}")


def heldout():
    print("\nHeld-out datasets")
    ho = read("heldout_confirmation_perquery.csv")
    for scope in C.HELDOUT_DATASETS + ["POOLED"]:
        g = ho if scope == "POOLED" else ho[ho.dataset == scope]
        p = C.paired_perm_p(g["all8"].values, g["strict"].values)
        print(f"  {scope:9s} n = {len(g):3d}: strict {g.strict.mean():.4f}, integration {g.all8.mean():.4f}, "
              f"difference {g.all8.mean() - g.strict.mean():+.4f}, p = {p:.4f}")


def main():
    sv, s = series()
    table3(sv, s)
    table4(sv, s)
    selection_vs_integration(sv, s)
    degradation_and_regret(sv, s)
    across_datasets()
    heldout()


if __name__ == "__main__":
    main()
