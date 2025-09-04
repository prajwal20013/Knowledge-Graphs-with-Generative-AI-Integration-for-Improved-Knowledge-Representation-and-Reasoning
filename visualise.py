from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

#  Utilities 

def savefig(name: str):
    out = FIGDIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

def load_csv(path_or_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_or_url)
    except Exception:
        df = pd.read_csv(path_or_url, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_numeric(df: pd.DataFrame, num_cols: Optional[List[str]]) -> List[str]:
    if num_cols:
        cols = [c for c in num_cols if c in df.columns]
        if cols:
            return cols
    # auto-pick numeric columns
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def pick_categorical(df: pd.DataFrame, cat_cols: Optional[List[str]]) -> List[str]:
    if cat_cols:
        cols = [c for c in cat_cols if c in df.columns]
        if cols:
            return cols
    # auto-pick low-cardinality non-numeric columns
    cats = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) > 1 and df[c].nunique(dropna=True) <= 12:
                cats.append(c)
    return cats[:2]

#  Demo data 

def make_demo_data(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # Heart-like
    age = rng.integers(29, 79, size=n)
    chol = rng.normal(220, 45, size=n)
    st = rng.normal(1.2, 0.7, size=n).clip(0, None)
    bp = rng.normal(130, 18, size=n)
    sex = rng.choice(["male", "female"], size=n, p=[0.6, 0.4])
    diagnosis = rng.choice(["no_disease", "angina", "ischemia"], size=n, p=[0.5, 0.3, 0.2])
    # scores for RAG vs GraphRAG illustration
    rag = rng.normal(0.58, 0.05, size=5).clip(0, 1)
    grag = (rag + rng.normal(0.06, 0.02, size=5)).clip(0, 1)
    k_values = np.array([3, 5, 10, 15, 20])
    # dataset label for aggregation later
    df = pd.DataFrame({
        "age": age,
        "cholesterol": chol,
        "st_depression": st,
        "blood_pressure": bp,
        "sex": sex,
        "diagnosis": diagnosis,
    })
    # attach a small metrics frame as attributes for plotting (not saved)
    df._meta = {
        "k_values": k_values,
        "rag_scores": rag,
        "grag_scores": grag
    }
    return df

#  Individual plots 

def plot_histogram(df: pd.DataFrame, col: str, title: Optional[str] = None, fname: Optional[str] = None):
    plt.figure()
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        print(f"[WARN] histogram skipped: {col} is not numeric or empty.")
        plt.close()
        return
    plt.hist(s, bins=30)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(title or f"Distribution of {col}")
    savefig(fname or f"hist_{col}.png")

def plot_bar(df: pd.DataFrame, col: str, by: Optional[str] = None, title: Optional[str] = None, fname: Optional[str] = None):
    plt.figure()
    if by and by in df.columns:
        # grouped mean of numeric by category
        gc = df.groupby(by)[col].mean(numeric_only=True).dropna()
        gc.plot(kind="bar")
        plt.ylabel(f"Mean {col}")
        plt.xlabel(by)
        plt.title(title or f"{col} by {by} (mean)")
        savefig(fname or f"bar_{col}_by_{by}.png")
    else:
        # simple category count
        vc = df[col].astype(str).value_counts().head(12)
        vc.plot(kind="bar")
        plt.ylabel("Count")
        plt.title(title or f"Bar of {col}")
        savefig(fname or f"bar_{col}.png")

def plot_grouped_bar(labels: List[str], rag: List[float], grag: List[float], ylabel="Score", title=None, fname="grouped_bar_rag_grag.png"):
    idx = np.arange(len(labels))
    w = 0.35
    plt.figure()
    plt.bar(idx - w/2, rag, w, label="RAG")
    plt.bar(idx + w/2, grag, w, label="GraphRAG")
    plt.xticks(idx, labels)
    plt.ylabel(ylabel)
    plt.title(title or "RAG vs GraphRAG (grouped)")
    plt.legend()
    savefig(fname)

def plot_stacked_bar(categories: List[str], comp_a: List[float], comp_b: List[float], title=None, fname="stacked_bar.png"):
    idx = np.arange(len(categories))
    comp_a = np.array(comp_a)
    comp_b = np.array(comp_b)
    plt.figure()
    plt.bar(idx, comp_a, label="Component A")
    plt.bar(idx, comp_b, bottom=comp_a, label="Component B")
    plt.xticks(idx, categories)
    plt.ylabel("Value")
    plt.title(title or "Stacked Bar")
    plt.legend()
    savefig(fname)

def plot_line(x: List[float], y: List[float], label: Optional[str] = None, title: Optional[str] = None, fname: str = "line.png"):
    plt.figure()
    plt.plot(x, y, marker="o", label=label if label else None)
    plt.xlabel("x")
    plt.ylabel("y")
    if label:
        plt.legend()
    plt.title(title or "Line plot")
    savefig(fname)

def plot_area(x: List[float], ys: List[List[float]], labels: List[str], title: Optional[str] = None, fname: str = "area.png"):
    plt.figure()
    ys_arr = np.array(ys)
    plt.stackplot(x, ys_arr, labels=labels)
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title(title or "Area (stacked)")
    plt.legend(loc="upper left")
    savefig(fname)

def plot_scatter(df: pd.DataFrame, x: str, y: str, title: Optional[str] = None, fname: Optional[str] = None):
    plt.figure()
    xvals = pd.to_numeric(df[x], errors="coerce")
    yvals = pd.to_numeric(df[y], errors="coerce")
    mask = xvals.notna() & yvals.notna()
    if not mask.any():
        print(f"[WARN] scatter skipped: {x} vs {y} not numeric/available.")
        plt.close()
        return
    plt.scatter(xvals[mask], yvals[mask])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{y} vs {x}")
    savefig(fname or f"scatter_{x}_vs_{y}.png")

def plot_box(df: pd.DataFrame, col: str, by: Optional[str] = None, title: Optional[str] = None, fname: Optional[str] = None):
    plt.figure()
    if by and by in df.columns and df[by].nunique() > 1:
        # create grouped data
        groups = []
        labels = []
        for level, sub in df[[col, by]].dropna().groupby(by):
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if not s.empty:
                groups.append(s.values)
                labels.append(str(level))
        if groups:
            plt.boxplot(groups, labels=labels)
            plt.xlabel(by)
            plt.ylabel(col)
            plt.title(title or f"{col} by {by}")
            savefig(fname or f"box_{col}_by_{by}.png")
            return
    # single series box
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        print(f"[WARN] boxplot skipped: {col} not numeric/available.")
        plt.close()
        return
    plt.boxplot(s.values, labels=[col])
    plt.title(title or f"Boxplot of {col}")
    savefig(fname or f"box_{col}.png")

def plot_violin(df: pd.DataFrame, col: str, by: Optional[str] = None, title: Optional[str] = None, fname: Optional[str] = None):
    plt.figure()
    if by and by in df.columns and df[by].nunique() > 1:
        data = []
        ticks = []
        for level, sub in df[[col, by]].dropna().groupby(by):
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if not s.empty:
                data.append(s.values)
                ticks.append(str(level))
        if data:
            plt.violinplot(data, showmeans=True, showmedians=False)
            plt.xticks(np.arange(1, len(ticks) + 1), ticks)
            plt.xlabel(by)
            plt.ylabel(col)
            plt.title(title or f"{col} by {by} (violin)")
            savefig(fname or f"violin_{col}_by_{by}.png")
            return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        print(f"[WARN] violin skipped: {col} not numeric/available.")
        plt.close()
        return
    plt.violinplot([s.values], showmeans=True)
    plt.xticks([1], [col])
    plt.title(title or f"Violin of {col}")
    savefig(fname or f"violin_{col}.png")

def plot_heatmap_like(matrix: np.ndarray, xlabels: List[str], ylabels: List[str], title: Optional[str] = None, fname: str = "heatmap.png"):
    plt.figure()
    plt.imshow(matrix, aspect="auto")
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=45, ha="right")
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.colorbar(label="Value")
    plt.title(title or "Heatmap")
    savefig(fname)

def plot_dashboard(df: pd.DataFrame, num_cols: List[str]):
    # Multi-subplot 2x2 summary (single figure with multiple plots)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # 1) histgram
    try:
        s = pd.to_numeric(df[num_cols[0]], errors="coerce").dropna()
        axes[0,0].hist(s, bins=30)
        axes[0,0].set_title(f"Hist: {num_cols[0]}")
    except Exception:
        axes[0,0].set_visible(False)
    # 2) scatter plot
    if len(num_cols) >= 2:
        try:
            x = pd.to_numeric(df[num_cols[0]], errors="coerce")
            y = pd.to_numeric(df[num_cols[1]], errors="coerce")
            m = x.notna() & y.notna()
            axes[0,1].scatter(x[m], y[m])
            axes[0,1].set_title(f"Scatter: {num_cols[1]} vs {num_cols[0]}")
        except Exception:
            axes[0,1].set_visible(False)
    else:
        axes[0,1].set_visible(False)
    # 3) boxplot
    try:
        s2 = pd.to_numeric(df[num_cols[0]], errors="coerce").dropna()
        axes[1,0].boxplot(s2.values, labels=[num_cols[0]])
        axes[1,0].set_title(f"Box: {num_cols[0]}")
    except Exception:
        axes[1,0].set_visible(False)
    # 4) violin plot
    try:
        axes[1,1].violinplot([s.values], showmeans=True)
        axes[1,1].set_xticks([1])
        axes[1,1].set_xticklabels([num_cols[0]])
        axes[1,1].set_title(f"Violin: {num_cols[0]}")
    except Exception:
        axes[1,1].set_visible(False)

    plt.suptitle("Quick Dashboard")
    savefig("dashboard_quick.png")

#  Orchestrator 

def run_all_plots(df: pd.DataFrame, ds_name: str = "Dataset", num_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):
    # choose columns
    num_cols = pick_numeric(df, num_cols)
    cat_cols = pick_categorical(df, cat_cols)

    if not num_cols:
        # if still no numerics, synthesize one from index
        df = df.copy()
        df["_row_ix"] = np.arange(len(df))
        num_cols = ["_row_ix"]

    # 1) Histogram (first numeric)
    plot_histogram(df, num_cols[0], title=f"{ds_name}: Distribution of {num_cols[0]}", fname=f"{ds_name}_hist_{num_cols[0]}.png")

    # 2) Scatter plot(first two numerics)
    if len(num_cols) >= 2:
        plot_scatter(df, num_cols[0], num_cols[1], title=f"{ds_name}: {num_cols[1]} vs {num_cols[0]}", fname=f"{ds_name}_scatter_{num_cols[0]}_{num_cols[1]}.png")

    # 3) Box plot(by first categorical if available)
    if cat_cols:
        plot_box(df, num_cols[0], by=cat_cols[0], title=f"{ds_name}: {num_cols[0]} by {cat_cols[0]}", fname=f"{ds_name}_box_{num_cols[0]}_by_{cat_cols[0]}.png")
        plot_violin(df, num_cols[0], by=cat_cols[0], title=f"{ds_name}: {num_cols[0]} by {cat_cols[0]} (violin)", fname=f"{ds_name}_violin_{num_cols[0]}_by_{cat_cols[0]}.png")
    else:
        plot_box(df, num_cols[0], title=f"{ds_name}: Box of {num_cols[0]}", fname=f"{ds_name}_box_{num_cols[0]}.png")
        plot_violin(df, num_cols[0], title=f"{ds_name}: Violin of {num_cols[0]}", fname=f"{ds_name}_violin_{num_cols[0]}.png")

    # 4) Bar graph(category counts if categorical available; else bin numeric)
    if cat_cols:
        plot_bar(df, cat_cols[0], title=f"{ds_name}: Bar of {cat_cols[0]}", fname=f"{ds_name}_bar_{cat_cols[0]}.png")
        # grouped mean of numeric by categorical
        plot_bar(df, num_cols[0], by=cat_cols[0], title=f"{ds_name}: Mean {num_cols[0]} by {cat_cols[0]}", fname=f"{ds_name}_bar_mean_{num_cols[0]}_by_{cat_cols[0]}.png")
    else:
        # numeric → discretize into bins for bar
        s = pd.to_numeric(df[num_cols[0]], errors="coerce").dropna()
        bins = np.linspace(s.min(), s.max(), 11)
        cats = pd.cut(s, bins=bins, include_lowest=True)
        vc = cats.value_counts().sort_index()
        plt.figure()
        vc.plot(kind="bar")
        plt.title(f"{ds_name}: Binned {num_cols[0]} counts")
        plt.ylabel("Count")
        savefig(f"{ds_name}_bar_binned_{num_cols[0]}.png")

    # 5) Grouped bar (RAG vs GraphRAG) — demo if metrics present, else synthetic
    if hasattr(df, "_meta"):
        k_values = df._meta["k_values"]
        rag = df._meta["rag_scores"]
        grag = df._meta["grag_scores"]
    else:
        k_values = np.array([3, 5, 10, 15, 20])
        rng = np.random.default_rng(0)
        rag = np.clip(rng.normal(0.55, 0.05, size=5), 0, 1)
        grag = np.clip(rag + rng.normal(0.05, 0.02, size=5), 0, 1)

    labels = [f"k={int(k)}" for k in k_values]
    plot_grouped_bar(labels, list(rag), list(grag),
                     ylabel="Avg top-k cosine", title=f"{ds_name}: RAG vs GraphRAG", fname=f"{ds_name}_grouped_bar_rag_grag.png")

    # 6) Stacked bar (e.g., component A/B)
    comp_a = list(np.maximum(0, rag - 0.02))
    comp_b = list(np.maximum(0, grag - np.array(comp_a)))
    plot_stacked_bar(labels, comp_a, comp_b, title=f"{ds_name}: stacked components", fname=f"{ds_name}_stacked_bar.png")

    # 7) Line graph(performance vs k)
    plot_line(list(k_values), list(rag), label="RAG", title=f"{ds_name}: Performance vs k (RAG)", fname=f"{ds_name}_line_rag.png")
    plot_line(list(k_values), list(grag), label="GraphRAG", title=f"{ds_name}: Performance vs k (GraphRAG)", fname=f"{ds_name}_line_grag.png")

    # 8) Area (stacked across two series)
    plot_area(list(k_values), [rag, grag], ["RAG", "GraphRAG"], title=f"{ds_name}: cumulative area", fname=f"{ds_name}_area.png")

    # 9) Heatmap-like (datasets × models)
    mat = np.vstack([rag, grag])
    plot_heatmap_like(mat, xlabels=[f"k={int(k)}" for k in k_values], ylabels=["RAG", "GraphRAG"],
                      title=f"{ds_name}: score matrix", fname=f"{ds_name}_heatmap.png")

    # 10) Multi-subplot dashboard (quick overview)
    if len(num_cols) >= 1:
        plot_dashboard(df, num_cols)

#  CLI (Command Line Interface)

def parse_cols(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [c.strip() for c in arg.split(",") if c.strip()]

def main():
    ap = argparse.ArgumentParser(description="Supervisor-friendly plotting suite (hist, bar, line, scatter, box, violin, heatmap, dashboard).")
    ap.add_argument("--demo", action="store_true", help="Generate plots using synthetic demo data.")
    ap.add_argument("--csv", type=str, help="Path/URL to CSV (optional).")
    ap.add_argument("--num-cols", type=str, help="Comma-separated numeric columns to use.")
    ap.add_argument("--cat-cols", type=str, help="Comma-separated categorical columns to use.")
    args = ap.parse_args()

    if args.demo and args.csv:
        print("[INFO] --csv ignored because --demo was provided.")
    if args.demo or not args.csv:
        df = make_demo_data()
        run_all_plots(df, ds_name="Demo", num_cols=parse_cols(args.num_cols), cat_cols=parse_cols(args.cat_cols))
    else:
        df = load_csv(args.csv)
        run_all_plots(df, ds_name=Path(args.csv).stem, num_cols=parse_cols(args.num_cols), cat_cols=parse_cols(args.cat_cols))

    print("All plots generated in ./figures")

if __name__ == "__main__":
    main()
