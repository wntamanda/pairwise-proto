# notebooks/notebook_helper.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

# Safe import of display for notebooks; with fallback
try:
    from IPython.display import display
except Exception:
    def display(x):
        # Fallback for non-notebook environments
        # For pandas DataFrame this prints a readable table.
        print(x)

# path finding
def find_project_root(start: Path | None = None) -> Path:
    """Walk up until we find results/pairwise/summary; fallback to CWD."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(8):
        if (cur / "results" / "pairwise" / "summary").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd().resolve()

PROJECT_ROOT = find_project_root()
SUMMARY_ROOT = PROJECT_ROOT / "results" / "pairwise" / "summary"

def latest_scoped_summary(gateset: str, family: str) -> Path | None:
    """Latest CSV for one (gateset,family). Filenames: YYYY-MM-DD_<family>_<gateset>_pairwise.csv"""
    folder = SUMMARY_ROOT / gateset / family
    if not folder.exists():
        return None
    cands = sorted(folder.glob(f"*_{family}_{gateset}_pairwise.csv"))
    return cands[-1] if cands else None

# loading & augmentation
def load_scoped(gateset: str, family: str) -> pd.DataFrame:
    """Read one backend’s scoped summary into a DataFrame; empty if not found."""
    p = latest_scoped_summary(gateset, family)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["backend"] = gateset
    if "variant" not in df.columns:
        df["variant"] = ""  # families without reps/seeds

    # New metrics (backwards-compatible fill if missing)
    optional_cols = [
        "n_ops_total_gates_before","n_ops_total_gates_after","n_ops_total_gates_delta",
        "barrier_before","barrier_after","barrier_delta",
        "measure_before","measure_after","measure_delta",
        "reset_before","reset_after","reset_delta",
        "top_ops_before","top_ops_after",
        "other_before","other_after","other_delta",
    ]
    for c in optional_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def add_sequence_column(df: pd.DataFrame) -> pd.DataFrame:
    """Make explicit pass order label: 'RR→RB' instead of A/B + direction."""
    if df.empty:
        return df
    if "sequence" in df.columns:
        return df
    def _seq(r):
        d = r.get("direction")
        a, b = r.get("passA",""), r.get("passB","")
        if pd.isna(d): return f"{a}→{b}"
        return f"{a}→{b}" if str(d)=="A_then_B" else f"{b}→{a}"
    seq = df.apply(_seq, axis=1)
    df = df.copy()
    insert_at = df.columns.get_loc("direction")+1 if "direction" in df.columns else len(df.columns)
    df.insert(insert_at, "sequence", seq)
    return df

def load_family(family: str, gatesets=("ibm_falcon","quantinuum")) -> dict[str, pd.DataFrame]:
    """Load latest summaries for a family across multiple backends, add sequence column."""
    out: dict[str, pd.DataFrame] = {}
    for gs in gatesets:
        df = load_scoped(gs, family)
        if df.empty: 
            continue
        df = add_sequence_column(df)
        out[gs] = df
    return out

# tidy / optional ranking
DEFAULT_DISPLAY_COLS = [
    "backend","family","size","variant","sequence",
    "depth_before","depth_after","depth_delta",
    "n_ops_total_before","n_ops_total_after","n_ops_total_delta",
    "n_ops_total_gates_before","n_ops_total_gates_after","n_ops_total_gates_delta",
    "n_ops1_before","n_ops1_after","n_ops1_delta",
    "n_ops2_before","n_ops2_after","n_ops2_delta",
    "other_before","other_after","other_delta",
    "barrier_before","barrier_after","barrier_delta",
    "measure_before","measure_after","measure_delta",
    "reset_before","reset_after","reset_delta",
    "top_ops_before","top_ops_after",
    "n_qubits_before","n_qubits_after","timestamp",
]

def tidy(df: pd.DataFrame, display_cols=DEFAULT_DISPLAY_COLS) -> pd.DataFrame:
    """Select and order readable columns."""
    if df.empty:
        return df
    cols = [c for c in display_cols if c in df.columns]
    return df[cols].copy()

def sort_best_worst(
    df: pd.DataFrame,
    primary: str = "depth_delta",
    tiebreakers: list[str] | None = None,
    per_size: bool = True,
) -> pd.DataFrame:
    """Optional: sort best→worst (more negative primary is better)."""
    if df.empty:
        return df
    if tiebreakers is None:
        tiebreakers = ["n_ops_total_delta","n_ops2_delta","sequence"]
    keys = (["size"] if per_size and "size" in df.columns else []) + [primary] + [k for k in tiebreakers if k in df.columns]
    ascending = [True]*len(keys)  # negative deltas first
    return df.sort_values(keys, ascending=ascending).reset_index(drop=True)

def add_rank_within_size(df: pd.DataFrame, by: str = "depth_delta", rank_col: str = "rank") -> pd.DataFrame:
    """Optional: add rank per size (1 = best)."""
    if df.empty or by not in df.columns or "size" not in df.columns:
        return df
    out = df.copy()
    out[rank_col] = out.groupby("size")[by].rank(method="dense", ascending=True).astype(int)
    col = out.pop(rank_col)
    pos = out.columns.get_loc("size")+1
    out.insert(pos, rank_col, col)
    return out

# display
STYLE_FMT = {
    "rank": "{:d}", "size": "{:d}",
    "depth_before": "{:d}", "depth_after": "{:d}", "depth_delta": "{:+d}",
    "n_ops_total_before": "{:d}", "n_ops_total_after": "{:d}", "n_ops_total_delta": "{:+d}",
    "n_ops_total_gates_before": "{:d}", "n_ops_total_gates_after": "{:d}", "n_ops_total_gates_delta": "{:+d}",
    "n_ops1_before": "{:d}", "n_ops1_after": "{:d}", "n_ops1_delta": "{:+d}",
    "n_ops2_before": "{:d}", "n_ops2_after": "{:d}", "n_ops2_delta": "{:+d}",
    "other_before": "{:d}", "other_after": "{:d}", "other_delta": "{:+d}",
    "barrier_before": "{:d}", "barrier_after": "{:d}", "barrier_delta": "{:+d}",
    "measure_before": "{:d}", "measure_after": "{:d}", "measure_delta": "{:+d}",
    "reset_before": "{:d}", "reset_after": "{:d}", "reset_delta": "{:+d}",
    "n_qubits_before": "{:d}", "n_qubits_after": "{:d}",
}

def show_table(df: pd.DataFrame, caption: str):
    if df.empty:
        display(pd.DataFrame({"note": [f"(empty) {caption}"]}))
    else:
        display(df.style.format(STYLE_FMT, na_rep="-").set_caption(caption))
