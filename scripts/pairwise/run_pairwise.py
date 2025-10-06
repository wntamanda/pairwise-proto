#!/usr/bin/env python3
import argparse
import csv
import datetime
import json
import os
import re

from scripts.pairwise.passes import apply_sequence
from scripts.pairwise.metrics import compute_metrics
from scripts.pairwise.io_utils import load_tket, load_meta, find_circuits

ID_COLS = [
    "timestamp",
    "gateset",
    "family",
    "size",
    "variant",
    "passA",
    "passB",
    "direction",
]

PRETTY_COLS = (
    ID_COLS
    + ["depth_before", "depth_after", "depth_delta"]
    + ["n_ops_total_before", "n_ops_total_after", "n_ops_total_delta"]
    + ["n_ops_total_gates_before", "n_ops_total_gates_after", "n_ops_total_gates_delta"]
    + ["n_ops1_before", "n_ops1_after", "n_ops1_delta"]
    + ["n_ops2_before", "n_ops2_after", "n_ops2_delta"]
    + ["other_before", "other_after", "other_delta"]
    + ["barrier_before", "barrier_after", "barrier_delta"]
    + ["measure_before", "measure_after", "measure_delta"]
    + ["reset_before", "reset_after", "reset_delta"]
    + ["top_ops_before", "top_ops_after"]
    + ["n_qubits_before", "n_qubits_after"]
)

def row_pretty(
    gateset,
    family,
    size,
    variant,
    passA,
    passB,
    direction,
    before,
    after,
):
    ts = datetime.datetime.now().isoformat(timespec="seconds")

    # Helper: safe-get with default
    def g(d, k, default=0):
        return d.get(k, default)

    # "Other" bucket (everything not counted by 1q/2q)
    other_before = before["n_ops_total"] - (before["n_ops_1q"] + before["n_ops_2q"])
    other_after  = after["n_ops_total"]  - (after["n_ops_1q"]  + after["n_ops_2q"])

    # Compact op histograms as JSON strings (top 8)
    top_ops_before_json = json.dumps(g(before, "top_ops", []))
    top_ops_after_json  = json.dumps(g(after,  "top_ops", []))

    return {
        "timestamp": ts,
        "gateset": gateset, "family": family, "size": size, "variant": variant,
        "passA": passA, "passB": passB, "direction": direction,

        # Depth
        "depth_before": before["depth"], "depth_after": after["depth"],
        "depth_delta": after["depth"] - before["depth"],

        # Totals (including directives)
        "n_ops_total_before": before["n_ops_total"],
        "n_ops_total_after":  after["n_ops_total"],
        "n_ops_total_delta":  after["n_ops_total"] - before["n_ops_total"],

        # Gates-only totals (excludes barrier/measure/reset)
        "n_ops_total_gates_before": g(before, "n_ops_total_gates", before["n_ops_total"]),
        "n_ops_total_gates_after":  g(after,  "n_ops_total_gates",  after["n_ops_total"]),
        "n_ops_total_gates_delta":  g(after,  "n_ops_total_gates",  after["n_ops_total"]) \
                                  - g(before, "n_ops_total_gates", before["n_ops_total"]),

        # 1q / 2q breakdown
        "n_ops1_before": before["n_ops_1q"], "n_ops1_after": after["n_ops_1q"],
        "n_ops1_delta":  after["n_ops_1q"] - before["n_ops_1q"],
        "n_ops2_before": before["n_ops_2q"], "n_ops2_after": after["n_ops_2q"],
        "n_ops2_delta":  after["n_ops_2q"] - before["n_ops_2q"],

        # Other bucket
        "other_before": other_before,
        "other_after":  other_after,
        "other_delta":  other_after - other_before,

        # Directive counts
        "barrier_before": g(before, "n_ops_directive_barrier"),
        "barrier_after":  g(after,  "n_ops_directive_barrier"),
        "barrier_delta":  g(after,  "n_ops_directive_barrier") - g(before, "n_ops_directive_barrier"),

        "measure_before": g(before, "n_ops_directive_measure"),
        "measure_after":  g(after,  "n_ops_directive_measure"),
        "measure_delta":  g(after,  "n_ops_directive_measure") - g(before, "n_ops_directive_measure"),

        "reset_before": g(before, "n_ops_directive_reset"),
        "reset_after":  g(after,  "n_ops_directive_reset"),
        "reset_delta":  g(after,  "n_ops_directive_reset") - g(before, "n_ops_directive_reset"),

        # Compact histograms
        "top_ops_before": top_ops_before_json,
        "top_ops_after":  top_ops_after_json,

        # Qubit count
        "n_qubits_before": before["n_qubits"], "n_qubits_after": after["n_qubits"],
    }

def parse_size(meta: dict, pjson_basename: str):
    size = meta.get("size") or meta.get("n_qubits") or meta.get("n")
    if size is not None:
        return int(size)
    m = re.search(r"_n(\d+)", pjson_basename)
    return int(m.group(1)) if m else None

def parse_meta_variant_bits(meta: dict, family: str):
    # Returns (reps, seed, params_tag) where params_tag in {"sym","num",None}
    reps = meta.get("reps")
    seed = meta.get("seed")
    params = meta.get("parameters")  # "symbolic" or "numeric", typically
    tag = (
        "sym" if isinstance(params, str) and params.lower().startswith("symbolic")
        else ("num" if isinstance(params, str) and params.lower().startswith("numeric") else None)
    )
    return (int(reps) if reps is not None else None,
            int(seed) if seed is not None else None,
            tag)

def parse_filename_variant_bits(basename: str, family: str):
    # Fallback: parse from filename stem
    if family == "qaoa":
        m = re.search(r"_r(\d+)_seed(\d+)_(sym|num)__", basename)
        if m: return int(m.group(1)), int(m.group(2)), m.group(3)
    elif family == "vqe_two_local":
        m = re.search(r"_r(\d+)_(sym|num)__", basename)
        if m: return int(m.group(1)), None, m.group(2)
    return None, None, None

def derive_variant_string(reps, seed, params_tag, family: str) -> str:
    parts = []
    if family in ("qaoa","vqe_two_local"):
        if reps is not None: parts.append(f"r{reps}")
        if family == "qaoa" and seed is not None: parts.append(f"seed{seed}")
        if params_tag: parts.append(params_tag)
    return "_".join(parts)

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def open_writer(path: str, fieldnames, append=True):
    mode = "a" if append and os.path.exists(path) else "w"
    ensure_parent(path)
    f = open(path, mode, newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if mode == "w":
        w.writeheader()
    return f, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits-root", default="circuits")
    ap.add_argument("--results-root",  default="results/pairwise")
    ap.add_argument("--gatesets",      default="ibm_falcon,quantinuum")
    ap.add_argument("--families",      default="qft,ghz,grover,vbe_adder,vqe_two_local,qaoa,randomcircuit")
    ap.add_argument("--sizes",         default="")                  # e.g. "8,16"
    ap.add_argument("--reps",          default="", help="Filter reps for qaoa and vqe_two_local, e.g. '1,3'")
    ap.add_argument("--qaoa-seeds",    default="", help="Filter seeds for qaoa, e.g. '11,22,33'")
    ap.add_argument("--only-symbolic", action="store_true", help="Keep only circuits with symbolic parameters")
    ap.add_argument("--pairs",         default="RB-RR,RB-CTM,RR-CTM")
    ap.add_argument("--direction",     default="both", choices=["both","A_then_B","B_then_A"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--fresh-summary", action="store_true", help="Overwrite scoped summary file instead of appending")
    ap.add_argument("--verbose",       action="store_true")
    args = ap.parse_args()

    size_filter = {int(s) for s in args.sizes.split(",") if s}
    reps_filter = {int(r) for r in args.reps.split(",") if r}
    seeds_filter = {int(s) for s in args.qaoa_seeds.split(",") if s}  # qaoa only
    pairs = [tuple(p.split("-")) for p in args.pairs.split(",") if p]
    date_str = str(datetime.date.today())

    for gateset in args.gatesets.split(","):
        for family in args.families.split(","):
            # Scoped summary named with family + gateset
            scoped_path = os.path.join(
                args.results_root, "summary", gateset, family, f"{date_str}_{family}_{gateset}_pairwise.csv"
            )
            sf, scoped_writer = open_writer(scoped_path, PRETTY_COLS, append=not args.fresh_summary)

            found_any = False
            for pjson, pmeta in find_circuits(args.circuits_root, gateset, family):
                found_any = True
                if args.verbose:
                    print(f"[found] {pjson}  |  meta={pmeta}")

                meta = load_meta(pmeta)
                basename = os.path.basename(pjson)

                size = parse_size(meta, basename)
                if size is None:
                    if args.verbose:
                        print(f"[skip] Could not determine size for: {pjson}")
                    continue
                if size_filter and size not in size_filter:
                    continue

                # reps/seed/params
                reps_m, seed_m, tag_m = parse_meta_variant_bits(meta, family)
                reps_f, seed_f, tag_f = parse_filename_variant_bits(basename, family)
                reps = reps_m if reps_m is not None else reps_f
                seed = seed_m if seed_m is not None else seed_f
                params_tag = tag_m if tag_m is not None else tag_f

                # Apply filters
                if family in ("qaoa","vqe_two_local") and reps_filter and (reps is None or reps not in reps_filter):
                    if args.verbose:
                        print(f"[filter] reps mismatch (have={reps}, need in {sorted(reps_filter)})")
                    continue
                if family == "qaoa" and seeds_filter and (seed is None or seed not in seeds_filter):
                    if args.verbose:
                        print(f"[filter] seed mismatch (have={seed}, need in {sorted(seeds_filter)})")
                    continue
                if args.only_symbolic and params_tag not in ("sym", None):  # allow None if not encoded
                    if args.verbose:
                        print(f"[filter] params not symbolic (tag={params_tag})")
                    continue

                variant = derive_variant_string(reps, seed, params_tag, family)

                # Per-circuit dir
                outdir = os.path.join(args.results_root, gateset, family, f"n{size}")
                os.makedirs(outdir, exist_ok=True)

                # Baseline
                base = load_tket(pjson)
                base_metrics = compute_metrics(base)

                for (A, B) in pairs:
                    plans = [("A_then_B", [A, B]), ("B_then_A", [B, A])]
                    if args.direction != "both":
                        plans = [x for x in plans if x[0] == args.direction]

                    for direction, seq in plans:
                        outfile = os.path.join(outdir, f"{family}_n{size}_{variant}__{A}_{B}__{direction}.csv")
                        if args.skip_existing and os.path.exists(outfile):
                            if args.verbose:
                                print(f"[skip-existing] {outfile}")
                            continue

                        # Apply passes on a copy
                        after_circ = apply_sequence(base, seq)
                        after_metrics = compute_metrics(after_circ)

                        # Save before/after frozen tket jsons for debugging
                        safe_base_name = f"{family}_n{size}_{variant}__{A}_{B}__{direction}"
                        before_json = os.path.join(outdir, safe_base_name + "__before.tket.json")
                        after_json  = os.path.join(outdir, safe_base_name + "__after.tket.json")
                        try:
                            with open(before_json, "w") as f:
                                json.dump(base.to_dict(), f, indent=2)
                            with open(after_json, "w") as f:
                                json.dump(after_circ.to_dict(), f, indent=2)
                        except Exception as e:
                            if args.verbose:
                                print(f"[warn] Could not write tket jsons: {e}")

                        row = row_pretty(gateset, family, size, variant, A, B, direction, base_metrics, after_metrics)

                        # Per-circuit CSV
                        os.makedirs(os.path.dirname(outfile), exist_ok=True)
                        with open(outfile, "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=PRETTY_COLS)
                            w.writeheader()
                            w.writerow(row)

                        # Append to scoped summary
                        scoped_writer.writerow(row)

            if args.verbose and not found_any:
                print(f"[warn] No circuits under {args.circuits_root}/{gateset}/{family}")

            sf.close()

if __name__ == "__main__":
    main()
