# metrics.py
from __future__ import annotations
from collections import Counter
from typing import Dict, Any, Iterable, Optional
from pytket.circuit import Circuit

# Ops as "directives"/non-gate for reporting "gates-only" totals
_DIRECTIVE_NAMES = {"barrier", "measure", "reset"}

def _op_name_counts(circ: Circuit) -> Counter:
    """
    Robust, lowercase op-name counter. Includes directives like 'barrier', 'measure', 'reset'.
    """
    names = []
    for cmd in circ.get_commands():
        op = getattr(cmd, "op", None)
        if op is None:
            names.append(str(cmd).lower())
            continue
        nm = getattr(op, "name", None) or getattr(op, "type", None) or str(op)
        if hasattr(nm, "name"):
            nm = nm.name
        names.append(str(nm).lower())
    return Counter(names)

def _count_ops_by_qubits(circ: Circuit):
    n1 = n2 = ntot = 0
    for cmd in circ.get_commands():
        # total counts every instruction (incl. barrier/measure/reset)
        ntot += 1
        k = cmd.op.n_qubits
        if k == 1: n1 += 1
        elif k == 2: n2 += 1
    return ntot, n1, n2

def compute_metrics(circ: Circuit) -> Dict[str, Any]:
    """
    Returns:
      - depth, n_qubits
      - n_ops_total (includes directives)
      - n_ops_total_gates (excludes barrier/measure/reset)
      - n_ops_1q, n_ops_2q
      - n_ops_directive_barrier/measure/reset
      - top_ops (JSON-ready list of (name, count), top 8)
    """
    ntot, n1, n2 = _count_ops_by_qubits(circ)
    counts = _op_name_counts(circ)

    # Directive counts
    n_barrier = sum(v for k, v in counts.items() if k == "barrier")
    n_measure = sum(v for k, v in counts.items() if k == "measure")
    n_reset   = sum(v for k, v in counts.items() if k == "reset")

    # Gates-only = everything minus directives listed
    directives_total = n_barrier + n_measure + n_reset
    n_total_gates_only = ntot - directives_total

    # Compact top-k to help debugging (kept small to avoid bloating CSVs)
    top_ops = counts.most_common(8)

    return {
        "n_qubits": circ.n_qubits,
        "depth": circ.depth(),
        "n_ops_total": ntot,
        "n_ops_total_gates": n_total_gates_only,
        "n_ops_1q": n1,
        "n_ops_2q": n2,
        "n_ops_directive_barrier": n_barrier,
        "n_ops_directive_measure": n_measure,
        "n_ops_directive_reset": n_reset,
        "top_ops": top_ops,  # list of tuples; run_pairwise will json.dumps it
    }
