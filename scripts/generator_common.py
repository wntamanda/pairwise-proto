from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import json, re, hashlib, tempfile, shutil
from datetime import datetime

# mqt bench
from mqt.bench import get_benchmark_native_gates
from mqt.bench.targets import get_target_for_gateset
from mqt.bench.benchmarks.qaoa import create_circuit as qaoa_create
from mqt.bench.benchmarks.vqe_two_local import create_circuit as vqe_create
from mqt.bench.benchmarks.qft import create_circuit as qft_create
from mqt.bench.benchmarks.ghz import create_circuit as ghz_create
from mqt.bench.benchmarks.grover import create_circuit as grover_create
from mqt.bench.benchmarks.vbe_ripple_carry_adder import create_circuit as vbe_create
from mqt.bench.benchmarks.randomcircuit import create_circuit as rc_create

# qiskit/tket
from qiskit.circuit import Parameter
from pytket.extensions.qiskit import qiskit_to_tk

Native = Literal["ibm_falcon","quantinuum"]
Family = Literal["qaoa","vqe_two_local","qft","ghz","grover","vbe_adder","randomcircuit"]

SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")

def sanitize_parameter_names(qc):
    """Rename Qiskit Parameters to SymPy-safe identifiers (e.g., g[0]->g_0, θ[1]->theta_1)."""
    rename = {}
    for p in sorted(qc.parameters, key=lambda x: x.name):
        name = p.name
        if name.startswith("θ[") or name.startswith("ϑ["):
            # VQE TwoLocal: θ[i]
            idx = name[name.index("[")+1:name.index("]")]
            new = f"theta_{idx}"
        else:
            new = SAFE_NAME_RE.sub("_", name)
            if new and new[0].isdigit():
                new = "_" + new
        if new != name:
            rename[p] = Parameter(new)
    return qc.assign_parameters(rename, inplace=False) if rename else qc

def _hash_tket(tkc) -> str:
    return hashlib.sha256(json.dumps(tkc.to_dict(), sort_keys=True).encode()).hexdigest()

def _freeze_json(tkc, path: Path, meta: Dict[str, Any], *, force=False, compare=False) -> str:
    """Write tket circ to path and meta next to it. Returns final hash of tkc JSON.
       - if exists and not force/compare: skip
       - if compare: write temp, replace only if different
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_json = json.dumps(tkc.to_dict(), indent=2)
    new_hash = hashlib.sha256(new_json.encode()).hexdigest()

    if path.exists() and not (force or compare):
        # leave as-is
        return new_hash

    if compare and path.exists():
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if old_hash == new_hash:
            # identical -> do nothing
            return new_hash

    # write atomically
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(new_json)
        tmp_path = Path(tmp.name)
    shutil.move(str(tmp_path), str(path))

    # meta
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return new_hash

# algorithmic builders (family-aware)

def build_algorithmic(family: Family, *, size: int, reps: Optional[int]=None, seed: Optional[int]=None):
    if family == "qaoa":
        assert reps is not None and seed is not None
        return qaoa_create(num_qubits=size, repetitions=reps, seed=seed) # symbolic by default
    if family == "vqe_two_local":
        assert reps is not None
        return vqe_create(num_qubits=size, reps=reps) # symbolic θ[i]
    if family == "qft":
        return qft_create(num_qubits=size)
    if family == "ghz":
        return ghz_create(num_qubits=size)
    if family == "grover":
        return grover_create(num_qubits=size)
    if family == "vbe_adder":
        # size here is as defined by MQT for the ripple-carry variant
        return vbe_create(num_qubits=size)
    if family == "randomcircuit":
        # uses mqt’s default generator
        return rc_create(num_qubits=size)
    raise ValueError(f"Unsupported family: {family}")

def to_native(qc_alg, *, native: Native, size: int, opt_level: int = 0, keep_symbolic: bool = True):
    """Compile to native gate set (no qubit mapping), optionally keep parameters symbolic."""
    target = get_target_for_gateset(native, num_qubits=size)
    qc_native = get_benchmark_native_gates(
        benchmark=qc_alg,
        circuit_size=None,
        target=target,
        opt_level=opt_level,
        random_parameters=not keep_symbolic, # keep_symbolic=True -> random_parameters=False
        generate_mirror_circuit=False,
    )
    return qc_native

# path helpers

def file_stem(family: Family, *, size: int, reps: Optional[int], seed: Optional[int],
              symbolic: bool, native: Native) -> str:
    tag = "sym" if symbolic else "num"
    if family == "qaoa":
        core = f"qaoa_n{size}_r{reps}_seed{seed}_{tag}"
    elif family == "vqe_two_local":
        core = f"vqe2l_n{size}_r{reps}_{tag}"
    elif family == "qft":
        core = f"qft_n{size}"
    elif family == "ghz":
        core = f"ghz_n{size}"
    elif family == "grover":
        core = f"grover_n{size}"
    elif family == "vbe_adder":
        core = f"vbe_adder_n{size}"
    elif family == "randomcircuit":
        core = f"random_n{size}"
    else:
        raise ValueError(family)
    return f"{core}__{native}"

def out_path(root: Path, native: Native, family: Family, *, size: int, reps: Optional[int],
             seed: Optional[int], symbolic: bool) -> Path:
    fname = file_stem(family, size=size, reps=reps, seed=seed, symbolic=symbolic, native=native) + ".tket.json"
    folder = {
        "qaoa": "qaoa", "vqe_two_local": "vqe_two_local", "qft": "qft",
        "ghz": "ghz", "grover": "grover", "vbe_adder": "vbe_adder", "randomcircuit": "randomcircuit",
    }[family]
    return root / native / folder / fname

# public entry point

@dataclass
class GenSpec:
    family: Family
    native: Native
    size: int
    reps: Optional[int] = None
    seed: Optional[int] = None
    symbolic: bool = True # QAOA/VQE: True by default; others ignored

def generate_and_freeze(spec: GenSpec, *, out_root: Path = Path("circuits"), force=False, compare=False) -> Path:
    """Build algorithmic, compile to native, sanitize, convert to TKET, freeze, return path."""
    qc_alg = build_algorithmic(spec.family, size=spec.size, reps=spec.reps, seed=spec.seed)
    keep_symbolic = spec.symbolic if spec.family in ("qaoa","vqe_two_local") else True
    qc_native = to_native(qc_alg, native=spec.native, size=spec.size, opt_level=0, keep_symbolic=keep_symbolic)
    qc_native = sanitize_parameter_names(qc_native)

    tkc = qiskit_to_tk(qc_native)
    path = out_path(out_root, spec.native, spec.family, size=spec.size, reps=spec.reps, seed=spec.seed, symbolic=keep_symbolic)

    meta = {
        "family": spec.family,
        "native_gateset": spec.native,
        "size": spec.size,
        "reps": spec.reps,
        "seed": spec.seed,
        "parameters": "symbolic" if keep_symbolic else "numeric",
        "opt_level": 0,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    _freeze_json(tkc, path, meta, force=force, compare=compare)
    return path

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    fam_choices = ["qaoa","vqe_two_local","qft","ghz","grover","vbe_adder","randomcircuit"]
    nat_choices = ["ibm_falcon","quantinuum"]

    ap = argparse.ArgumentParser(description="Generate a single frozen TKET circuit.")
    ap.add_argument("--family", required=True, choices=fam_choices)
    ap.add_argument("--native", required=True, choices=nat_choices)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--reps", type=int, help="Required for qaoa and vqe_two_local")
    ap.add_argument("--seed", type=int, help="Required for qaoa (e.g., 11,22,33)")
    ap.add_argument("--symbolic", action="store_true", help="Keep parameters symbolic (default for qaoa/vqe)")
    ap.add_argument("--numeric", dest="symbolic", action="store_false", help="Bind random params (not recommended)")
    ap.set_defaults(symbolic=True)
    ap.add_argument("--out-root", type=Path, default=Path("circuits"))
    ap.add_argument("--force", action="store_true", help="Overwrite even if file exists")
    ap.add_argument("--compare", action="store_true", help="Replace only if different (safe default)")
    args = ap.parse_args()

    spec = GenSpec(
        family=args.family,
        native=args.native,
        size=args.size,
        reps=args.reps,
        seed=args.seed,
        symbolic=args.symbolic if args.family in ("qaoa","vqe_two_local") else True,
    )
    p = generate_and_freeze(spec, out_root=args.out_root, force=args.force, compare=args.compare)
    print("✔ wrote:", p)
