# loader_common.py

from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, Literal
from pytket.circuit import Circuit

Native = Literal["ibm_falcon","quantinuum"]
Family = Literal["qaoa","vqe_two_local","qft","ghz","grover","vbe_adder","randomcircuit"]

def find_project_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for _ in range(6):
        if (cur / "circuits").is_dir() and (cur / "scripts").is_dir():
            return cur
        if cur.parent == cur: break
        cur = cur.parent
    return (start or Path.cwd()).resolve()

def file_stem(family: Family, *, size: int, reps: Optional[int], seed: Optional[int], symbolic: bool) -> str:
    tag = "sym" if symbolic else "num"
    if family == "qaoa":
        return f"qaoa_n{size}_r{reps}_seed{seed}_{tag}"
    if family == "vqe_two_local":
        return f"vqe2l_n{size}_r{reps}_{tag}"
    if family == "qft":
        return f"qft_n{size}"
    if family == "ghz":
        return f"ghz_n{size}"
    if family == "grover":
        return f"grover_n{size}"
    if family == "vbe_adder":
        return f"vbe_adder_n{size}"
    if family == "randomcircuit":
        return f"random_n{size}"
    raise ValueError(family)

def path_for(
    family: Family, native: Native, *, size: int, reps: Optional[int],
    seed: Optional[int], symbolic: bool=True, root: Optional[Path]=None
) -> Path:
    root = find_project_root(root)
    folder = {
        "qaoa": "qaoa",
        "vqe_two_local": "vqe_two_local",
        "qft": "qft",
        "ghz": "ghz",
        "grover": "grover",
        "vbe_adder": "vbe_adder",
        "randomcircuit": "randomcircuit",
    }[family]
    core = file_stem(family, size=size, reps=reps, seed=seed, symbolic=symbolic)
    fname = f"{core}__{native}.tket.json"  # append native ONLY here
    return root / "circuits" / native / folder / fname

def load_tket_json(path: Path) -> Circuit:
    if not path.exists():
        raise FileNotFoundError(f"Frozen circuit not found:\n  {path}")
    data = json.loads(path.read_text())
    return Circuit.from_dict(data)

def fresh_copy(tkc: Circuit) -> Circuit:
    return Circuit.from_dict(tkc.to_dict())

def load(
    family: Family, native: Native, *, size: int, reps: Optional[int]=None,
    seed: Optional[int]=None, symbolic: bool=True, root: Optional[Path]=None, return_copy=True
) -> Circuit:
    p = path_for(family, native, size=size, reps=reps, seed=seed, symbolic=symbolic, root=root)
    tkc = load_tket_json(p)
    return fresh_copy(tkc) if return_copy else tkc

# optional render passthrough
try:
    from pytket.circuit.display import render_circuit_jupyter as render
except Exception:
    try:
        from pytket.extensions.offline_display import render_circuit_jupyter as render  # type: ignore
    except Exception:
        def render(*_args, **_kwargs):
            print("[loader_common] Rendering not available.")
