# io_utils.py
import json, pathlib
from pytket.circuit import Circuit

def load_tket(path_json: str) -> Circuit:
    # Mirror loader_common: read file -> from_dict
    data = json.loads(pathlib.Path(path_json).read_text())
    return Circuit.from_dict(data)

def load_meta(path_meta: str) -> dict:
    with open(path_meta, "r") as f:
        return json.load(f)

def _meta_candidates(path: pathlib.Path):
    # a) foo.tket.json -> foo.tket.meta.json
    cand_keep = path.with_suffix(".meta.json")
    # b) foo.tket.json -> foo.meta.json
    cand_drop = path.with_suffix("").with_suffix(".meta.json")
    return [cand_keep, cand_drop]

def find_circuits(root: str, gateset: str, family: str):
    base = pathlib.Path(root) / gateset / family
    for path in base.rglob("*.tket.json"):
        meta = next((m for m in _meta_candidates(path) if m.exists()), None)
        if meta:
            yield str(path), str(meta)
