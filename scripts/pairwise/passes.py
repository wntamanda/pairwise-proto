from pytket.passes import RemoveBarriers, RemoveRedundancies, CommuteThroughMultis
from pytket.circuit import Circuit

PASS_OBJS = {
    "RB": RemoveBarriers(),
    "RR": RemoveRedundancies(),
    "CTM": CommuteThroughMultis(),
}
PAIRS = [("RB","RR"), ("RB","CTM"), ("RR","CTM")]

def apply_sequence(circ: Circuit, seq_names: list[str]) -> Circuit:
    out = circ.copy()
    for name in seq_names:
        PASS_OBJS[name].apply(out)
    return out
