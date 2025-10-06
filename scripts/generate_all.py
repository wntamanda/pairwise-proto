from pathlib import Path
from generator_common import GenSpec, generate_and_freeze

def main():
    out_root = Path("circuits")
    natives = ["ibm_falcon","quantinuum"]

    specs = []

    # QFT 8,16,32
    for n in [8,16,32]:
        for nat in natives:
            specs.append(GenSpec("qft", nat, size=n))

    # QAOA size 8,16; reps 1,3; seed 11,22,33 (symbolic)
    for n in [8,16]:
        for r in [1,3]:
            for s in [11,22,33]:
                for nat in natives:
                    specs.append(GenSpec("qaoa", nat, size=n, reps=r, seed=s, symbolic=True))

    # VQE TwoLocal size 8,16; reps 1,3 (symbolic)
    for n in [8,16]:
        for r in [1,3]:
            for nat in natives:
                specs.append(GenSpec("vqe_two_local", nat, size=n, reps=r, symbolic=True))

    # VBE Adder size 10,13
    for n in [10,13]:
        for nat in natives:
            specs.append(GenSpec("vbe_adder", nat, size=n))

    # Grover size 8,16
    for n in [8,16]:
        for nat in natives:
            specs.append(GenSpec("grover", nat, size=n))

    # GHZ size 16
    for nat in natives:
        specs.append(GenSpec("ghz", nat, size=16))

    # RandomCircuit size 8,16,32
    for n in [8,16,32]:
        for nat in natives:
            specs.append(GenSpec("randomcircuit", nat, size=n))

    for spec in specs:
        path = generate_and_freeze(spec, out_root=out_root, force=False, compare=True)
        print("✔", path)

if __name__ == "__main__":
    main()
