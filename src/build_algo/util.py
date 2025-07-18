from typing import Dict, Tuple

import numpy as np


def seq_to_array(seq: str) -> np.ndarray:
    arr = np.zeros(len(seq), dtype=np.int8)
    a = 0b0001
    c = 0b0010
    g = 0b0100
    t = 0b1000
    for idx_, c_ in enumerate(seq):
        match c_.upper():
            case "A":
                arr[idx_] = a
            case "C":
                arr[idx_] = c
            case "G":
                arr[idx_] = g
            case "T":
                arr[idx_] = t
            case "R":
                arr[idx_] = a + g
            case "Y":
                arr[idx_] = c + t
            case "S":
                arr[idx_] = g + c
            case "W":
                arr[idx_] = a + t
            case "K":
                arr[idx_] = g + t
            case "M":
                arr[idx_] = a + c
            case "B":
                arr[idx_] = c + g + t
            case "D":
                arr[idx_] = a + g + t
            case "H":
                arr[idx_] = a + c + t
            case "V":
                arr[idx_] = a + c + g
            case "N":
                arr[idx_] = a + c + g + t
            case _:
                # gap
                arr[idx_] = 0
    return arr


def read_phylip(filename) -> Tuple[int, Dict[str, np.ndarray]]:
    data: Dict[str, np.ndarray] = dict()
    with open(filename, "r") as file:
        n_species, seq_len = map(lambda s: int(s.strip()), file.readline().strip().split())

        for line in file:
            species, *sequence = line.strip().split()
            sequence = "".join(sequence)
            assert len(sequence) == seq_len, f"seq length mismatch for {species}"
            data[species] = seq_to_array(sequence)
    return seq_len, data
