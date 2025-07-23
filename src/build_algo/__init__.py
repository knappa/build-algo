from functools import reduce
from typing import Sequence, Tuple

import numpy as np
import scipy
from ete3 import Tree


def get_triplets_from_file(trip_file):
    """
    Load unweighted triplets from a file

    :param trip_file: filename
    :return: list of triplets
    """
    taxa = set()
    triplets = list()

    with open(trip_file, "r") as file:
        for line in file:
            a, b, c, _ = parse_triplet_line(line)
            triplets.append((a, b, c))
            taxa = taxa.union([a, b, c])

    return taxa, triplets


def get_triplets_from_string(trip_str):
    """
    Load unweighted triplets from a string

    :param trip_str: filename
    :return: list of triplets
    """

    taxa = set()
    triplets = list()

    for line in trip_str.split("\n"):
        a, b, c, _ = parse_triplet_line(line)
        triplets.append((a, b, c))
        taxa = taxa.union([a, b, c])

    return taxa, triplets


def parse_triplet_line(line):
    """
    Parse a line of a triplet file
    :param line: triplet encoded with the syntax "a,b|c weight"
    :return:
    """
    line = line.strip()
    a_end = line.find(",")
    a = line[:a_end].strip()
    line = line[a_end:].strip()
    b_end = line.find("|")
    b = line[:b_end].strip()
    line = line[b_end:].strip()
    c_end = line.find(" ")
    if c_end > 0:
        c = line[:c_end].strip()
        weight = float(line[c_end:].strip())
    else:
        c = line
        weight = None
    return a, b, c, weight


def spectral_laplacian_partition(*, adj_matrix, taxa):
    """
    Partition a graph using the spectral laplacian method.

    :param adj_matrix: adjacency matrix of the graph (indices should correspond to order in `taxa`)
    :param taxa: list of taxon/vertex names
    :return: partition of `taxa` in the form of a pair of arrays
    """
    degree = np.sum(adj_matrix, axis=1)
    L = np.diag(degree) - adj_matrix
    evals, evecs = np.linalg.eigh(L)  # right eigenvectors
    evals = evals.real.astype(np.float16).astype(np.float64)
    evecs = evecs.real.astype(np.float16).astype(np.float64)
    special_evals = np.isclose(evals, 0.0)
    num_special_evals = np.sum(special_evals)
    if num_special_evals > 1:
        # When the graph is not connected, first eigenvalues are indicators for components.
        # noinspection PyTupleAssignmentBalance
        L, U = scipy.linalg.lu(
            evecs[:, special_evals].T,
            permute_l=True,
        )
        U = U.real.astype(np.float16)
        component_vec = U[-1, :]
        zero_components = np.isclose(component_vec, 0.0)
        component_a = np.array(taxa)[zero_components]
        component_b = np.array(taxa)[~zero_components]
    else:
        idcs = np.argsort(evals)
        second_smallest_idx = idcs[1]
        special_evals = np.isclose(evals, evals[second_smallest_idx])

        if np.sum(special_evals) > 1:
            # noinspection PyTupleAssignmentBalance
            L, U = scipy.linalg.lu(
                evecs[:, special_evals].T,
                permute_l=True,
            )
            U = U.real.astype(np.float16)
            component_vec = U[-1, :]
        else:
            component_vec = evecs[:, second_smallest_idx]

        pos_count = np.sum(component_vec > 0)
        neg_count = np.sum(component_vec < 0)
        # put the zeros with whichever side is smaller, ties to negative side
        if pos_count < neg_count:
            component_a = np.array(taxa)[component_vec >= 0]
            component_b = np.array(taxa)[component_vec < 0]
        else:
            component_a = np.array(taxa)[component_vec > 0]
            component_b = np.array(taxa)[component_vec <= 0]
    components = [component_a, component_b]
    return components


def gen_tree(
    triplets: Sequence[Tuple[str, str, str]],
):
    """
    Generate a tree from a list of triplets, possibly with errors.

    :param triplets: sequence of triples a,b|c encoded as tuples (a,b,c).
    :return: ete3 Tree corresponding to the triplets
    """
    tree = Tree()
    _gen_tree(triplets=triplets, node=tree)
    return tree


def _gen_tree(
    *,
    triplets: Sequence[Tuple[str, str, str]],
    node: Tree,
) -> None:
    """
    Helper for gen_tree; generates a tree from a list of triplets, possibly with errors.

    :param triplets: sequence of triples a,b|c encoded as tuples (a,b,c).
    :param node:
    :return:
    """
    if node is None:
        assert "inconsistent state"

    if len(triplets) == 1:
        a, b, c = triplets[0]
        # noinspection PyTypeChecker
        node.add_child(name=c)
        subnode = node.add_child()
        subnode.add_child(name=a)
        subnode.add_child(name=b)

    # build the adjacency matrix for the spectral laplacian
    taxa: list = list(set(reduce(set.union, triplets, set())))
    taxa.sort()
    taxa_to_index = {s: i for i, s in enumerate(taxa)}
    adj_matrix = np.zeros((len(taxa), len(taxa)), dtype=np.float64)
    for a, b, c in triplets:
        adj_matrix[taxa_to_index[a], taxa_to_index[b]] += 1
        adj_matrix[taxa_to_index[b], taxa_to_index[a]] += 1

    components = spectral_laplacian_partition(adj_matrix=adj_matrix, taxa=taxa)

    assert len(components) > 1, "No splitting found"
    assert all(len(component) > 0 for component in components), "One split was empty"

    for component in components:
        if len(component) == 1:
            # single taxon component
            member = list(component)[0]
            node.add_child(name=member)
        elif len(component) > 1:
            # filter triplets by component
            component_triplets = [
                (a, b, c) for (a, b, c) in triplets if all([x in component for x in [a, b, c]])
            ]
            subnode = node.add_child()
            if len(component_triplets) == 0:
                for member in component:
                    subnode.add_child(name=member)
            else:
                _gen_tree(triplets=component_triplets, node=subnode)
