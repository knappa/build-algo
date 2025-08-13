from functools import reduce
from typing import List, Literal, Optional, Sequence, Set, Tuple

import ete3
import numpy as np
import scipy

PartitionMethods = Literal["spec_lap", "agg_cluster", "cograph_spectral", "spectral_consensus"]


def get_triplets_from_file(trip_file) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
    """
    Load unweighted triplets from a file

    :param trip_file: filename
    :return: tuple consisting of the set of taxa and a list of triplets
    """
    taxa = set()
    triplets = list()

    with open(trip_file, "r") as file:
        for line in file:
            a, b, c, _ = parse_triplet_line(line)
            triplets.append((a, b, c))
            taxa = taxa.union([a, b, c])

    return taxa, triplets


def get_triplets_from_string(trip_str) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
    """
    Load unweighted triplets from a string

    :param trip_str: filename
    :return: list of triplets
    """

    taxa = set()
    triplets = list()

    for line in trip_str.split("\n"):
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        a, b, c, _ = parse_triplet_line(line)
        triplets.append((a, b, c))
        taxa = taxa.union([a, b, c])

    return taxa, triplets


def parse_triplet_line(line) -> Tuple[str, str, str, Optional[float]]:
    """
    Parse a line of a triplet file
    :param line: triplet encoded with the syntax "a,b|c weight"
    :return:
    """
    a_end = line.find(",")
    if a_end == -1:
        raise Exception(f"Error at:\n{line}")
    a = line[:a_end].strip()
    line = line[a_end + 1 :].strip()

    b_end = line.find("|")
    if b_end == -1:
        raise Exception(f"Error at:\n{line}")
    b = line[:b_end].strip()
    line = line[b_end + 1 :].strip()

    c_end = line.find(" ")
    if c_end > 0:
        c = line[:c_end].strip()
        try:
            weight = float(line[c_end + 1 :].strip())
        except ValueError:
            raise Exception(f"Error in weight:\n{line[c_end+1:].strip()}")
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
    # noinspection PyPep8Naming
    L = np.diag(degree) - adj_matrix
    evals, evecs = np.linalg.eigh(L)  # right eigenvectors
    evals = evals.real.astype(np.float16).astype(np.float64)
    evecs = evecs.real.astype(np.float16).astype(np.float64)
    special_evals = np.isclose(evals, 0.0)
    num_special_evals = np.sum(special_evals)
    if num_special_evals > 1:
        # When the graph is not connected, first eigenvalues are indicators for components.
        # noinspection PyPep8Naming
        L, U = scipy.linalg.lu(
            evecs[:, special_evals].T,
            permute_l=True,
        )
        # noinspection PyPep8Naming
        U = U.real.astype(np.float16)
        component_vec = U[-1, :]
        zero_components = np.isclose(component_vec, 0.0)
        np_taxa = np.array(taxa)
        component_a = np_taxa[zero_components]
        component_b = np_taxa[~zero_components]
    else:
        idcs = np.argsort(evals)
        second_smallest_idx = idcs[1]
        special_evals = np.isclose(evals, evals[second_smallest_idx])

        if np.sum(special_evals) > 1:
            # noinspection PyPep8Naming
            L, U = scipy.linalg.lu(
                evecs[:, special_evals].T,
                permute_l=True,
            )
            # noinspection PyPep8Naming
            U = U.real.astype(np.float16)
            component_vec = U[-1, :]
        else:
            component_vec = evecs[:, second_smallest_idx]

        pos_count = np.sum(component_vec > 0)
        neg_count = np.sum(component_vec < 0)
        # put the zeros with whichever side is smaller, ties to negative side
        np_taxa = np.array(taxa)
        if pos_count < neg_count:
            component_a = np_taxa[component_vec >= 0]
            component_b = np_taxa[component_vec < 0]
        else:
            component_a = np_taxa[component_vec > 0]
            component_b = np_taxa[component_vec <= 0]
    components = [component_a, component_b]
    return components


def cospectral_laplacian_partition(*, adj_matrix, taxa):
    """
    Partition a graph using a spectral laplacian method based on the cograph. The graph should be approximately
    a disjoint union of two complete graphs. The method partitions the

    :param adj_matrix: adjacency matrix of the graph (indices should correspond to order in `taxa`)
    :param taxa: list of taxon/vertex names
    :return: partition of `taxa` in the form of a pair of arrays
    """
    import warnings

    # noinspection PyUnresolvedReferences
    coadj_matrix = (adj_matrix != 0).astype(np.float64)
    codegree = np.sum(coadj_matrix, axis=1)
    coL = np.diag(codegree) - coadj_matrix

    evals, evecs = np.linalg.eigh(coL)  # right eigenvectors
    evals = evals.real.astype(np.float64)
    evecs = evecs.real.astype(np.float64)

    if evals[0] >= 0.0:
        warnings.warn(
            "No negative eigenvalues for graph complement; a good splitting is doubtful. "
            "Falling back to standard spectral method."
        )
        return spectral_laplacian_partition(adj_matrix=adj_matrix, taxa=taxa)

    component_vec = evecs[:, 0]

    pos_count = np.sum(component_vec > 0)
    neg_count = np.sum(component_vec < 0)
    # put the zeros with whichever side is smaller, ties to negative side
    np_taxa = np.array(taxa)
    if pos_count < neg_count:
        component_a = np_taxa[component_vec >= 0]
        component_b = np_taxa[component_vec < 0]
    else:
        component_a = np_taxa[component_vec > 0]
        component_b = np_taxa[component_vec <= 0]

    components = [component_a, component_b]
    return components


def agglomerative_spectral_partition(*, adj_matrix, taxa):
    """
    Partition a graph using an agglomerative method in the spectral embedding.

    :param adj_matrix: adjacency matrix of the graph (indices should correspond to order in `taxa`)
    :param taxa: list of taxon/vertex names
    :return: partition of `taxa` in the form of a pair of arrays
    """
    from sklearn.cluster import AgglomerativeClustering

    degree = np.sum(adj_matrix, axis=1)
    # noinspection PyPep8Naming
    L = np.diag(degree) - adj_matrix
    evals, evecs = np.linalg.eigh(L)  # right eigenvectors
    evals = evals.real.astype(np.float64)
    evecs = evecs.real.astype(np.float64)

    # Only the smaller eigenvalues should be used as the higher eigenvalues tend to be "high frequency vibrations"
    # (i.e. mostly noise) So we take the bottom quarter by absolute value. This is somewhat arbitrary and might
    # be too large for more bigger trees. An alternative would be to cap it at 2 or 3.
    tame_evals = evals <= np.quantile(np.abs(evals), 0.25)
    embedding_vectors = evecs[:, tame_evals]

    labels: np.ndarray = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(
        embedding_vectors
    )

    np_taxa = np.array(taxa)
    component_a = np_taxa[labels == 0]
    component_b = np_taxa[labels != 0]

    components = [component_a, component_b]
    return components


def spectral_consensus_partition(*, adj_matrix, taxa):
    import itertools

    component_a1, component_b1 = spectral_laplacian_partition(adj_matrix=adj_matrix, taxa=taxa)
    component_a2, component_b2 = cospectral_laplacian_partition(adj_matrix=adj_matrix, taxa=taxa)

    a1_indicator = np.array([taxon in component_a1 for taxon in taxa], dtype=bool)
    a2_indicator = np.array([taxon in component_a2 for taxon in taxa], dtype=bool)

    # try to match these as best as possible
    agreement = np.sum(a1_indicator == a2_indicator)
    counter_agreement = np.sum(a1_indicator == ~a2_indicator)
    if counter_agreement > agreement:
        a2_indicator = ~a2_indicator

    disagreement_locus = a1_indicator != a2_indicator
    num_disagreements = np.sum(disagreement_locus)

    if num_disagreements == 0:
        np_taxa = np.array(taxa)
        component_a = np_taxa[a1_indicator]
        component_b = np_taxa[~a1_indicator]
        return [component_a, component_b]

    print(f"num_disagreements: {num_disagreements}")

    # brute force it on the sites where the methods disagree.
    def score(indicator):
        partition_a_size = np.sum(indicator)
        partition_b_size = np.sum(~indicator)
        if partition_a_size == 0 or partition_b_size == 0:
            return float("inf")
        edge_penalty = np.sum(adj_matrix[indicator, :][:, ~indicator])
        return (
            edge_penalty + partition_a_size / partition_b_size + partition_b_size / partition_a_size
        )

    disagreement_sites = np.arange(len(taxa))[disagreement_locus]
    best_a_pattern = a1_indicator.copy()
    best_score = score(best_a_pattern)
    # TODO: limit the number of tries?
    for count, ambig_pattern in enumerate(
        itertools.product([True, False], repeat=len(disagreement_sites))
    ):
        a_indicator = a1_indicator.copy()
        a_indicator[disagreement_locus] = ambig_pattern
        test_score = score(a_indicator)
        if test_score < best_score:
            best_score = test_score
            best_a_pattern = a_indicator

    np_taxa = np.array(taxa)
    component_a = np_taxa[best_a_pattern]
    component_b = np_taxa[~best_a_pattern]

    return [component_a, component_b]


def gen_tree_from_triplet_file(triplet_file) -> str:
    taxa, triplets = get_triplets_from_file(triplet_file)
    return gen_tree(triplets).write(format=9)


def gen_tree_from_string(triplet_string) -> str:
    taxa, triplets = get_triplets_from_string(triplet_string)
    return gen_tree(triplets).write(format=9)


def gen_tree(
    triplets: Sequence[Tuple[str, str, str]],
    *,
    method: PartitionMethods = "spec_lap",
) -> ete3.TreeNode:
    """
    Generate a tree from a list of triplets, possibly with errors.

    :param triplets: sequence of triples a,b|c encoded as tuples (a,b,c).
    :param method: which method to use to partition taxa
    :return: ete3 Tree corresponding to the triplets
    """
    tree = ete3.Tree()
    _gen_tree(triplets=triplets, node=tree, method=method)
    return tree


def _gen_tree(
    *,
    triplets: Sequence[Tuple[str, str, str]],
    node: ete3.Tree,
    method: PartitionMethods = "spec_lap",
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
        return

    # build the adjacency matrix for the spectral laplacian
    taxa: list = list(set(reduce(set.union, triplets, set())))
    taxa.sort()
    taxa_to_index = {s: i for i, s in enumerate(taxa)}
    adj_matrix = np.zeros((len(taxa), len(taxa)), dtype=np.float64)
    for a, b, c in triplets:
        adj_matrix[taxa_to_index[a], taxa_to_index[b]] += 1
        adj_matrix[taxa_to_index[b], taxa_to_index[a]] += 1

    match method:
        case "spec_lap":
            components = spectral_laplacian_partition(adj_matrix=adj_matrix, taxa=taxa)
        case "agg_cluster":
            components = agglomerative_spectral_partition(adj_matrix=adj_matrix, taxa=taxa)
        case "cograph_spectral":
            components = cospectral_laplacian_partition(adj_matrix=adj_matrix, taxa=taxa)
        case "spectral_consensus":
            components = spectral_consensus_partition(adj_matrix=adj_matrix, taxa=taxa)

    assert len(components) > 1, "No splitting found"
    assert all(len(component) > 0 for component in components), "One split was empty"

    for component in components:
        if len(component) == 1:
            # single taxon component is added as a leaf
            member = list(component)[0]
            node.add_child(name=member)
        elif len(component) > 1:
            # Either a multifurcation (hopefully bifurcation), or we need to recurse
            # filter triplets by component
            component_triplets = [
                triplet for triplet in triplets if all([x in component for x in triplet])
            ]
            subnode = node.add_child()
            if len(component_triplets) == 0:
                # if there are no triplets, this must be a multifurcation node. Hopefully a cherry.
                for member in component:
                    subnode.add_child(name=member)
            else:
                _gen_tree(triplets=component_triplets, node=subnode)
