#!/usr/bin/env python3
# the BUILD algorithm as described in Semple and Steel


def main_cli():
    import argparse

    from build_algo import PartitionMethods, gen_tree, get_triplets_from_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", required=True, help="triplets file")

    parser.add_argument("--viz", action="store_true", help="print visualization")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--spec_lap", action="store_true", help="spectral laplacian method")
    group.add_argument("--agg_cluster", action="store_true", help="agglomerative clustering method")
    group.add_argument("--cograph_spectral", action="store_true", help="cograph spectral method")
    group.add_argument(
        "--spectral_consensus", action="store_true", help="consensus spectral method"
    )

    opt = parser.parse_args()
    # print(opt)

    method: PartitionMethods
    if hasattr(opt, "spec_lap") and opt.spec_lap:
        method = "spec_lap"
    elif hasattr(opt, "agg_cluster") and opt.agg_cluster:
        method = "agg_cluster"
    elif hasattr(opt, "cograph_spectral") and opt.cograph_spectral:
        method = "cograph_spectral"
    elif hasattr(opt, "spectral_consensus") and opt.spectral_consensus:
        method = "spectral_consensus"
    else:
        method = "spec_lap"

    all_species, all_triplets = get_triplets_from_file(opt.triplets)

    tree = gen_tree(all_triplets, method=method)

    if opt.viz:
        print(tree)
    else:
        print(tree.write(format=9))
