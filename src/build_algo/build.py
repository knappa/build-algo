#!/usr/bin/env python3
# the BUILD algorithm as described in Semple and Steel


def main_cli():
    import argparse

    from build_algo import PartitionMethods, gen_tree, get_triplets_from_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", required=True, help="triplets file")

    # parser.add_argument("--viz", action="store_true", help="print visualization")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--spec_lap", action="store_true", help="spectral laplacian method")
    group.add_argument(
        "--unweighted_spec_lap", action="store_true", help="unweighted spectral laplacian method"
    )
    group.add_argument("--agg_cluster", action="store_true", help="agglomerative clustering method")
    group.add_argument("--cograph_spectral", action="store_true", help="cograph spectral method")
    group.add_argument("--consensus", action="store_true", help="consensus method")
    group.add_argument("--collapsing", action="store_true", help="collapsing method")

    output_group = parser.add_mutually_exclusive_group(required=False)
    output_group.add_argument("--output", type=str, help="output file")
    # output_group.add_argument("--zoutput", type=str, help="compressed output file")

    opt = parser.parse_args()
    # print(opt)

    method: PartitionMethods
    if hasattr(opt, "spec_lap") and opt.spec_lap:
        method = "spec_lap"
    elif hasattr(opt, "unweighted_spec_lap") and opt.unweighted_spec_lap:
        method = "unweighted_spec_lap"
    elif hasattr(opt, "agg_cluster") and opt.agg_cluster:
        method = "agg_cluster"
    elif hasattr(opt, "cograph_spectral") and opt.cograph_spectral:
        method = "cograph_spectral"
    elif hasattr(opt, "consensus") and opt.consensus:
        method = "consensus"
    elif hasattr(opt, "collapsing") and opt.collapsing:
        method = "collapsing"
    else:
        method = "spec_lap"

    all_species, all_triplets = get_triplets_from_file(opt.triplets)

    tree = gen_tree(all_triplets, method=method)

    # if opt.viz:
    #     print(tree)
    # else:
    if hasattr(opt, "output") and opt.output:
        with open(opt.output, "wt") as f:
            f.write(tree.write(format=9))
    else:
        print(tree.write(format=9))
