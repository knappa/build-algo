#!/usr/bin/env bash

../../gen_triplet_weights.py --phylip data_"$1"_GTR_I_Gamma_1K_sites_ultrametric.phy --output triplets_"$1".txt --method J || exit
../../weighted_build_algorithm.py --triplets triplets_"$1".txt --out reconstructed_tree_l_"$1".nwk --method L || exit
../../weighted_build_algorithm.py --triplets triplets_"$1".txt --out reconstructed_tree_p_"$1".nwk --method P || exit
../../weighted_build_algorithm.py --triplets triplets_"$1".txt --out reconstructed_tree_mcl_"$1".nwk --method MCL
rm triplets_"$1".txt
