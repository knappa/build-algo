library(reticulate)

# install the python library from github or a local directory
py_install(
  "build_algo@git+https://github.com/knappa/build-algo.git",
  method = c("virtualenv"),
  pip_ignore_installed=TRUE,
)
# py_install("/home/knappa/build-algo/", pip_options="-e")

# this may be redundant?
py_require("build_algo")


build.algo <- import("build_algo")

# the gen_tree_from_* methods return a newick string representation of the tree

build.algo$gen_tree_from_string("a,b|c
  a,b|d
  a,b|e
  a,b|f
  c,d|a
  c,e|a
  c,f|a
  d,e|a
  d,f|a
  e,f|a
  c,d|b
  c,e|b
  c,f|b
  d,e|b
  d,f|b
  e,f|b
  c,d|e
  c,d|f
  c,e|f
  d,e|f")
  
# default algorithm is the spectral laplacian
build.algo$gen_tree_from_triplet_file("/home/knappa/build-algo/test_data/test-triplets-1.txt")

# spectral laplacian method
build.algo$gen_tree_from_triplet_file("/home/knappa/build-algo/test_data/test-triplets-1.txt", method="spec_lap")

# agglomerative clustering in a spectral embedding
build.algo$gen_tree_from_triplet_file("/home/knappa/build-algo/test_data/test-triplets-1.txt", method="agg_cluster")

# Spectral method on the cograph.
build.algo$gen_tree_from_triplet_file("/home/knappa/build-algo/test_data/test-triplets-1.txt", method="cograph_spectral")

# Method which accepts consensus between spectral laplacian and cograph methods; does brute force search on disagreement
build.algo$gen_tree_from_triplet_file("/home/knappa/build-algo/test_data/test-triplets-1.txt", method="spectral_consensus")
