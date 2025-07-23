library(reticulate)

# install the python library from github or a local directory
py_install("build_algo@git+https://github.com/knappa/build-algo.git", pip_options="-e")
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
  
build.algo$gen_tree_from_triplet_file("/home/knappa/build-algo/test_data/test-triplets-1.txt")
