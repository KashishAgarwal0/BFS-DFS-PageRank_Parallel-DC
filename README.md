# BFS-DFS-PageRank_Parallel-DC
CUDA visualization of real time twitter dataset using PageRank, BFS, DFS
# CUDA-Optimized Graph Algorithms

## 📁 Data Format (CSR):
- Graph is stored in Compressed Sparse Row (CSR) format using two files:
  - row_ptr.npy — Row pointer array
  - col_ind.npy — Column index array

## 💻 How to Load the Graph in Python:
  import numpy as np
  row_ptr = np.load("row_ptr.npy")
  col_ind = np.load("col_ind.npy")

## 👥 Team Task Breakdown:
- bfs_gpu.cu → BFS implementation on GPU (Person 1)
- dfs_gpu.cu → DFS on GPU (Person 2)
- pagerank_gpu.cu → PageRank using power iteration (Person 3)

Each file should load the graph using CSR format and implement its algorithm on the GPU.

➡️ All outputs should print execution time (vs CPU if possible) and be wrapped in a main() so we can compile/run easily.
