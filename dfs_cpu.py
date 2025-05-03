#DFS_CPU
import numpy as np
import time

# Load CSR graph
row_ptr = np.load("Parallel_Project_WonderWorvicks/row_ptr.npy")
col_ind = np.load("Parallel_Project_WonderWorvicks/col_ind.npy")

def dfs_cpu_csr(row_ptr, col_ind, start_node=0):
    V = len(row_ptr) - 1
    visited = [False] * V
    stack = [start_node]

    start_time = time.time()

    while stack:
        u = stack.pop()
        if not visited[u]:
            visited[u] = True
            for i in range(row_ptr[u + 1] - 1, row_ptr[u] - 1, -1):
                v = col_ind[i]
                if not visited[v]:
                    stack.append(v)

    end_time = time.time()
    print(f"DFS (CPU) completed in {(end_time - start_time)*1000:.2f} ms")

dfs_cpu_csr(row_ptr, col_ind)

