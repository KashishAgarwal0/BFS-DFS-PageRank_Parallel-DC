
import numpy as np
from collections import deque
import time
import os

def bfs_csr(row_ptr, col_ind, source):
    V = len(row_ptr) - 1
    visited = [False] * V
    distance = [-1] * V

    queue = deque()
    queue.append(source)
    visited[source] = True
    distance[source] = 0

    while queue:
        u = queue.popleft()
        for i in range(row_ptr[u], row_ptr[u+1]):
            v = col_ind[i]
            if not visited[v]:
                visited[v] = True
                distance[v] = distance[u] + 1
                queue.append(v)

    return distance

if __name__ == "__main__":
  # Load the CSR graph
    row_ptr = np.load("Parallel_Project_WonderWorvicks/row_ptr.npy")
    col_ind = np.load("Parallel_Project_WonderWorvicks/col_ind.npy")

    
    source = 0
    
    start = time.time()
    dist = bfs_csr(row_ptr, col_ind, source)
    end = time.time()

    print("BFS Distance from node 0 (first 20 nodes):")
    for i, d in enumerate(dist[:20]):
        print(f"Node {i}: Distance = {d}")

    print(f"\nCPU BFS completed in {end - start:.4f} seconds")
