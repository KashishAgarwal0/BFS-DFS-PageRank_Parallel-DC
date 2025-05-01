import numpy as np
import time

# Load the CSR graph
row_ptr = np.load("Parallel_Project_WonderWorvicks/row_ptr.npy")
col_ind = np.load("Parallel_Project_WonderWorvicks/col_ind.npy")
num_nodes = row_ptr.shape[0] - 1  # row_ptr size is (N+1)

# PageRank parameters
damping = 0.85
num_iters = 20  # or 100 if you want
epsilon = 1e-6  # convergence threshold (not used here since we fix iterations)

# Initialize ranks
ranks = np.full(num_nodes, 1.0 / num_nodes)

# Compute outdegrees (needed for division)
out_degree = row_ptr[1:] - row_ptr[:-1]

print(f"Running PageRank CPU for {num_iters} iterations...")

start_time = time.time()

for iteration in range(num_iters):
    new_ranks = np.full(num_nodes, (1 - damping) / num_nodes)  # teleportation

    for node in range(num_nodes):
        start = row_ptr[node]
        end = row_ptr[node+1]
        
        for idx in range(start, end):
            neighbor = col_ind[idx]
            if out_degree[node] > 0:
                new_ranks[neighbor] += damping * (ranks[node] / out_degree[node])
                
    ranks = new_ranks

end_time = time.time()

print(f"CPU PageRank finished in {end_time - start_time:.4f} seconds 🚀")
print(f"First 10 node ranks: {ranks[:10]}")
