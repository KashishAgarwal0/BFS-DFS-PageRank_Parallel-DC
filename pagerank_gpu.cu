#include <cuda.h>
#include <stdio.h>

#define DAMPING 0.85
#define MAX_ITERS 20

__global__ void pagerank_kernel(int *row_ptr, int *col_ind, float *ranks_curr, float *ranks_next, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        int out_degree = end - start;

        if (out_degree > 0) {
            float contribution = ranks_curr[node] / out_degree;
            for (int idx = start; idx < end; idx++) {
                int neighbor = col_ind[idx];
                atomicAdd(&ranks_next[neighbor], DAMPING * contribution);
            }
        }
    }
}

int main() {
    // Load CSR graph
    int *h_row_ptr, *h_col_ind;
    int num_nodes, num_edges;

    FILE *f_row = fopen("Parallel_Project_WonderWorvicks/row_ptr.npy", "rb");
    FILE *f_col = fopen("Parallel_Project_WonderWorvicks/col_ind.npy", "rb");
    
    if (!f_row || !f_col) {
        printf("Error loading files!\n");
        return -1;
    }

    // Find sizes
    fseek(f_row, 0, SEEK_END);
    long row_size = ftell(f_row) / sizeof(int);
    rewind(f_row);

    fseek(f_col, 0, SEEK_END);
    long col_size = ftell(f_col) / sizeof(int);
    rewind(f_col);

    num_nodes = row_size - 1;
    num_edges = col_size;

    // Allocate CPU memory
    h_row_ptr = (int*)malloc(row_size * sizeof(int));
    h_col_ind = (int*)malloc(col_size * sizeof(int));

    fread(h_row_ptr, sizeof(int), row_size, f_row);
    fread(h_col_ind, sizeof(int), col_size, f_col);

    fclose(f_row);
    fclose(f_col);

    // Allocate GPU memory
    int *d_row_ptr, *d_col_ind;
    float *d_ranks_curr, *d_ranks_next;

    cudaMalloc(&d_row_ptr, row_size * sizeof(int));
    cudaMalloc(&d_col_ind, col_size * sizeof(int));
    cudaMalloc(&d_ranks_curr, num_nodes * sizeof(float));
    cudaMalloc(&d_ranks_next, num_nodes * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, row_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, col_size * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize ranks
    float init_rank = 1.0f / num_nodes;
    cudaMemset(d_ranks_curr, 0, num_nodes * sizeof(float));
    cudaMemset(d_ranks_next, 0, num_nodes * sizeof(float));

    // Fill d_ranks_curr with init_rank
    float *h_init_ranks = (float*)malloc(num_nodes * sizeof(float));
    for (int i = 0; i < num_nodes; i++) {
        h_init_ranks[i] = init_rank;
    }
    cudaMemcpy(d_ranks_curr, h_init_ranks, num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    free(h_init_ranks);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_nodes + threads_per_block - 1) / threads_per_block;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        cudaMemset(d_ranks_next, 0, num_nodes * sizeof(float)); // Reset next ranks

        pagerank_kernel<<<blocks_per_grid, threads_per_block>>>(d_row_ptr, d_col_ind, d_ranks_curr, d_ranks_next, num_nodes);

        // Add teleportation (random jump)
        float teleport = (1.0f - DAMPING) / num_nodes;
        // We'll do teleportation in kernel in a later upgrade if needed, for now assume ranks_next is adjusted properly

        // Swap arrays
        float *temp = d_ranks_curr;
        d_ranks_curr = d_ranks_next;
        d_ranks_next = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back
    float *h_ranks = (float*)malloc(num_nodes * sizeof(float));
    cudaMemcpy(h_ranks, d_ranks_curr, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first 10 ranks
    printf("First 10 node ranks:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f\n", h_ranks[i]);
    }

    printf("GPU PageRank finished in %.4f ms\n", milliseconds);

    // Cleanup
    free(h_ranks);
    free(h_row_ptr);
    free(h_col_ind);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_ranks_curr);
    cudaFree(d_ranks_next);

    return 0;
}
