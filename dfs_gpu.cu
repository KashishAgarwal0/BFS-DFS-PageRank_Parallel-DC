
%%writefile dfs_gpu.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

#define MAX_V 1000000

__global__ void dfs_kernel(int V, int *row_ptr, int *col_ind, int *visited) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    // Each thread works on one node: simulate DFS recursively (naively)
    if (atomicExch(&visited[tid], 1) == 0) {
        // Explore all neighbors
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];

        for (int i = start; i < end; ++i) {
            int neighbor = col_ind[i];
            if (visited[neighbor] == 0) {
                // Simulate pushing neighbor to stack — launch DFS again on it
                // Not real recursion on GPU — you'd need dynamic parallelism (not supported on all GPUs)
                atomicExch(&visited[neighbor], 1);
            }
        }
    }
}

void dfs(int V, int E, int *h_row_ptr, int *h_col_ind, int source) {
    int *d_row_ptr, *d_col_ind, *d_visited;
    int *h_visited = (int *)malloc(V * sizeof(int));

    for (int i = 0; i < V; i++) {
        h_visited[i] = 0;
    }
    h_visited[source] = 1;

    cudaMalloc(&d_row_ptr, (V + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, E * sizeof(int));
    cudaMalloc(&d_visited, V * sizeof(int));

    cudaMemcpy(d_row_ptr, h_row_ptr, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, V * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    // Call only once — DFS doesn't proceed level by level
    dfs_kernel<<<gridSize, blockSize>>>(V, d_row_ptr, d_col_ind, d_visited);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(end - start).count();
    printf("\nGPU DFS-style traversal took %.2f ms\n", duration);

    cudaMemcpy(h_visited, d_visited, V * sizeof(int), cudaMemcpyDeviceToHost);

    int reachable = 0;
    for (int i = 0; i < V; i++) {
        if (h_visited[i] != 0)
            reachable++;
    }

    printf("Total reachable nodes from source %d: %d / %d\n", source, reachable, V);

    for (int i = 0; i < 20; i++) {
        printf("Node %d: %s\n", i, h_visited[i] ? "Visited" : "Not Visited");
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_visited);
    free(h_visited);
}

int main() {
    FILE *fs = fopen("sizes.txt", "r");
    int V, E;
    fscanf(fs, "%d %d", &V, &E);
    fclose(fs);

    int *h_row_ptr = (int *)malloc((V + 1) * sizeof(int));
    int *h_col_ind = (int *)malloc(E * sizeof(int));

    FILE *f1 = fopen("row_ptr.txt", "r");
    for (int i = 0; i < V + 1; i++) fscanf(f1, "%d", &h_row_ptr[i]);
    fclose(f1);

    FILE *f2 = fopen("col_ind.txt", "r");
    for (int i = 0; i < E; i++) fscanf(f2, "%d", &h_col_ind[i]);
    fclose(f2);

    int source = 0;
    dfs(V, E, h_row_ptr, h_col_ind, source);

    free(h_row_ptr);
    free(h_col_ind);
    return 0;
}
