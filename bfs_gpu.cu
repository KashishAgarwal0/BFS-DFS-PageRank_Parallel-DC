%%writefile bfs_gpu.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

#define INF 9999
#define MAX_V 1000000

__global__ void bfs_kernel(int V, int *row_ptr, int *col_ind, int *cost, int *visited, int level) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V || cost[u] != level) return;

    int start = row_ptr[u];
    int end = row_ptr[u + 1];

    for (int i = start; i < end; i++) {
        int v = col_ind[i];
        if (visited[v] == 0) {
            visited[v] = 1;
            cost[v] = level + 1;
        }
    }
}

void bfs(int V, int E, int *h_row_ptr, int *h_col_ind, int source) {
    int *d_row_ptr, *d_col_ind, *d_cost, *d_visited;
    int *h_cost = (int *)malloc(V * sizeof(int));
    int *h_visited = (int *)malloc(V * sizeof(int));

    for (int i = 0; i < V; i++) {
        h_cost[i] = INF;
        h_visited[i] = 0;
    }
    h_cost[source] = 0;
    h_visited[source] = 1;

    cudaMalloc(&d_row_ptr, (V + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, E * sizeof(int));
    cudaMalloc(&d_cost, V * sizeof(int));
    cudaMalloc(&d_visited, V * sizeof(int));

    cudaMemcpy(d_row_ptr, h_row_ptr, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, h_cost, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, V * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();

    for (int level = 0; level < V; level++) {
        bfs_kernel<<<gridSize, blockSize>>>(V, d_row_ptr, d_col_ind, d_cost, d_visited, level);
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(end - start).count();
    printf("\nGPU BFS took %.2f ms\n", duration);

    cudaMemcpy(h_cost, d_cost, V * sizeof(int), cudaMemcpyDeviceToHost);

    int reachable = 0;
    for (int i = 0; i < V; i++) {
        if (h_cost[i] != INF)
            reachable++;
    }

    printf("Total reachable nodes from source %d: %d / %d\n", source, reachable, V);

    // Optional: print first 20 distances
    for (int i = 0; i < 20; i++) {
        printf("Node %d: %d\n", i, h_cost[i] == INF ? -1 : h_cost[i]);
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_cost);
    cudaFree(d_visited);
    free(h_cost);
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

    int source = 0; // Try 0, 39564, etc.
    bfs(V, E, h_row_ptr, h_col_ind, source);

    free(h_row_ptr);
    free(h_col_ind);
    return 0;
}


