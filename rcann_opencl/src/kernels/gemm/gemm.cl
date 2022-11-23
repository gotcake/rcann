
__kernel void gemm(
        const uint M, const uint K, const uint N,
        const real ALPHA,
        const __global realX* A,
        const __global realX* B,
        const real BETA,
        __global realX* C
) {

    const uint l_row = get_local_id(0); // (0..TILE_SIZE]
    const uint l_col = get_local_id(1); // (0..TILE_SIZE / WIDTH]
    const uint g_row = get_global_id(0); // row of C (0..M]
    const uint g_col = TILE_SIZE / VECTOR_WIDTH * get_group_id(1) + l_col; // col of C (0..N]

    // Local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
    __local realX Asub[TILE_SIZE][TILE_SIZE / VECTOR_WIDTH];
    __local realX Bsub[TILE_SIZE][TILE_SIZE / VECTOR_WIDTH];

    // Initialise the accumulation registers
    realX acc = (realX)(0.0);

    const uint num_tiles = K / TILE_SIZE;
    for (uint t = 0; t < num_tiles; t++) {

        const uint t_row = TILE_SIZE * t + l_row;
        const uint t_col = TILE_SIZE / VECTOR_WIDTH * t + l_col;

        // Load one tile of A and B into local memory
        Asub[l_row][l_col] = A[g_row * K / VECTOR_WIDTH + t_col];
        Bsub[l_row][l_col] = B[t_row * N / VECTOR_WIDTH + g_col];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint k = 0; k < TILE_SIZE / VECTOR_WIDTH; k++) {
            const realX vecA = Asub[l_row][k];
            #pragma unroll
            for (uint w = 0; w < VECTOR_WIDTH; w++) {
                acc += VEC_IDX(vecA, w) * Bsub[k * VECTOR_WIDTH + w][l_col];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Store the final results in C
    const uint c_idx = g_row * N / VECTOR_WIDTH + g_col;
    C[c_idx] = ALPHA * acc + BETA * C[c_idx];
}
