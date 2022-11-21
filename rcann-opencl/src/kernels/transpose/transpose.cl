__kernel void transpose(
        const uint ROWS_IN, const uint COLS_IN,
        const uint ROW_STRIDE_IN,
        const uint ROWS_OUT, const uint COLS_OUT,
        const uint ROW_STRIDE_OUT,
        const __global float* input,
        __global float* output
) {
    // Thread identifiers
    const uint l_row = get_local_id(0);
    const uint l_col = get_local_id(1);
    uint g_row = get_global_id(0);
    uint g_col = get_global_id(1);

    // set up the local memory for shuffling
    __local float buffer[BLOCK_SIZE + 2][BLOCK_SIZE];

    // Save the block to a local buffer (coalesced)
    if (g_row < ROWS_IN && g_col < COLS_IN) {
        buffer[l_col][l_row] = input[g_row * ROW_STRIDE_IN + g_col];
    } else {
        buffer[l_col][l_row] = 0.0f;
    }

    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    g_row = get_group_id(1) * BLOCK_SIZE + l_row;
    g_col = get_group_id(0) * BLOCK_SIZE + l_col;

    // Store the transposed result (coalesced)
    if (g_row < ROWS_OUT && g_col < COLS_OUT) {
        // rows and columns are swapped
        output[g_row * ROW_STRIDE_OUT + g_col] = buffer[l_row][l_col];
    }
}