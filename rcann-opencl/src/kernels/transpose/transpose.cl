__kernel void transpose(
        const uint ROWS, const uint COLS,
        const uint ROW_STRIDE_IN,
        const uint ROWS_OUT_BUFF, const uint COLS_OUT_BUFF,
        const __global real* input,
        __global real* output
) {
    // Thread identifiers
    const uint l_row = get_local_id(0);
    const uint l_col = get_local_id(1);
    uint g_row = get_global_id(0);
    uint g_col = get_global_id(1);

    // set up the local memory for shuffling
    __local real buffer[BLOCK_SIZE + 2][BLOCK_SIZE];

    // Save the block to a local buffer (coalesced)
    if (g_row < ROWS && g_col < COLS) {
        buffer[l_col][l_row] = input[g_row * ROW_STRIDE_IN + g_col];
    }/* else {
        // TODO: is this needed?
        buffer[l_col][l_row] = 0.0f;
    }*/

    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    g_row = get_group_id(1) * BLOCK_SIZE + l_row;
    g_col = get_group_id(0) * BLOCK_SIZE + l_col;

    // Store the transposed result (coalesced)
    if (g_row < COLS && g_col < ROWS) {
        // rows and columns are swapped
        output[g_row * COLS_OUT_BUFF + g_col] = buffer[l_row][l_col];
    } else /*if (g_row < ROWS_OUT_BUFF && g_col < COLS_OUT_BUFF)*/ {
        output[g_row * COLS_OUT_BUFF + g_col] = 0.0f;
    }
}