__kernel void zero_padding(
        const uint ROWS, const uint COLS,
        const uint BUFF_ROWS, const uint BUFF_COLS,
        __global float* matrix
) {
    const uint n = get_global_id(0);
    if (n < BUFF_ROWS) {
        const uint row_offset = n * BUFF_COLS;
        for (uint k = COLS; k < BUFF_COLS; k++) {
            matrix[row_offset + k] = 0.0f;
        }
    }
    if (n < BUFF_COLS) {
        for (uint k = ROWS; k < BUFF_ROWS; k++) {
            matrix[k * BUFF_COLS + n] = 0.0f;
        }
    }
}