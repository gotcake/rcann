__kernel void fill_pad_zero(
        const int ROWS, const int COLS,
        const int BUFF_ROWS, const int BUFF_COLS,
        __global float* matrix
) {
    const int n = get_global_id(0);
    if (n < BUFF_ROWS) {
        const int row_offset = n * BUFF_COLS;
        for (int k = COLS; k < BUFF_COLS; k++) {
            matrix[row_offset + k] = 0.0f;
        }
    }
    if (n < BUFF_COLS) {
        for (int k = ROWS; k < BUFF_ROWS; k++) {
            matrix[k * BUFF_COLS + n] = 0.0f;
        }
    }
}

__kernel void naive_sgemm(
        const int M, const int K, const int N,
        const float ALPHA,
        const __global float* A,
        const __global float* B,
        const float BETA,
        __global float* C
) {

    const int g_row = get_global_id(0); // row of C (0..M)
    const int g_col = get_global_id(1); // col of C (0..N)

    const int g_rowK = g_row * K;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        acc += A[g_rowK + k] * B[k * N + g_col];
    }

    const int c_idx = g_row * N + g_col;
    C[c_idx] = ALPHA * acc + BETA * C[c_idx];
}