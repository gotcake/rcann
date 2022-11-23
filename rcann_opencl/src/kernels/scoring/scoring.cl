__kernel void accum_multiclass_confusion_matrix(
        const uint ROWS,
        const uint N,
        const uint OUTPUT_ROW_STRIDE,
        const uint EXPECTED_ROW_STRIDE,
        const uint MATRIX_ROW_STRIDE,
        const __global float* output,
        const __global float* expected,
        __global uint* matrix
) {
    const uint row = get_global_id(0);

    if (row >= ROWS) {
        return;
    }

    const uint output_offset = row * OUTPUT_ROW_STRIDE;
    uint max_output_idx = 0;
    float max_output = output[output_offset];
    for (uint i = 1; i < N; i++) {
        if (output[output_offset + i] > max_output) {
            max_output = output[output_offset + i];
            max_output_idx = i;
        }
    }

    const uint expected_offset = row * EXPECTED_ROW_STRIDE;
    uint max_expected_idx = 0;
    float max_expected = expected[expected_offset];
    for (uint i = 1; i < N; i++) {
        if (expected[expected_offset + i] > max_expected) {
            max_expected = expected[expected_offset + i];
            max_expected_idx = i;
        }
    }

    atomic_inc(matrix[max_expected_idx * MATRIX_ROW_STRIDE + max_output_idx])

}