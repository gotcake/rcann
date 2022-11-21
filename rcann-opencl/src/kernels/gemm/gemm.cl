#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))

#if PRECISION == 64
    typedef real double;
#elif PRECISION == 32
    typedef real float;
#endif

// Data-widths
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#endif

__kernel void sgemm3(
        const uint M, const uint K, const uint N,
        const float ALPHA,
        const __global float* A,
        const __global float* B,
        const float BETA,
        __global float* C
) {

    const uint l_row = get_local_id(0); // (0..TILE_SIZE]
    const uint l_col = get_local_id(1); // (0..TILE_SIZE]
    const uint g_row = get_global_id(0); // row of C (0..M]
    const uint g_col = TILE_SIZE * get_group_id(1) + l_col; // col of C (0..N]

    // Local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    // Initialise the accumulation registers
    float acc[WORK_PER_THREAD];
    #pragma unroll
    for (uint w = 0; w < WORK_PER_THREAD; w++) {
        acc[w] = 0.0f;
    }

    const uint num_tiles = K / TILE_SIZE;
    for (uint t = 0; t < num_tiles; t++) {

        const uint t_row = TILE_SIZE * t + l_row;
        const uint t_col = TILE_SIZE * t + l_col;

        // Load one tile of A and B into local memory
        #pragma unroll
        for (uint w = 0; w < WORK_PER_THREAD; w++) {
            const uint w_offset = w * REDUCED_TILE_SIZE;
            Asub[l_row][l_col + w_offset] = A[g_row * K + (t_col + w_offset)];
            Bsub[l_row][l_col + w_offset] = B[t_row * N + (g_col + w_offset)];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (uint w = 0; w < WORK_PER_THREAD; w++) {
                acc[w] += Asub[l_row][k] * Bsub[k][l_col + w * REDUCED_TILE_SIZE];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Store the final results in C
    const uint c_offset = g_row * N + g_col;
    if (BETA == 0.0f) {
        #pragma unroll
        for (uint w = 0; w < WORK_PER_THREAD; w++) {
            const uint c_idx = c_offset + w * REDUCED_TILE_SIZE;
            C[c_idx] = ALPHA * acc[w];
        }
    } else {
        #pragma unroll
        for (uint w = 0; w < WORK_PER_THREAD; w++) {
            const uint c_idx = c_offset + w * REDUCED_TILE_SIZE;
            C[c_idx] = ALPHA * acc[w] + BETA * C[c_idx];
        }
    }
}
/*
__kernel void sgemm7(
        const int M, const int N, const int K,
        const float ALPHA,
        const __global floatX* A,
        const __global floatX* B,
        const float BETA,
        __global float* C
) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSK][TSN];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA/WIDTH; la++) {
            const int tid = tidn*RTSM + tidm;
            const int id = la*RTSN*RTSM + tid;
            const int row = MOD2(id,TSM/WIDTH);
            const int col = DIV2(id,TSM/WIDTH);

            // Load the values (wide vector load)
            const int tiledIndex = TSK*t + col;
            floatX vecA = A[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
            floatX vecB = B[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];

            // Store the loaded vectors into local memory
            #if WIDTH == 1
                Asub[col][row] = vecA;
            #elif WIDTH == 2
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
            #elif WIDTH == 4
                Asub[col][WIDTH*row + 0] = vecA.x;
                Asub[col][WIDTH*row + 1] = vecA.y;
                Asub[col][WIDTH*row + 2] = vecA.z;
                Asub[col][WIDTH*row + 3] = vecA.w;
            #endif
            #if WIDTH == 1
                Bsub[col][row] = vecB;
            #elif WIDTH == 2
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
            #elif WIDTH == 4
                Bsub[col][WIDTH*row + 0] = vecB.x;
                Bsub[col][WIDTH*row + 1] = vecB.y;
                Bsub[col][WIDTH*row + 2] = vecB.z;
                Bsub[col][WIDTH*row + 3] = vecB.w;
            #endif
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        #pragma unroll
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                const int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[k][col];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                const int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        const int globalRowN = (offsetM + tidm + wm*RTSM) * N;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            const int globalCol = offsetN + tidn + wn*RTSN;
            const int c_idx = globalRowN + globalCol;
            C[c_idx] = ALPHA * acc[wm][wn] + BETA * C[c_idx];
        }
    }
}*/