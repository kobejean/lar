#include <metal_stdlib>
using namespace metal;

// Constants for matching
constant int K_MATCHES = 8;           // Fixed k=8 nearest neighbors
constant int DESCRIPTOR_DIM = 128;     // SIFT descriptor dimension

/**
 * @brief Brute-force k-NN matching kernel for uint8 descriptors - THREADGROUP OPTIMIZED VERSION.
 *
 * Optimizations:
 * 1. Threadgroup shared memory caching of train descriptor tiles
 * 2. Vectorized distance computation using float4 SIMD
 * 3. Manual loop unrolling (4x float4 operations per 16 elements)
 * 4. Selection-based top-K update (O(k) vs O(k²) insertion sort)
 * 5. Deferred sqrt operations (only compute at final write)
 * 6. Unrolled worst-distance tracking
 *
 * Strategy: Process train descriptors in tiles of 64. Load each tile cooperatively into
 * threadgroup shared memory, then all threads compute distances against the tile.
 * This amortizes memory bandwidth across the threadgroup.
 *
 * @param queryDesc Query descriptors [numQuery x descriptorDim] (uint8)
 * @param trainDesc Train descriptors [numTrain x descriptorDim] (uint8)
 * @param outIndices Output train indices [numQuery x K]
 * @param outDistances Output squared distances [numQuery x K]
 * @param numQuery Number of query descriptors
 * @param numTrain Number of train descriptors
 * @param descriptorDim Descriptor dimension (should be 128)
 * @param tid Thread index in grid
 * @param localId Thread index in threadgroup
 */
kernel void bf_match_knn_uint8(
    device const uchar* queryDesc          [[buffer(0)]],
    device const uchar* trainDesc          [[buffer(1)]],
    device int* outIndices                 [[buffer(2)]],
    device float* outDistances             [[buffer(3)]],
    constant int& numQuery                 [[buffer(4)]],
    constant int& numTrain                 [[buffer(5)]],
    constant int& descriptorDim            [[buffer(6)]],
    uint tid                               [[thread_position_in_grid]],
    uint localId                           [[thread_position_in_threadgroup]]
) {
    // Threadgroup shared memory for train descriptor tile
    // 64 descriptors × 128 bytes = 8KB (fits comfortably in 32KB threadgroup memory)
    constexpr int TILE_SIZE = 64;
    threadgroup uchar sharedTrain[TILE_SIZE][DESCRIPTOR_DIM];

    // Boundary check
    if (tid >= numQuery) return;

    // Thread-local top-K storage (kept unsorted for faster updates)
    float topK_distances[K_MATCHES];
    int topK_indices[K_MATCHES];

    // Track the worst (maximum) distance in our top-K
    float worstDist = FLT_MAX;
    int worstIdx = 0;

    // Initialize with worst possible values
    #pragma unroll
    for (int i = 0; i < K_MATCHES; i++) {
        topK_distances[i] = FLT_MAX;
        topK_indices[i] = -1;
    }

    // Cache query descriptor pointer (reused across all tiles)
    device const uchar* query = queryDesc + tid * descriptorDim;

    // Process train descriptors in tiles
    for (int tileStart = 0; tileStart < numTrain; tileStart += TILE_SIZE) {
        int tileEnd = min(tileStart + TILE_SIZE, numTrain);
        int tileCount = tileEnd - tileStart;

        // Cooperatively load train tile into shared memory
        // Each thread loads one or more descriptors
        for (int i = localId; i < tileCount; i += 256) {  // Assume threadgroup size 256
            device const uchar* trainSrc = trainDesc + (tileStart + i) * descriptorDim;
            for (int j = 0; j < DESCRIPTOR_DIM; j++) {
                sharedTrain[i][j] = trainSrc[j];
            }
        }

        // Wait for all threads to finish loading
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute distances against all descriptors in this tile
        for (int i = 0; i < tileCount; i++) {
            // Compute L2 squared distance with aggressive unrolling for 128D
            float sum = 0.0f;

            #pragma unroll
            for (int j = 0; j < DESCRIPTOR_DIM; j += 16) {
                // Load and compute 4 elements at a time using SIMD
                float4 q0 = float4(query[j], query[j+1], query[j+2], query[j+3]);
                float4 t0 = float4(sharedTrain[i][j], sharedTrain[i][j+1], sharedTrain[i][j+2], sharedTrain[i][j+3]);
                float4 d0 = q0 - t0;
                sum += dot(d0, d0);

                float4 q1 = float4(query[j+4], query[j+5], query[j+6], query[j+7]);
                float4 t1 = float4(sharedTrain[i][j+4], sharedTrain[i][j+5], sharedTrain[i][j+6], sharedTrain[i][j+7]);
                float4 d1 = q1 - t1;
                sum += dot(d1, d1);

                float4 q2 = float4(query[j+8], query[j+9], query[j+10], query[j+11]);
                float4 t2 = float4(sharedTrain[i][j+8], sharedTrain[i][j+9], sharedTrain[i][j+10], sharedTrain[i][j+11]);
                float4 d2 = q2 - t2;
                sum += dot(d2, d2);

                float4 q3 = float4(query[j+12], query[j+13], query[j+14], query[j+15]);
                float4 t3 = float4(sharedTrain[i][j+12], sharedTrain[i][j+13], sharedTrain[i][j+14], sharedTrain[i][j+15]);
                float4 d3 = q3 - t3;
                sum += dot(d3, d3);
            }

            float dist = sum;
            int trainIdx = tileStart + i;

            // Only update if this distance is better than the worst in our top-K
            if (dist < worstDist) {
                // Replace the worst entry
                topK_distances[worstIdx] = dist;
                topK_indices[worstIdx] = trainIdx;

                // Find new worst entry with unrolled comparison
                worstDist = topK_distances[0];
                worstIdx = 0;

                if (topK_distances[1] > worstDist) { worstDist = topK_distances[1]; worstIdx = 1; }
                if (topK_distances[2] > worstDist) { worstDist = topK_distances[2]; worstIdx = 2; }
                if (topK_distances[3] > worstDist) { worstDist = topK_distances[3]; worstIdx = 3; }
                if (topK_distances[4] > worstDist) { worstDist = topK_distances[4]; worstIdx = 4; }
                if (topK_distances[5] > worstDist) { worstDist = topK_distances[5]; worstIdx = 5; }
                if (topK_distances[6] > worstDist) { worstDist = topK_distances[6]; worstIdx = 6; }
                if (topK_distances[7] > worstDist) { worstDist = topK_distances[7]; worstIdx = 7; }
            }
        }

        // Wait before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Sort the top-K results before writing out (simple insertion sort for k=8)
    for (int i = 1; i < K_MATCHES; i++) {
        float keyDist = topK_distances[i];
        int keyIdx = topK_indices[i];
        int j = i - 1;

        while (j >= 0 && topK_distances[j] > keyDist) {
            topK_distances[j + 1] = topK_distances[j];
            topK_indices[j + 1] = topK_indices[j];
            j--;
        }
        topK_distances[j + 1] = keyDist;
        topK_indices[j + 1] = keyIdx;
    }

    // Write results to global memory (only compute sqrt at the very end)
    int outOffset = tid * K_MATCHES;
    #pragma unroll
    for (int i = 0; i < K_MATCHES; i++) {
        outIndices[outOffset + i] = topK_indices[i];
        // Take sqrt to get actual L2 distance (OpenCV convention)
        outDistances[outOffset + i] = sqrt(topK_distances[i]);
    }
}
