// internal ranking implementation
template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT,
          typename CountsCallback>
struct BlockRadixRankMatchInternal
{
    TempStorage& s;
    DigitExtractorT digit_extractor;
    CountsCallback callback;
    int warp;
    int lane;

    __device__ __forceinline__ std::uint32_t Digit(UnsignedBits key)
    {
        std::uint32_t digit =  digit_extractor.Digit(key);
        return IS_DESCENDING ? RADIX_DIGITS - 1 - digit : digit;
    }

    __device__ __forceinline__ int ThreadBin(int u)
    {
        int bin = threadIdx.x * BINS_PER_THREAD + u;
        return IS_DESCENDING ? RADIX_DIGITS - 1 - bin : bin;
    }

    __device__ __forceinline__
    void ComputeHistogramsWarp(UnsignedBits (&keys)[KEYS_PER_THREAD])
    {
        //int* warp_offsets = &s.warp_offsets[warp][0];
        int (&warp_histograms)[RADIX_DIGITS][NUM_PARTS] = s.warp_histograms[warp];
        // compute warp-private histograms
        #pragma unroll
        for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
        {
            #pragma unroll
            for (int part = 0; part < NUM_PARTS; ++part)
            {
                warp_histograms[bin][part] = 0;
            }
        }
        if (MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR)
        {
            int* match_masks = &s.match_masks[warp][0];
            #pragma unroll
            for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
            {
                match_masks[bin] = 0;
            }
        }
        WARP_SYNC(WARP_MASK);

        // compute private per-part histograms
        int part = lane % NUM_PARTS;
        #pragma unroll
        for (int u = 0; u < KEYS_PER_THREAD; ++u)
        {
            atomicAdd(&warp_histograms[Digit(keys[u])][part], 1);
        }

        // sum different parts;
        // no extra work is necessary if NUM_PARTS == 1
        if (NUM_PARTS > 1)
        {
            WARP_SYNC(WARP_MASK);
            // TODO: handle RADIX_DIGITS % WARP_THREADS != 0 if it becomes necessary
            const int WARP_BINS_PER_THREAD = RADIX_DIGITS / WARP_THREADS;
            int bins[WARP_BINS_PER_THREAD];
            #pragma unroll
            for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
            {
                int bin = lane + u * WARP_THREADS;
                bins[u] = internal::ThreadReduce(warp_histograms[bin], Sum());
            }
            CTA_SYNC();

            // store the resulting histogram in shared memory
            int* warp_offsets = &s.warp_offsets[warp][0];
            #pragma unroll
            for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
            {
                int bin = lane + u * WARP_THREADS;
                warp_offsets[bin] = bins[u];
            }
        }
    }

    __device__ __forceinline__
    void ComputeOffsetsWarpUpsweep(int (&bins)[BINS_PER_THREAD])
    {
        // sum up warp-private histograms
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            bins[u] = 0;
            int bin = ThreadBin(u);
            if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
            {
                #pragma unroll
                for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                {
                    int warp_offset = s.warp_offsets[j_warp][bin];
                    s.warp_offsets[j_warp][bin] = bins[u];
                    bins[u] += warp_offset;
                }
            }
        }
    }

    __device__ __forceinline__
    void ComputeOffsetsWarpDownsweep(int (&offsets)[BINS_PER_THREAD])
    {
        #pragma unroll
        for (int u = 0; u < BINS_PER_THREAD; ++u)
        {
            int bin = ThreadBin(u);
            if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
            {
                int digit_offset = offsets[u];
                #pragma unroll
                for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                {
                    s.warp_offsets[j_warp][bin] += digit_offset;
                }
            }
        }
    }

    __device__ __forceinline__
    void ComputeRanksItem(
        UnsignedBits (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
        Int2Type<WARP_MATCH_ATOMIC_OR>)
    {
        // compute key ranks
        int lane_mask = 1 << lane;
        int* warp_offsets = &s.warp_offsets[warp][0];
        int* match_masks = &s.match_masks[warp][0];
        #pragma unroll
        for (int u = 0; u < KEYS_PER_THREAD; ++u)
        {
            std::uint32_t bin = Digit(keys[u]);
            int* p_match_mask = &match_masks[bin];
            atomicOr(p_match_mask, lane_mask);
            WARP_SYNC(WARP_MASK);
            int bin_mask = *p_match_mask;
            int leader = (WARP_THREADS - 1) - __clz(bin_mask);
            int warp_offset = 0;
            int popc = __popc(bin_mask & LaneMaskLe());
            if (lane == leader)
            {
                // atomic is a bit faster
                warp_offset = atomicAdd(&warp_offsets[bin], popc);
            }
            warp_offset = SHFL_IDX_SYNC(warp_offset, leader, WARP_MASK);
            if (lane == leader) *p_match_mask = 0;
            WARP_SYNC(WARP_MASK);
            ranks[u] = warp_offset + popc - 1;
        }
    }

    __device__ __forceinline__
    void ComputeRanksItem(
        UnsignedBits (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
        Int2Type<WARP_MATCH_ANY>)
    {
        // compute key ranks
        int* warp_offsets = &s.warp_offsets[warp][0];
        #pragma unroll
        for (int u = 0; u < KEYS_PER_THREAD; ++u)
        {
            std::uint32_t bin = Digit(keys[u]);
            int bin_mask = detail::warp_in_block_matcher_t<RADIX_BITS,
                                                           PARTIAL_WARP_THREADS,
                                                           BLOCK_WARPS - 1>::match_any(bin,
                                                                                       warp);
            int leader = (WARP_THREADS - 1) - __clz(bin_mask);
            int warp_offset = 0;
            int popc = __popc(bin_mask & LaneMaskLe());
            if (lane == leader)
            {
                // atomic is a bit faster
                warp_offset = atomicAdd(&warp_offsets[bin], popc);
            }
            warp_offset = SHFL_IDX_SYNC(warp_offset, leader, WARP_MASK);
            ranks[u] = warp_offset + popc - 1;
        }
    }

    __device__ __forceinline__ void RankKeys(
        UnsignedBits (&keys)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD],
        int (&exclusive_digit_prefix)[BINS_PER_THREAD])
    {
        ComputeHistogramsWarp(keys);

        CTA_SYNC();
        int bins[BINS_PER_THREAD];
        ComputeOffsetsWarpUpsweep(bins);
        callback(bins);

        BlockScan(s.prefix_tmp).ExclusiveSum(bins, exclusive_digit_prefix);

        ComputeOffsetsWarpDownsweep(exclusive_digit_prefix);
        CTA_SYNC();
        ComputeRanksItem(keys, ranks, Int2Type<MATCH_ALGORITHM>());
    }

    __device__ __forceinline__ BlockRadixRankMatchInternal
    (TempStorage& temp_storage, DigitExtractorT digit_extractor, CountsCallback callback)
        : s(temp_storage), digit_extractor(digit_extractor),
          callback(callback), warp(threadIdx.x / WARP_THREADS), lane(LaneId())
        {}
};
