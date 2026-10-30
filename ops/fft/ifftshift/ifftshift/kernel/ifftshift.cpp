/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CATLASS_ARCH 3510

#include "kernel_operator.h"

using namespace AscendC;

namespace IfftShiftNs {

constexpr int64_t COMPLEX_BYTES = 8;

// Fast-path ring depth. A 3-slot build measured identical on the MRI shape
// (the residual MTE2 idle is DRAM read/write turnaround, not pipeline depth).
constexpr int64_t SLOTS = 2;

// Generic-path staging: complex elements per ping-pong buffer (32KB each).
constexpr int64_t TILE_COMPLEX = 4096;
constexpr int64_t COMPLEX_PER_BLOCK = 4;
constexpr int64_t PING_PONG = 2;

// One full vector register of floats (VECTOR_REG_WIDTH / sizeof(float)).
constexpr uint32_t VEC_FLOATS = 256 / sizeof(float);

// Must keep the same layout as AsdSip::IfftShiftTilingData in
// ops/fft/ifftshift/ifftshift/tiling/tiling_data.h
struct IfftShiftTilingData {
    int64_t row;
    int64_t col;
    int64_t batch;
    int64_t dist;
    int64_t totalRows;
    int64_t ubSize;
    float scale;
};

/*
 * IfftShift, aligned with the reference CUDA ifftShiftScaleBatch2DKernel:
 *   dst[s][(r + row / 2) % row][(c + col / 2) % col] = scale * src[s][r][c]
 * Input slices are `dist` complex elements apart, output slices are contiguous.
 * This is the inverse of FftShift (shift by floor(n/2) instead of ceil(n/2)):
 * both coincide for even dimensions and differ by one for odd dimensions.
 * `scale` folds an element-wise multiplier into the shift pass (e.g. the
 * inverse-FFT normalization 1 / (row * col)); scale == 1 keeps the pure
 * permutation bit-exact.
 *
 * One kernel, two runtime paths (dispatched on whether a whole row fits the
 * per-slot UB budget, i.e. col <= ubBudget / (SLOTS*2*8) ~ 7800):
 *
 * FAST PATH (all common shapes): 3-stage pipeline
 *   whole-row MTE2 read -> vector register repack -> whole-row MTE3 write.
 *   - MTE2 reads each tile as ONE contiguous block (rows of a tile are
 *     consecutive in GM) into Area A, rows packed at col*8 bytes.
 *   - The vector stage assembles each OUTPUT ROW contiguously in RowArea:
 *       piece A: dst[0..shiftC-1]   <- src[col-shiftC .. col-1]
 *       piece B: dst[shiftC..col-1] <- src[0 .. col-shiftC-1]
 *     where shiftC = col / 2 (for even col the two pieces are equal halves;
 *     odd col and col == 1 also work). When scale != 1 each loaded chunk is
 *     scaled in-register by MicroAPI::Muls (element-wise float32 multiply
 *     with an all-lane mask; the tail chunk's undefined lanes are multiplied
 *     too but never stored) before it is stored.
 *   - MTE3 writes each tile as ONE contiguous block: full cache lines, no
 *     write-allocate RFO, no 64B lines split across descriptors.
 *   - Synchronization: on the 950 (3510) a wait_flag(src, dst) marker blocks
 *     the DST pipe queue, NOT the issuing scalar stream, and PipeBarrier only
 *     orders within one queue. Cross-pipe dependencies therefore use explicit
 *     HardEvent channels with the wait issued on the consumer pipe:
 *       MTE2_V (fill, ids 0-1): A[s] filled      -> V may read
 *       MTE3_V (free, ids 2-3): B[s] written out -> V may overwrite
 *       V_MTE3 (v3,  id 4): V repack done        -> MTE3 may read B[s]
 *       V_MTE2 (v2,  id 5): V repack done        -> MTE2 may reuse A[s]
 *     v3/v2 are strictly back-to-back set-then-wait handshakes and share one
 *     ID per channel: for v2 the MTE2-queue order makes set(t+1) provably
 *     follow wait(t); the same FIFO event semantics covers v3.
 *
 * GENERIC PATH (very wide columns): the former piecewise ping-pong kernel,
 * kept verbatim as the correctness fallback for columns that cannot stage a
 * whole row per slot. When scale != 1 a vector multiply stage is inserted
 * between copy-in and copy-out (MTE2 -> V -> MTE3 with MTE2_V / V_MTE3
 * events); with scale == 1 the direct MTE2 -> MTE3 handoff is unchanged.
 *
 * Both paths partition the output rows over ALL mix contexts: on the 950 a
 * mix launch exposes GetBlockIdx() in [0, blockNum*subBlockNum), where
 * physical block = idx/2 and vector sub-core = idx%2 (GetSubBlockIdx() itself
 * always reads 0 there and must not be used). A "blockIdx < GetBlockNum()"
 * guard would leave half of the vector cores idle.
 *
 * The fast pipe is a template on COL: COL > 0 bakes the piece lengths and row
 * pitch into compile-time constants, which lets the compiler interleave the
 * per-row vldas/vldus/vstus chains (the runtime-count form is ~6x slower on
 * the V pipe: latency-bound without cross-row ILP). COL == 0 takes the
 * lengths from the tiling at runtime and serves every other feasible column.
 */
template <int64_t COL>
class IfftShiftFastPipe {
public:
    // Rows per tile for the given column; 0 means the fast path is not viable.
    static __aicore__ inline int64_t ComputeRmax(int64_t col, int64_t ubSize)
    {
        constexpr int64_t TPIPE_RESERVE = 4 * 1024;
        int64_t ubBudget = ubSize - TPIPE_RESERVE;
        if (ubBudget < 200 * 1024) {
            ubBudget = 200 * 1024;
        }
        int64_t rmax = ubBudget / (SLOTS * 2 * col * COMPLEX_BYTES);
        if (rmax > 4095) {
            rmax = 4095;
        }
        // Keep every MTE2 dst / MTE3 src base 32B aligned: the per-slot layout
        // is [Area A: rmax*col | RowArea: rmax*col] complex elements, so
        // rmax*col*8 must stay a multiple of 32, i.e. rmax*col a multiple of
        // 4 complex elements (at most 3 decrements).
        while (rmax > 0 && (rmax * col) % 4 != 0) {
            rmax--;
        }
        return rmax;
    }

    __aicore__ inline void Init(GlobalTensor<float>& src, GlobalTensor<float>& dst, TPipe& pipe,
                                TBuf<TPosition::VECCALC>& ubBuf, int64_t row, int64_t col, int64_t dist, int64_t ubSize,
                                float scale)
    {
        srcGM_ = src;
        dstGM_ = dst;
        row_ = row;
        col_ = col;
        dist_ = dist;
        // Inverse shift: floor(n/2) instead of FftShift's ceil(n/2).
        shiftRow_ = row_ / 2;
        shiftCol_ = col_ / 2;
        lenA_ = shiftCol_;
        lenB_ = col_ - shiftCol_;
        scale_ = scale;
        needScale_ = (scale_ != 1.0f);
        rmax_ = ComputeRmax(col, ubSize);

        pipe.InitBuffer(ubBuf, SLOTS * rmax_ * 2 * col_ * COMPLEX_BYTES);
        ubBuf_ = &ubBuf;

        // Event IDs: distinct range per channel (QUE_MAX_EVENT = 8 on 3510).
        // fill/free are buffered states (prologue/init set several before the
        // first wait), so they need one ID per slot.
        for (int64_t s = 0; s < SLOTS; s++) {
            evFill_[s] = static_cast<event_t>(s);           // MTE2_V 0..SLOTS-1
            evFree_[s] = static_cast<event_t>(s + SLOTS);   // MTE3_V SLOTS..2*SLOTS-1
            evV3_[s] = static_cast<event_t>(2 * SLOTS);     // V_MTE3 shared
            evV2_[s] = static_cast<event_t>(2 * SLOTS + 1); // V_MTE2 shared
        }
        // All slots' RowAreas are initially free for the V stage.
        for (int64_t s = 0; s < SLOTS; s++) {
            SetFlag<HardEvent::MTE3_V>(evFree_[s]);
        }
    }

    // reg *= scale (element-wise float32). The all-lane mask matches the
    // framework idiom (Reg::Muls in the 3510 activation impls); the tail
    // chunk's undefined lanes are multiplied too but never stored.
    static __aicore__ inline void ScaleReg(MicroAPI::RegTensor<float>& reg, float scale)
    {
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<float>();
        MicroAPI::Muls(reg, reg, scale, mask);
    }

    __aicore__ inline void Process(int64_t rowStart, int64_t rowCount)
    {
        tileRow_ = rowStart;
        rowEnd_ = rowStart + rowCount;

        TileDesc ring[SLOTS];
        int64_t total = 0;

        // Prologue: MTE2 fills both slots' Area A.
        for (int64_t i = 0; i < SLOTS; i++) {
            if (!NextTile(ring[i])) {
                break;
            }
            CopyInWholeRows(ring[i], static_cast<int>(i));
            total++;
        }

        // Steady state: V(t) is serialized by events on the consumer pipes;
        // MTE3(t) and MTE2(t+2) run async and overlap with V(t+1).
        for (int64_t t = 0; t < total; t++) {
            const int s = static_cast<int>(t % SLOTS);

            // V stage waits: A[s] filled by MTE2, RowArea[s] released by MTE3(t-2).
            WaitFlag<HardEvent::MTE2_V>(evFill_[s]);
            WaitFlag<HardEvent::MTE3_V>(evFree_[s]);

            // V repack: A[s] (full rows) -> RowArea[s] (contiguous output rows).
            VectorRepack(ring[s], s);

            // MTE3 may read RowArea[s] only after the V writes complete.
            SetFlag<HardEvent::V_MTE3>(evV3_[s]);
            WaitFlag<HardEvent::V_MTE3>(evV3_[s]);
            CopyOutRows(ring[s], s);

            // MTE2 may overwrite A[s] only after the V reads complete.
            TileDesc nt;
            if (NextTile(nt)) {
                SetFlag<HardEvent::V_MTE2>(evV2_[s]);
                WaitFlag<HardEvent::V_MTE2>(evV2_[s]);
                ring[s] = nt;
                CopyInWholeRows(ring[s], s);
                total++;
            }
        }
        // All pipes drain at kernel exit.
    }

private:
    struct TileDesc {
        int64_t srcElem; // first source complex element of the tile
        int64_t dstElem; // first output complex element of the tile
        int64_t rows;    // rows in this tile (source and destination rows consecutive)
    };

    // Tiles never cross a run boundary (slice end or source-row wrap) so both
    // spans of a tile stay contiguous in GM.
    __aicore__ inline bool NextTile(TileDesc& t)
    {
        if (tileRow_ >= rowEnd_) {
            return false;
        }
        const int64_t slice = tileRow_ / row_;
        const int64_t dstRow = tileRow_ - slice * row_;
        int64_t srcRow = dstRow - shiftRow_;
        if (srcRow < 0) {
            srcRow += row_;
        }
        int64_t runLen = rowEnd_ - tileRow_;
        if (row_ - dstRow < runLen) {
            runLen = row_ - dstRow;
        }
        if (row_ - srcRow < runLen) {
            runLen = row_ - srcRow;
        }
        int64_t rows = rmax_;
        if (runLen < rows) {
            rows = runLen;
        }
        t.srcElem = slice * dist_ + srcRow * col_;
        t.dstElem = tileRow_ * col_;
        t.rows = rows;
        tileRow_ += rows;
        return true;
    }

    // Stage 1 (MTE2): the whole tile as ONE contiguous block into Area A[slot]
    // (rows of a tile are consecutive in GM).
    __aicore__ inline void CopyInWholeRows(const TileDesc& t, int s)
    {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(t.rows * col_ * COMPLEX_BYTES);
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        DataCopyPad(ubBuf_->Get<float>()[AreaAFloatOff(s)], srcGM_[t.srcElem * 2], p, padParams);
        SetFlag<HardEvent::MTE2_V>(evFill_[s]);
    }

    // Stage 2 (Vector): repack from Area A (full rows) into RowArea, building
    // each OUTPUT ROW contiguously: [piece A | piece B]. Ordering is enforced
    // by the V_MTE3 / V_MTE2 SetFlag events issued right after this in
    // Process (they attach to the V queue tail and fire only when every V
    // instruction below has completed). When needScale_ each loaded chunk is
    // scaled in-register before it is stored.
    // LoadUnAlign/StoreUnAlign move at most ONE vector register per call
    // (VECTOR_REG_WIDTH / sizeof(float) = 64 floats), so wider pieces are
    // chunked like the framework's own pad implementation; the 4-argument
    // POST_MODE_UPDATE forms advance src/dst for the next chunk.
    // For COL > 0 the counts/pitch become compile-time constants, which is
    // what lets the compiler interleave the dependent unaligned chains of
    // consecutive rows (runtime counts serialize them, ~6x V-pipe time).
    __aicore__ inline void VectorRepack(const TileDesc& t, int s)
    {
        uint32_t countA;
        uint32_t countB;
        uint32_t rowPitch;
        if constexpr (COL > 0) {
            constexpr int64_t shiftC = COL / 2;
            countA = static_cast<uint32_t>(shiftC * 2);
            countB = static_cast<uint32_t>((COL - shiftC) * 2);
            rowPitch = static_cast<uint32_t>(COL * 2);
        } else {
            countA = static_cast<uint32_t>(lenA_ * 2);
            countB = static_cast<uint32_t>(lenB_ * 2);
            rowPitch = static_cast<uint32_t>(col_ * 2);
        }
        const uint32_t offA = countB; // src complex offset col-shiftC
        const uint16_t rows16 = static_cast<uint16_t>(t.rows);
        __ubuf__ float* areaA = AreaABase(s);
        __ubuf__ float* rowArea = RowAreaBase(s);

        __VEC_SCOPE__
        {
            // Piece A: dst cols [0, shiftC) <- src cols [col-shiftC, col).
            // Empty when col == 1 (shiftC == 0); the whole row is piece B.
            if (countA > 0) {
                for (uint16_t k = 0; k < rows16; k++) {
                    MicroAPI::RegTensor<float> regA;
                    MicroAPI::UnalignRegForLoad uLoad;
                    MicroAPI::UnalignRegForStore uStore;
                    __ubuf__ float* src = areaA + static_cast<uint32_t>(k) * rowPitch + offA;
                    __ubuf__ float* dst = rowArea + static_cast<uint32_t>(k) * rowPitch;
                    uint32_t rem = countA;
                    while (rem > 0) {
                        const uint32_t chunk = (rem > VEC_FLOATS) ? VEC_FLOATS : rem;
                        MicroAPI::LoadUnAlignPre(uLoad, src);
                        MicroAPI::LoadUnAlign<float>(regA, uLoad, src, chunk);
                        if (needScale_) {
                            ScaleReg(regA, scale_);
                        }
                        MicroAPI::StoreUnAlign<float>(dst, regA, uStore, chunk);
                        rem -= chunk;
                    }
                    MicroAPI::StoreUnAlignPost(dst, uStore, 0);
                }
            }
            // Piece B: dst cols [shiftC, col) <- src cols [0, col-shiftC).
            // Empty when col == 1 (shiftC == 0, col-shiftC == 0).
            if (countB > 0) {
                for (uint16_t k = 0; k < rows16; k++) {
                    MicroAPI::RegTensor<float> regB;
                    MicroAPI::UnalignRegForLoad uLoad;
                    MicroAPI::UnalignRegForStore uStore;
                    __ubuf__ float* src = areaA + static_cast<uint32_t>(k) * rowPitch;
                    __ubuf__ float* dst = rowArea + static_cast<uint32_t>(k) * rowPitch + countA;
                    uint32_t rem = countB;
                    while (rem > 0) {
                        const uint32_t chunk = (rem > VEC_FLOATS) ? VEC_FLOATS : rem;
                        MicroAPI::LoadUnAlignPre(uLoad, src);
                        MicroAPI::LoadUnAlign<float>(regB, uLoad, src, chunk);
                        if (needScale_) {
                            ScaleReg(regB, scale_);
                        }
                        MicroAPI::StoreUnAlign<float>(dst, regB, uStore, chunk);
                        rem -= chunk;
                    }
                    MicroAPI::StoreUnAlignPost(dst, uStore, 0);
                }
            }
        }
    }

    // Stage 3 (MTE3): ONE descriptor writes the tile's rows contiguously from
    // RowArea[slot]. Rows are consecutive output rows, so the GM destination
    // is one contiguous run: full-line writes, no write-allocate RFO, and no
    // lines split across descriptors.
    __aicore__ inline void CopyOutRows(const TileDesc& t, int s)
    {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(t.rows * col_ * COMPLEX_BYTES);
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPad(dstGM_[t.dstElem * 2], ubBuf_->Get<float>()[RowAreaFloatOff(s)], p);
        SetFlag<HardEvent::MTE3_V>(evFree_[s]);
    }

    // UB layout per slot: [Area A: R rows | RowArea: R rows], rows packed at
    // col*8 bytes; ComputeRmax keeps rmax*col a multiple of 4 complex
    // elements so all area bases are 32B aligned.
    __aicore__ inline uint32_t SlotFloatOff(int s) const
    {
        return static_cast<uint32_t>(s) * static_cast<uint32_t>(rmax_ * 2 * col_ * 2);
    }
    __aicore__ inline uint32_t AreaAFloatOff(int s) const { return SlotFloatOff(s); }
    __aicore__ inline uint32_t RowAreaFloatOff(int s) const
    {
        return SlotFloatOff(s) + static_cast<uint32_t>(rmax_ * col_ * 2);
    }
    __aicore__ inline __ubuf__ float* AreaABase(int s)
    {
        auto tensor = ubBuf_->Get<float>();
        return (__ubuf__ float*)tensor[AreaAFloatOff(s)].GetPhyAddr();
    }
    __aicore__ inline __ubuf__ float* RowAreaBase(int s)
    {
        auto tensor = ubBuf_->Get<float>();
        return (__ubuf__ float*)tensor[RowAreaFloatOff(s)].GetPhyAddr();
    }

    GlobalTensor<float> srcGM_;
    GlobalTensor<float> dstGM_;
    TBuf<TPosition::VECCALC>* ubBuf_ = nullptr;
    int64_t row_ = 0;
    int64_t col_ = 0;
    int64_t dist_ = 0;
    int64_t shiftRow_ = 0;
    int64_t shiftCol_ = 0;
    int64_t lenA_ = 0;
    int64_t lenB_ = 0;
    float scale_ = 1.0f;
    bool needScale_ = false;
    int64_t tileRow_ = 0;
    int64_t rowEnd_ = 0;
    int64_t rmax_ = 0;
    event_t evFill_[SLOTS]; // MTE2_V: A[s] filled
    event_t evFree_[SLOTS]; // MTE3_V: RowArea[s] released
    event_t evV3_[SLOTS];   // V_MTE3: V repack of RowArea[s] complete
    event_t evV2_[SLOTS];   // V_MTE2: V reads of A[s] complete
};

class IfftShiftAIV {
public:
    __aicore__ inline IfftShiftAIV() {}

    __aicore__ inline void Init(GM_ADDR gmSrc, GM_ADDR gmDst, GM_ADDR gmTiling)
    {
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        blockNum_ = static_cast<int64_t>(GetBlockNum());
        subBlockNum_ = static_cast<int64_t>(GetSubBlockNum());

        auto tilingPtr = reinterpret_cast<__gm__ IfftShiftTilingData*>(gmTiling);
        row_ = tilingPtr->row;
        col_ = tilingPtr->col;
        dist_ = tilingPtr->dist;
        totalRows_ = tilingPtr->totalRows;
        ubSize_ = tilingPtr->ubSize;
        scale_ = tilingPtr->scale;
        needScale_ = (scale_ != 1.0f);

        srcGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmSrc));
        dstGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmDst));

        const int64_t rmax = IfftShiftFastPipe<0>::ComputeRmax(col_, ubSize_);
        if (rmax < 1) {
            // Very wide column: piecewise fallback.
            pipe_.InitBuffer(ubBuf_, PING_PONG * TILE_COMPLEX * COMPLEX_BYTES);
            pingIdx_ = 0;
            // Both ping-pong buffers are initially free.
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            return;
        }
        if (col_ == 50) {
            fast50_.Init(srcGM_, dstGM_, pipe_, ubBuf_, row_, col_, dist_, ubSize_, scale_);
            return;
        }
        if (col_ == 48) {
            fast48_.Init(srcGM_, dstGM_, pipe_, ubBuf_, row_, col_, dist_, ubSize_, scale_);
            return;
        }
        fastRt_.Init(srcGM_, dstGM_, pipe_, ubBuf_, row_, col_, dist_, ubSize_, scale_);
    }

    __aicore__ inline void Process()
    {
        // On the 950 mix launch, GetBlockIdx() spans the MIX dimension:
        // blockNum physical blocks x subBlockNum (2) vector sub-block
        // contexts, i.e. 0..111 for blockDim 56 (physical block = idx / 2,
        // sub-core = idx % 2). Partition the rows over ALL contexts.
        int64_t ctxNum = blockNum_ * subBlockNum_;
        if (blockIdx_ > ctxNum - 1) {
            return;
        }
        int64_t perCtx = totalRows_ / ctxNum;
        int64_t remain = totalRows_ - perCtx * ctxNum;
        if (blockIdx_ < remain) {
            rowStart_ = blockIdx_ * (perCtx + 1);
            rowCount_ = perCtx + 1;
        } else {
            rowStart_ = remain * (perCtx + 1) + (blockIdx_ - remain) * perCtx;
            rowCount_ = perCtx;
        }
        if (rowCount_ <= 0) {
            return;
        }
        // Dispatch mirrors Init (deterministic in col_ / ubSize_).
        const int64_t rmax = IfftShiftFastPipe<0>::ComputeRmax(col_, ubSize_);
        if (rmax < 1) {
            GenericProcess();
            return;
        }
        if (col_ == 50) {
            fast50_.Process(rowStart_, rowCount_);
            return;
        }
        if (col_ == 48) {
            fast48_.Process(rowStart_, rowCount_);
            return;
        }
        fastRt_.Process(rowStart_, rowCount_);
    }

private:
    /*
     * Generic fallback for very wide columns: piecewise ping-pong staging.
     *   1. Rows are split into runs of consecutive output rows whose source
     *      rows are also consecutive (broken at slice boundaries and at the
     *      row-shift wrap).
     *   2. With f(c) = (c - shiftC) mod col, the column shift splits every
     *      row into two contiguous pieces:
     *        P1: dst cols [0, shiftC)      <- src cols [col-shiftC, col)
     *        P2: dst cols [shiftC, col)    <- src cols [0, col-shiftC)
     *   3. Per piece, batches of rows are staged into UB with DataCopyPad and
     *      flushed with one batched DataCopyPad. DataCopyPad stride
     *      semantics: GM side advance = blockLen + stride (bytes); UB side
     *      advance = ceil32(blockLen) + stride*32B, so staged rows are packed
     *      at ceil4(segLen) complex elements with UB-side stride 0.
     *   4. When needScale_ the staged tile is multiplied in place by a vector
     *      pass between copy-in and copy-out (MTE2 -> V -> MTE3 with the same
     *      MTE2_V / V_MTE3 handshakes as the fast path; the pad lanes between
     *      staged rows are scaled too, which is harmless because copy-out
     *      writes only the blockLen payload of each row).
     *   5. Two ping-pong UB buffers with explicit MTE3->MTE2 / MTE2->MTE3
     *      events keep copy-in and copy-out overlapped.
     *   6. Columns wider than the tile are split into tile-sized contiguous
     *      sub-blocks (blockCount = 1 copies).
     */
    __aicore__ inline void GenericProcess()
    {
        const int64_t shiftRow = row_ / 2;
        const int64_t shiftCol = col_ / 2;

        int64_t i = rowStart_;
        const int64_t end = rowStart_ + rowCount_;
        while (i < end) {
            const int64_t slice = i / row_;
            const int64_t dstRow = i - slice * row_;
            int64_t srcRow = dstRow - shiftRow;
            if (srcRow < 0) {
                srcRow += row_;
            }
            // Consecutive output rows map to consecutive source rows; the run is
            // capped by the core range, the slice end and the source row wrap.
            int64_t runLen = end - i;
            if (row_ - dstRow < runLen) {
                runLen = row_ - dstRow;
            }
            if (row_ - srcRow < runLen) {
                runLen = row_ - srcRow;
            }

            const int64_t srcBase = slice * dist_ + srcRow * col_;
            const int64_t dstBase = i * col_;
            // P1: dst cols [0, shiftC) <- src cols [col-shiftC, col), length shiftC
            CopyPieceRun(srcBase + (col_ - shiftCol), dstBase, shiftCol, runLen);
            // P2: dst cols [shiftC, col) <- src cols [0, col-shiftC), length col-shiftC
            CopyPieceRun(srcBase, dstBase + shiftCol, col_ - shiftCol, runLen);
            i += runLen;
        }
    }

    // In-place vector multiply of totalFloats floats starting at p, used by
    // the scaled generic path between copy-in and copy-out. src/dst are two
    // independent pointer handles onto the same address: the 4-argument
    // POST_MODE_UPDATE load/store forms each advance THEIR OWN pointer by one
    // chunk, so sharing one variable would skew the store one chunk ahead of
    // the load (the fast path is immune because its src/dst never alias).
    __aicore__ inline void VectorScaleTile(__ubuf__ float* p, uint32_t totalFloats)
    {
        if (totalFloats == 0) {
            return;
        }
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> reg;
            MicroAPI::UnalignRegForLoad uLoad;
            MicroAPI::UnalignRegForStore uStore;
            __ubuf__ float* src = p;
            __ubuf__ float* dst = p;
            uint32_t rem = totalFloats;
            while (rem > 0) {
                const uint32_t chunk = (rem > VEC_FLOATS) ? VEC_FLOATS : rem;
                MicroAPI::LoadUnAlignPre(uLoad, src);
                MicroAPI::LoadUnAlign<float>(reg, uLoad, src, chunk);
                IfftShiftFastPipe<0>::ScaleReg(reg, scale_);
                MicroAPI::StoreUnAlign<float>(dst, reg, uStore, chunk);
                rem -= chunk;
            }
            MicroAPI::StoreUnAlignPost(dst, uStore, 0);
        }
    }

    // Copies `rows` consecutive rows of one column piece of length segLen.
    // srcElemBase / dstElemBase are complex-element offsets of the piece start
    // in row 0; row-to-row stride on both sides is col_ complex elements.
    __aicore__ inline void CopyPieceRun(int64_t srcElemBase, int64_t dstElemBase, int64_t segLen, int64_t rows)
    {
        if (segLen <= 0 || rows <= 0) {
            return;
        }
        if (segLen <= TILE_COMPLEX) {
            // Staged rows are packed at ceil4(segLen) complex elements: the
            // hardware advances UB blocks by ceil32(blockLen), so the UB-side
            // stride of both copies is 0 and no explicit padding is needed.
            const int64_t pitch = (segLen + COMPLEX_PER_BLOCK - 1) / COMPLEX_PER_BLOCK * COMPLEX_PER_BLOCK;
            const int64_t rowsPerBatch = TILE_COMPLEX / pitch;
            const uint32_t gmGapBytes = static_cast<uint32_t>((col_ - segLen) * COMPLEX_BYTES); // GM-side, bytes
            const uint32_t segBytes = static_cast<uint32_t>(segLen * COMPLEX_BYTES);
            for (int64_t r0 = 0; r0 < rows; r0 += rowsPerBatch) {
                int64_t r = rows - r0;
                if (r > rowsPerBatch) {
                    r = rowsPerBatch;
                }
                const int64_t srcOff = srcElemBase + r0 * col_;
                const int64_t dstOff = dstElemBase + r0 * col_;
                auto ub = ubBuf_.Get<float>()[static_cast<uint32_t>(pingIdx_ * TILE_COMPLEX * 2)];
                const auto ev = (pingIdx_ == 0) ? EVENT_ID0 : EVENT_ID1;
                pingIdx_ = 1 - pingIdx_;

                WaitFlag<HardEvent::MTE3_MTE2>(ev);
                DataCopyExtParams inParams;
                inParams.blockCount = static_cast<uint32_t>(r);
                inParams.blockLen = segBytes;
                inParams.srcStride = gmGapBytes;
                inParams.dstStride = 0;
                DataCopyPadExtParams<float> padParams;
                padParams.isPad = false;
                DataCopyPad(ub, srcGM_[srcOff * 2], inParams, padParams);
                if (needScale_) {
                    // Scale the staged tile (pads included, harmless) before
                    // the payload is flushed by the batched copy-out below.
                    // The fill -> multiply edge uses the MTE2_V handshake of
                    // the proven fast path; the multiply -> copy-out edge is
                    // ordered by V_MTE3.
                    SetFlag<HardEvent::MTE2_V>(ev);
                    WaitFlag<HardEvent::MTE2_V>(ev);
                    VectorScaleTile((__ubuf__ float*)ub.GetPhyAddr(), static_cast<uint32_t>(r * pitch * 2));
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                } else {
                    SetFlag<HardEvent::MTE2_MTE3>(ev);
                    WaitFlag<HardEvent::MTE2_MTE3>(ev);
                }

                DataCopyExtParams outParams;
                outParams.blockCount = static_cast<uint32_t>(r);
                outParams.blockLen = segBytes;
                outParams.srcStride = 0;
                outParams.dstStride = gmGapBytes;
                DataCopyPad(dstGM_[dstOff * 2], ub, outParams);
                SetFlag<HardEvent::MTE3_MTE2>(ev);
            }
            return;
        }

        // Wide columns: split the piece of each row into tile-sized contiguous sub-blocks.
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t j = 0; j < segLen; j += TILE_COMPLEX) {
                int64_t len = segLen - j;
                if (len > TILE_COMPLEX) {
                    len = TILE_COMPLEX;
                }
                const int64_t srcOff = srcElemBase + r * col_ + j;
                const int64_t dstOff = dstElemBase + r * col_ + j;
                auto ub = ubBuf_.Get<float>()[static_cast<uint32_t>(pingIdx_ * TILE_COMPLEX * 2)];
                const auto ev = (pingIdx_ == 0) ? EVENT_ID0 : EVENT_ID1;
                pingIdx_ = 1 - pingIdx_;

                WaitFlag<HardEvent::MTE3_MTE2>(ev);
                DataCopyExtParams inParams;
                inParams.blockCount = 1;
                inParams.blockLen = static_cast<uint32_t>(len * COMPLEX_BYTES);
                inParams.srcStride = 0;
                inParams.dstStride = 0;
                DataCopyPadExtParams<float> padParams;
                padParams.isPad = false;
                DataCopyPad(ub, srcGM_[srcOff * 2], inParams, padParams);
                if (needScale_) {
                    SetFlag<HardEvent::MTE2_V>(ev);
                    WaitFlag<HardEvent::MTE2_V>(ev);
                    VectorScaleTile((__ubuf__ float*)ub.GetPhyAddr(), static_cast<uint32_t>(len * 2));
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                } else {
                    SetFlag<HardEvent::MTE2_MTE3>(ev);
                    WaitFlag<HardEvent::MTE2_MTE3>(ev);
                }

                DataCopyExtParams outParams;
                outParams.blockCount = 1;
                outParams.blockLen = static_cast<uint32_t>(len * COMPLEX_BYTES);
                outParams.srcStride = 0;
                outParams.dstStride = 0;
                DataCopyPad(dstGM_[dstOff * 2], ub, outParams);
                SetFlag<HardEvent::MTE3_MTE2>(ev);
            }
        }
    }

private:
    IfftShiftFastPipe<50> fast50_;
    IfftShiftFastPipe<48> fast48_;
    IfftShiftFastPipe<0> fastRt_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuf_;
    GlobalTensor<float> srcGM_;
    GlobalTensor<float> dstGM_;

    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
    int64_t subBlockNum_ = 1;
    int64_t row_ = 0;
    int64_t col_ = 0;
    int64_t dist_ = 0;
    int64_t totalRows_ = 0;
    int64_t ubSize_ = 0;
    float scale_ = 1.0f;
    bool needScale_ = false;
    int64_t rowStart_ = 0;
    int64_t rowCount_ = 0;
    uint32_t pingIdx_ = 0;
};
} // namespace IfftShiftNs

extern "C" __global__ __aicore__ void IfftShift(GM_ADDR gm_src, GM_ADDR gm_dst, GM_ADDR wksp, GM_ADDR gm_tiling)
{
    IfftShiftNs::IfftShiftAIV op;
    op.Init(gm_src, gm_dst, gm_tiling);
    op.Process();
}
