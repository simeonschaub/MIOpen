/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_SMOOTH_L1LOSS_DRIVER_HPP
#define GUARD_MIOPEN_SMOOTH_L1LOSS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/errors.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_5d.hpp>

#include <vector>

#ifndef MLO_SMOOTH_L1LOSSMHOST_H_
#define MLO_SMOOTH_L1LOSSMHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloSmoothL1LossUnreducedForwardRunHost(const miopenTensorDescriptor_t iDesc,
                                               const miopenTensorDescriptor_t tDesc,
                                               const miopenTensorDescriptor_t oDesc,
                                               const Tgpu* input,
                                               const Tgpu* target,
                                               Tcheck* outputhost,
                                               const float beta)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv(miopen::deref(iDesc));
    auto T_tv = get_inner_expanded_tv(miopen::deref(tDesc));
    auto O_tv = get_inner_expanded_tv(miopen::deref(oDesc));

    auto size = miopen::deref(oDesc).GetElementSize();
    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, O_tv);

        uint64_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        uint64_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        uint64_t Oidx = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

        auto diff        = abs(input[Iidx] - target[Tidx]);
        outputhost[Oidx] = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloSmoothL1LossReducedForwardRunHost(const miopenTensorDescriptor_t iDesc,
                                             const miopenTensorDescriptor_t tDesc,
                                             const Tgpu* input,
                                             const Tgpu* target,
                                             Tcheck* workspacehost,
                                             Tcheck* outputhost,
                                             const float beta,
                                             const float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv(miopen::deref(iDesc));
    auto T_tv = get_inner_expanded_tv(miopen::deref(tDesc));

    auto size = miopen::deref(iDesc).GetElementSize();

    /* Phase 1: Calc loss for each element. */
    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);

        uint64_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        uint64_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

        auto diff = abs(input[Iidx] - target[Tidx]);
        workspacehost[Iidx] =
            (diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta) / divisor;
    });

    /* Phase 2: Reduce */
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            float shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspacehost[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < local_size; ++j)
                    if(j < offset)
                        shared[j] += shared[j + offset];
            if(_size <= local_size)
                outputhost[0] = shared[0];
            else
                workspacehost[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);

    return miopenStatusSuccess;
}
#endif

inline std::vector<int> GetStrides(std::vector<int> lengths, int contiguous)
{
    if(contiguous != 0 && contiguous != 1)
        std::cerr << "Error Tensor Contiguous should be 0 or 1" << std::endl;
    if(contiguous == 0)
        std::swap(lengths.front(), lengths.back());
    std::vector<int> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(contiguous == 0)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class SmoothL1LossDriver : public Driver
{
public:
    SmoothL1LossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SmoothL1LossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> tar_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> tar;
    std::vector<Tgpu> out;
    std::vector<Tgpu> workspace;
    std::vector<Tref> outhost;
    std::vector<Tref> workspacehost;

    size_t ws_sizeInBytes;

    float beta;
    float divisor;
};

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::GetandSetData()
{
    auto reduction = inflags.GetValueStr("Reduction");
    if(reduction != "none" && reduction != "mean" && reduction != "sum")
        return miopenStatusInvalidValue;

    auto length      = GetTensorLengthsFromCmdLine();
    auto in_strides  = GetStrides(length, 1);
    auto tar_strides = GetStrides(length, inflags.GetValueInt("Contiguous"));
    beta             = inflags.GetValueInt("Beta");

    SetTensorNd(inputDesc, length, in_strides, data_type);
    SetTensorNd(targetDesc, length, tar_strides, data_type);

    if(reduction == "none")
    {
        divisor = std::numeric_limits<float>::quiet_NaN();
        SetTensorNd(outputDesc, length, in_strides, data_type);
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(outputDesc, out_lens, data_type);
        if(reduction == "sum")
            divisor = 1;
        if(reduction == "mean")
            divisor = miopen::deref(inputDesc).GetElementSize();
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward SmoothL1Loss (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "256,4,1,1,8723",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag("Contiguous",
                         'C',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("Reduction",
                         'R',
                         "0",
                         "Specifies the reduction to apply to the output ('none'|'mean'|'sum') "
                         "(Default=none to indicate no reduction)",
                         "string");
    inflags.AddInputFlag("Beta",
                         'B',
                         "1",
                         "Specifies the threshold at which to change between L1 and L2 loss. The "
                         "value must be non-negative (Default=1)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> SmoothL1LossDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine()
{
    std::string lengthsStr = inflags.GetValueStr("DimLengths");

    std::vector<int> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);
    int len              = std::stoi(sliceStr);

    lengths.push_back(len);

    return (lengths);
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t tar_sz = GetTensorSize(targetDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    if(!std::isnan(divisor))
    {
        miopenGetSmoothL1LossReducedWorkspaceSize(
            GetHandle(), inputDesc, targetDesc, outputDesc, &ws_sizeInBytes);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            return miopenStatusAllocFailed;
    }
    else
        ws_sizeInBytes = 0;

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    tar_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in            = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    tar           = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    out           = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    workspace     = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    outhost       = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    workspacehost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(int i = 0; i < tar_sz; i++)
    {
        tar[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
        std::cerr << "Error copying (tar) to GPU, size: " << tar_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(std::isnan(divisor))
        {
            miopenSmoothL1LossUnreducedForward(GetHandle(),
                                               inputDesc,
                                               in_dev->GetMem(),
                                               targetDesc,
                                               tar_dev->GetMem(),
                                               outputDesc,
                                               out_dev->GetMem(),
                                               beta);
        }
        else
        {
            miopenSmoothL1LossReducedForward(GetHandle(),
                                             workspace_dev->GetMem(),
                                             ws_sizeInBytes,
                                             inputDesc,
                                             in_dev->GetMem(),
                                             targetDesc,
                                             tar_dev->GetMem(),
                                             outputDesc,
                                             out_dev->GetMem(),
                                             beta,
                                             divisor);
        }

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward SmoothL1Loss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward SmoothL1Loss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(std::isnan(divisor))
        mloSmoothL1LossUnreducedForwardRunHost<Tgpu, Tref>(
            inputDesc, targetDesc, outputDesc, in.data(), tar.data(), outhost.data(), beta);
    else
    {
        mloSmoothL1LossReducedForwardRunHost<Tgpu, Tref>(inputDesc,
                                                         targetDesc,
                                                         in.data(),
                                                         tar.data(),
                                                         workspacehost.data(),
                                                         outhost.data(),
                                                         beta,
                                                         divisor);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SmoothL1LossDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward SmoothL1Loss FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward SmoothL1Loss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SMOOTH_L1LOSS_DRIVER_HPP
