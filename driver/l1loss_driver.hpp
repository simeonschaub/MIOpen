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
#pragma once

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <limits>
#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

#include <vector>

template <typename Tgpu, typename Tcheck>
int mloL1LossReducedForwardRunHost(const miopenTensorDescriptor_t iDesc,
                                   const Tgpu* input,
                                   const Tgpu* target,
                                   Tcheck* workspacehost,
                                   Tcheck* outputhost,
                                   miopenLossReductionMode_t reduction)
{
    auto size      = miopen::deref(iDesc).GetElementSize();
    size_t divisor = (reduction == MIOPEN_LOSS_REDUCTION_MEAN) ? size : 1;

    // Phase 1: Calc loss for each element
    for(size_t i = 0; i < size; i++)
    {
        workspacehost[i] = abs(input[i] - target[i]) / divisor;
    }

    // Phase 2: Reduce
    float output = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        output += workspacehost[i];
    }
    outputhost[0] = output;

    return 0;
}

template <typename Tgpu, typename Tref>
class L1LossDriver : public Driver
{
public:
    L1LossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~L1LossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

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
    miopenLossReductionMode_t reduction;
};

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> L1LossDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    reduction    = static_cast<miopenLossReductionMode_t>(inflags.GetValueInt("reduction"));
    isContiguous = inflags.GetValueInt("contiguous") > 0 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::GetandSetData()
{
    auto in_len      = inflags.GetValueTensor("dim-lengths").lengths;
    auto in_strides  = ComputeStrides(in_len);
    auto tar_strides = ComputeStrides(in_len);

    SetTensorNd(inputDesc, in_len, in_strides, data_type);
    SetTensorNd(targetDesc, in_len, tar_strides, data_type);

    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        SetTensorNd(outputDesc, in_len, in_strides, data_type);
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(outputDesc, out_lens, data_type);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward L1Loss (Default=1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "256x512", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag("contiguous",
                         'C',
                         "1",
                         "Tensor is contiguous or not (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("reduction",
                         'R',
                         "0",
                         "Reduction mode ('none'(0) | 'sum'(1) |'mean'(2)) "
                         "(Default=0)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t tar_sz = GetTensorSize(targetDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    miopenGetL1LossForwardWorkspaceSize(
        GetHandle(), reduction, inputDesc, targetDesc, outputDesc, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    size_t ws_sz = ws_sizeInBytes / sizeof(Tgpu);

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    tar_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in        = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    tar       = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    out       = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    workspace = std::vector<Tgpu>(ws_sz, static_cast<Tgpu>(0));

    outhost       = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    workspacehost = std::vector<Tref>(ws_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(0.2));
    }

    for(int i = 0; i < tar_sz; i++)
    {
        tar[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.01), static_cast<Tgpu>(0.21));
    }

    fill(out.begin(), out.end(), static_cast<Tgpu>(0));

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
        std::cerr << "Error copying (tar) to GPU, size: " << tar_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenL1LossForward(GetHandle(),
                                                    reduction,
                                                    workspace_dev->GetMem(),
                                                    ws_sizeInBytes,
                                                    inputDesc,
                                                    in_dev->GetMem(),
                                                    targetDesc,
                                                    tar_dev->GetMem(),
                                                    outputDesc,
                                                    out_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenL1LossForward");

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
            std::cout << "Wall-clock Time Forward L1Loss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward L1Loss Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN || reduction == MIOPEN_LOSS_REDUCTION_SUM)
    {
        mloL1LossReducedForwardRunHost<Tgpu, Tref>(
            inputDesc, in.data(), tar.data(), workspacehost.data(), outhost.data(), reduction);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
Tref L1LossDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward L1Loss FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward L1Loss Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
