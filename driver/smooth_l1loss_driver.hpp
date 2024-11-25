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
#include "miopen/errors.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <vector>

#ifndef MLO_SMOOTH_L1LOSSMHOST_H_
#define MLO_SMOOTH_L1LOSSMHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloSmoothL1LossForwardRunHost(const miopenTensorDescriptor_t iDesc,
                                      const miopenTensorDescriptor_t tDesc,
                                      const miopenTensorDescriptor_t oDesc,
                                      const Tgpu* input,
                                      const Tgpu* target,
                                      Tcheck* outputhost,
                                      const float beta,
                                      const miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv<5>(miopen::deref(iDesc));
    auto T_tv = get_inner_expanded_tv<5>(miopen::deref(tDesc));
    auto O_tv = get_inner_expanded_tv<5>(miopen::deref(oDesc));

    auto size       = miopen::deref(iDesc).GetElementSize();
    double loss_sum = 0.0;

    ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);

        auto diff = abs(input[Iidx] - target[Tidx]);
        auto loss = (diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta);

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
            outputhost[O_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tcheck>(loss);
        else
            loss_sum += loss;
    });
    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        loss_sum /= size;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        outputhost[0] = static_cast<Tcheck>(loss_sum);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloSmoothL1LossBackwardRunHost(const miopenTensorDescriptor_t iDesc,
                                       const miopenTensorDescriptor_t tDesc,
                                       const miopenTensorDescriptor_t dODesc,
                                       const miopenTensorDescriptor_t diDesc,
                                       const miopenTensorDescriptor_t dtDesc,
                                       const Tgpu* input,
                                       const Tgpu* target,
                                       const Tgpu* dO,
                                       Tcheck* dI,
                                       Tcheck* dT,
                                       const float beta,
                                       const miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv  = get_inner_expanded_tv<5>(miopen::deref(iDesc));
    auto T_tv  = get_inner_expanded_tv<5>(miopen::deref(tDesc));
    auto dI_tv = get_inner_expanded_tv<5>(miopen::deref(diDesc));
    auto dT_tv = get_inner_expanded_tv<5>(miopen::deref(dtDesc));
    auto dO_tv = get_inner_expanded_tv<5>(miopen::deref(dODesc));

    auto size = miopen::deref(iDesc).GetElementSize();

    par_ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);

        float sub  = input[Iidx] - target[Tidx];
        float grad = 0.0f;

        if(fabs(sub) < beta)
            grad = sub / beta *
                   dO[reduction == MIOPEN_LOSS_REDUCTION_NONE
                          ? dO_tv.get_tensor_view_idx(tensor_layout)
                          : 0];
        else
            grad = (sub >= 0 ? 1.0f : -1.0f) * dO[reduction == MIOPEN_LOSS_REDUCTION_NONE
                                                      ? dO_tv.get_tensor_view_idx(tensor_layout)
                                                      : 0];

        if(dI)
            dI[dI_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tcheck>(grad);
        if(dT)
            dT[dT_tv.get_tensor_view_idx(tensor_layout)] = static_cast<Tcheck>(-grad);
    });

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
        miopenCreateTensorDescriptor(&diDesc);
        miopenCreateTensorDescriptor(&dtDesc);
        miopenCreateTensorDescriptor(&doDesc);

        data_type = miopen_type<Tgpu>{};
    }

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
    ~SmoothL1LossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(diDesc);
        miopenDestroyTensorDescriptor(dtDesc);
        miopenDestroyTensorDescriptor(doDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t diDesc;
    miopenTensorDescriptor_t dtDesc;
    miopenTensorDescriptor_t doDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> tar_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev;
    std::unique_ptr<GPUMem> dI_dev;
    std::unique_ptr<GPUMem> dT_dev;
    std::unique_ptr<GPUMem> dO_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> tar;
    std::vector<Tgpu> out;
    std::vector<Tgpu> workspace;
    std::vector<Tgpu> dI;
    std::vector<Tgpu> dT;
    std::vector<Tgpu> dO;

    std::vector<Tref> outhost;
    std::vector<Tref> dIhost;
    std::vector<Tref> dThost;

    size_t ws_sizeInBytes;

    float beta;
    miopenLossReductionMode_t reduction_mode;
};

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    auto reduction = inflags.GetValueStr("Reduction");
    if(reduction != "none" && reduction != "mean" && reduction != "sum")
        return miopenStatusInvalidValue;
    if(reduction == "none")
        reduction_mode = MIOPEN_LOSS_REDUCTION_NONE;
    else if(reduction == "mean")
        reduction_mode = MIOPEN_LOSS_REDUCTION_MEAN;
    else if(reduction == "sum")
        reduction_mode = MIOPEN_LOSS_REDUCTION_SUM;

    beta = inflags.GetValueInt("Beta");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::GetandSetData()
{
    auto length      = inflags.GetValueTensor("input").lengths;
    auto in_strides  = GetStrides(length, 1);
    auto tar_strides = GetStrides(length, inflags.GetValueInt("Contiguous"));

    if(SetTensorNd(inputDesc, length, in_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input") + ".");
    if(SetTensorNd(targetDesc, length, tar_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing target tensor");

    if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
    {
        if(SetTensorNd(outputDesc, length, in_strides, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output tensor");
    }
    else
    {
        std::vector<int> out_lens = {1};
        if(SetTensorNd(outputDesc, out_lens, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output tensor");
    }

    if(SetTensorNd(diDesc, length, in_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input gradient tensor");
    if(SetTensorNd(dtDesc, length, tar_strides, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing target gradient tensor");

    if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
    {
        if(SetTensorNd(doDesc, length, in_strides, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output gradient tensor");
    }
    else
    {
        std::vector<int> out_lens = {1};
        if(SetTensorNd(doDesc, out_lens, data_type) != miopenStatusSuccess)
            MIOPEN_THROW("Error parsing output gradient tensor");
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward SmoothL1Loss (Default=1)", "int");
    inflags.AddInputFlag("input",
                         'D',
                         "256x4x1x1x8723",
                         "Input tensor descriptor (Default=256x4x1x1x8723)",
                         "tensor");
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
int SmoothL1LossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t tar_sz = GetTensorSize(targetDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    miopenGetSmoothL1LossForwardWorkspaceSize(
        GetHandle(), inputDesc, outputDesc, reduction_mode, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;
    size_t ws_sz = ws_sizeInBytes / sizeof(Tgpu);

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    tar_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));
    dI_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dT_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    dO_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in        = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    tar       = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    out       = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    workspace = std::vector<Tgpu>(ws_sz, static_cast<Tgpu>(0));
    dI        = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dT        = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    dO        = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    dIhost  = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    dThost  = std::vector<Tref>(tar_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(0.2));
    }

    for(int i = 0; i < tar_sz; i++)
    {
        tar[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.01), static_cast<Tgpu>(0.21));
    }

    fill(out.begin(), out.end(), static_cast<Tgpu>(0));

    fill(dO.begin(), dO.end(), static_cast<Tgpu>(0.5));

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
    {
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
    {
        std::cerr << "Error copying (tar) to GPU, size: " << tar_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
    {
        std::cerr << "Error copying (out grad) to GPU, size: " << dO_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

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
        miopenSmoothL1LossForward(GetHandle(),
                                  workspace_dev->GetMem(),
                                  ws_sizeInBytes,
                                  inputDesc,
                                  in_dev->GetMem(),
                                  targetDesc,
                                  tar_dev->GetMem(),
                                  outputDesc,
                                  out_dev->GetMem(),
                                  beta,
                                  reduction_mode);

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
    {
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::RunForwardCPU()
{
    auto status = mloSmoothL1LossForwardRunHost<Tgpu, Tref>(inputDesc,
                                                            targetDesc,
                                                            outputDesc,
                                                            in.data(),
                                                            tar.data(),
                                                            outhost.data(),
                                                            beta,
                                                            reduction_mode);

    return status;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopen::deref(GetHandle()).ResetKernelTime();
        miopenSmoothL1LossBackward(GetHandle(),
                                   inputDesc,
                                   in_dev->GetMem(),
                                   targetDesc,
                                   tar_dev->GetMem(),
                                   doDesc,
                                   dO_dev->GetMem(),
                                   diDesc,
                                   dI_dev->GetMem(),
                                   dtDesc,
                                   dT_dev->GetMem(),
                                   beta,
                                   reduction_mode);

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
            std::cout << "Wall-clock Time Backward SmoothL1Loss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward SmoothL1Loss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(dI_dev->FromGPU(GetStream(), dI.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dI_dev->GetSize() << std::endl;
    if(dT_dev->FromGPU(GetStream(), dT.data()) != 0)
        std::cerr << "Error copying (dT_dev) from GPU, size: " << dT_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    auto status = mloSmoothL1LossBackwardRunHost<Tgpu, Tref>(inputDesc,
                                                             targetDesc,
                                                             doDesc,
                                                             diDesc,
                                                             dtDesc,
                                                             in.data(),
                                                             tar.data(),
                                                             dO.data(),
                                                             dIhost.data(),
                                                             dThost.data(),
                                                             beta,
                                                             reduction_mode);

    return status;
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
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_dI        = miopen::rms_range(dIhost, dI);
    auto error_dT        = miopen::rms_range(dThost, dT);

    if(!std::isfinite(error_dI) || error_dI > tolerance)
    {
        std::cout << "Backward SmoothL1Loss Input Gradient FAILED: " << error_dI << " > "
                  << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward SmoothL1Loss Input Gradient Verifies OK on CPU reference ("
                  << error_dI << " < " << tolerance << ')' << std::endl;
    }

    if(!std::isfinite(error_dT) || error_dT > tolerance)
    {
        std::cout << "Backward SmoothL1Loss Target Gradient FAILED: " << error_dT << " > "
                  << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward SmoothL1Loss Target Gradient Verifies OK on CPU reference ("
                  << error_dT << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
