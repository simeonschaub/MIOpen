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
#ifndef GUARD_MIOPEN_L1LOSS_DRIVER_HPP
#define GUARD_MIOPEN_L1LOSS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/errors.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <cmath>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_5d.hpp>

#include <vector>

#ifndef MLO_L1LOSSMHOST_H_
#define MLO_L1LOSSMHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloL1LossReducedForwardRunHost(const miopenTensorDescriptor_t iDesc,
                                       const miopenTensorDescriptor_t tDesc,
                                       const Tgpu* input,
                                       const Tgpu* target,
                                       Tcheck* workspacehost,
                                       Tcheck* outputhost,
                                       miopenL1LossReduction_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv(miopen::deref(iDesc));
    auto T_tv = get_inner_expanded_tv(miopen::deref(tDesc));

    auto size = miopen::deref(iDesc).GetElementSize();

    int32_t divisor = (reduction == MIOPEN_L1LOSS_MEAN_REDUCTION) ? size : 1;

    // Phase 1: Calc loss for each element
    for(size_t i = 0; i < size; i++)
    {
        // uint64_t n[5];
        // GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);
        // uint64_t Iidx       = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        // uint64_t Tidx       = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        workspacehost[i] = abs(input[i] - target[i]) / divisor;
    }

    // Phase 2: Reduce
    double output = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        output += workspacehost[i];
    }
    outputhost[0] = output;

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
class L1LossDriver : public Driver
{
public:
    L1LossDriver() : Driver()
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
    std::vector<int> GetTensorLengthsFromCmdLine();

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
    std::vector<Tref> workspacehost;
    std::vector<Tref> dIhost;
    std::vector<Tref> dThost;

    size_t ws_sizeInBytes;

    miopenL1LossReduction_t reduction;
};

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::GetandSetData()
{
    reduction = static_cast<miopenL1LossReduction_t>(inflags.GetValueInt("Reduction"));

    auto length      = GetTensorLengthsFromCmdLine();
    auto in_strides  = GetStrides(length, inflags.GetValueInt("Contiguous"));
    auto tar_strides = GetStrides(length, inflags.GetValueInt("Contiguous"));

    SetTensorNd(inputDesc, length, in_strides, data_type);
    SetTensorNd(targetDesc, length, tar_strides, data_type);

    if(reduction == MIOPEN_L1LOSS_NONE_REDUCTION)
    {
        SetTensorNd(outputDesc, length, in_strides, data_type);
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(outputDesc, out_lens, data_type);
    }

    SetTensorNd(diDesc, length, in_strides, data_type);
    SetTensorNd(dtDesc, length, tar_strides, data_type);

    if(reduction == MIOPEN_L1LOSS_NONE_REDUCTION)
    {
        SetTensorNd(doDesc, length, in_strides, data_type);
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(doDesc, out_lens, data_type);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward L1Loss (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "256", "Mini-batch size (Default=2)", "int");
    inflags.AddInputFlag("in_channels", 'c', "4", "Number of Input Channels (Default=2)", "int");
    inflags.AddInputFlag("in_d", 'D', "1", "Input Depth (Default=1)", "int");
    inflags.AddInputFlag("in_h", 'H', "1", "Input Height (Default=1)", "int");
    inflags.AddInputFlag("in_w", 'W', "128", "Input Width (Default=2)", "int");
    inflags.AddInputFlag("Contiguous",
                         'C',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("Reduction",
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
std::vector<int> L1LossDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_d = inflags.GetValueInt("in_d");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_w});
    }
    else if(in_n != 0)
    {
        return std::vector<int>({in_n});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
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

    outhost       = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    workspacehost = std::vector<Tref>(ws_sz, static_cast<Tref>(0));
    dIhost        = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    dThost        = std::vector<Tref>(tar_sz, static_cast<Tref>(0));

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
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
        std::cerr << "Error copying (tar) to GPU, size: " << tar_dev->GetSize() << std::endl;

    if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
        std::cerr << "Error copying (out grad) to GPU, size: " << dO_dev->GetSize() << std::endl;

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
        miopenL1LossForward(GetHandle(),
                            reduction,
                            workspace_dev->GetMem(),
                            ws_sizeInBytes,
                            inputDesc,
                            in_dev->GetMem(),
                            targetDesc,
                            tar_dev->GetMem(),
                            outputDesc,
                            out_dev->GetMem());

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

    if(workspace_dev->FromGPU(GetStream(), workspace.data()) != 0)
        std::cerr << "Error copying (workspace_dev) from GPU, size: " << workspace_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(reduction == MIOPEN_L1LOSS_MEAN_REDUCTION || reduction == MIOPEN_L1LOSS_SUM_REDUCTION)
    {
        mloL1LossReducedForwardRunHost<Tgpu, Tref>(inputDesc,
                                                   targetDesc,
                                                   in.data(),
                                                   tar.data(),
                                                   workspacehost.data(),
                                                   outhost.data(),
                                                   reduction);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    /*
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopen::deref(GetHandle()).ResetKernelTime();
        if(!std::isnan(divisor))
        {
            miopenSmoothL1LossReducedBackward(GetHandle(),
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
    */

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int L1LossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    /*
    if(!std::isnan(divisor))
    {
        mloSmoothL1LossReducedBackwardRunHost<Tgpu, Tref>(inputDesc,
                                                          targetDesc,
                                                          diDesc,
                                                          dtDesc,
                                                          in.data(),
                                                          tar.data(),
                                                          dO.data(),
                                                          dIhost.data(),
                                                          dThost.data(),
                                                          beta,
                                                          divisor);
    }
    */

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref L1LossDriver<Tgpu, Tref>::GetTolerance()
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
int L1LossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);
    std::cout << "out host = " << outhost[0] << " out = " << out[0] << std::endl;

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
    /*
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_dI        = miopen::rms_range(dIhost, dI);
    auto error_dT        = miopen::rms_range(dThost, dT);

    if(!std::isfinite(error_dI) || error_dI > tolerance || !std::isfinite(error_dT) ||
       error_dT > tolerance)
    {
        std::cout << "Backward SmoothL1Loss FAILED: {" << error_dI << "," << error_dT << "} > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward SmoothL1Loss Verifies OK on CPU reference ({" << error_dI << ","
                  << error_dT << "} < " << tolerance << ')' << std::endl;
    }
    */

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_L1LOSS_DRIVER_HPP
