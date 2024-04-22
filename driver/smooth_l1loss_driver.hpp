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

#include <vector>

#ifndef MLO_SMOOTH_L1LOSSMHOST_H_
#define MLO_SMOOTH_L1LOSSMHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloSmoothL1LossForwardRunHost(const miopenTensorDescriptor_t tensorDesc,
                                      const Tgpu* input,
                                      const Tgpu* target,
                                      Tcheck* outputhost,
                                      const miopenLossReduction_t reduction,
                                      const float beta)
{
    size_t size = miopen::deref(tensorDesc).GetElementSize();

    auto loss_no_reduce = [&]() {
        par_ford(size)([&](size_t i) {
            auto diff     = static_cast<Tcheck>(abs(input[i] - target[i]));
            outputhost[i] = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;
        });
    };

    switch(reduction)
    {
    case MIOPEN_LOSS_MEAN_REDUCTION: std::cout << "Unsupported Mean Reduction" << std::endl; break;
    case MIOPEN_LOSS_SUM_REDUCTION: std::cout << "Unsupported Sum Reduction" << std::endl; break;
    default: loss_no_reduce(); break;
    }

    return miopenStatusSuccess;
}
#endif

template <typename Tgpu, typename Tref>
class SmoothL1LossDriver : public Driver
{
public:
    SmoothL1LossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&tensorDesc);

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
    ~SmoothL1LossDriver() override { miopenDestroyTensorDescriptor(tensorDesc); }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t tensorDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> tar_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> tar;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    size_t ws_sizeInBytes;

    float beta;
    miopenLossReduction_t reduction;
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
    std::vector<int> in_len = GetTensorLengthsFromCmdLine();
    beta                    = inflags.GetValueInt("Beta");

    SetTensorNd(tensorDesc, in_len, data_type);

    reduction = static_cast<miopenLossReduction_t>(inflags.GetValueInt("Reduction"));

    return 0;
}

template <typename Tgpu, typename Tref>
int SmoothL1LossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward SmoothL1Loss (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");

    inflags.AddInputFlag("Reduction",
                         'R',
                         "0",
                         "Specifies the reduction to apply to the output (check the "
                         "miopenLossReduction_t in miopen.h) (Default=0 to indicate no reduction)",
                         "int");
    inflags.AddInputFlag("Beta",
                         'B',
                         "1",
                         "Specifies the threshold at which to change between L1 and L2 loss. The "
                         "value must be non-negative(Default=1)",
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
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_w = inflags.GetValueInt("in_w");
    int in_h = inflags.GetValueInt("in_h");
    int in_d = inflags.GetValueInt("in_d");

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
int SmoothL1LossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t size = GetTensorSize(tensorDesc);

    miopenGetSmoothL1LossWorkspaceSize(
        GetHandle(), reduction, tensorDesc, tensorDesc, tensorDesc, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, size, sizeof(Tgpu)));
    tar_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, size, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, size, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in      = std::vector<Tgpu>(size, static_cast<Tgpu>(0));
    tar     = std::vector<Tgpu>(size, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(size, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(size, static_cast<Tref>(0));

    for(int i = 0; i < size; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
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
        miopenSmoothL1LossForward(GetHandle(),
                                  reduction,
                                  tensorDesc,
                                  in_dev->GetMem(),
                                  tensorDesc,
                                  tar_dev->GetMem(),
                                  tensorDesc,
                                  out_dev->GetMem(),
                                  beta);

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
    mloSmoothL1LossForwardRunHost<Tgpu, Tref>(
        tensorDesc, in.data(), tar.data(), outhost.data(), reduction, beta);

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
