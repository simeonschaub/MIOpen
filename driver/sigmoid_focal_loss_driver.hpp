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
#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "mloSigmoidFocalLossHost.hpp"
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <cmath>
#include <vector>

const float MAX_FP16 = 65504;

template <typename Tgpu, typename Tref>
class SigmoidFocalLossDriver : public Driver
{
public:
    SigmoidFocalLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&dtargetDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
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
    ~SigmoidFocalLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(dtargetDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;
    miopenTensorDescriptor_t dtargetDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> dtarget_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> target;
    std::vector<Tgpu> output;
    std::vector<Tref> outputHost;
    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;
    std::vector<Tref> dinputHost;
    std::vector<Tgpu> dtarget;
    std::vector<Tref> dtargetHost;

    float alpha;
    float gamma;
    bool isContiguous;
    miopenLossReductionMode_t reduction;

    size_t workSpaceSizeInBytes;
};

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::GetandSetData()
{
    auto inDims  = inflags.GetValueTensor("dim-lengths").lengths;
    alpha        = inflags.GetValueDouble("alpha");
    gamma        = inflags.GetValueDouble("gamma");
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    reduction    = static_cast<miopenLossReductionMode_t>(inflags.GetValueInt("reduction"));

    std::vector<int> inStride = ComputeStrides(inDims);

    SetTensorNd(inputDesc, inDims, inStride, data_type);
    SetTensorNd(dinputDesc, inDims, data_type);
    SetTensorNd(targetDesc, inDims, inStride, data_type);
    SetTensorNd(dtargetDesc, inDims, data_type);

    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        SetTensorNd(outputDesc, inDims, data_type);
        SetTensorNd(doutputDesc, inDims, data_type);
    }
    else
    {
        std::vector<int> outDims = {1};
        SetTensorNd(outputDesc, outDims, data_type);
        SetTensorNd(doutputDesc, outDims, data_type);
    }

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> SigmoidFocalLossDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int SigmoidFocalLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "256x4x2", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag(
        "reduction", 'R', "0", "reduction mode: 0(default) - unreduced, 1 - sum, 2 -mean", "int");
    inflags.AddInputFlag("alpha", 'A', "0.25", "Alpha (Default=0.25)", "float");
    inflags.AddInputFlag("gamma", 'G', "2", "Gamma (Default=2)", "float");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz     = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dinputDesc).GetElementSize();
    size_t dT_sz     = miopen::deref(dtargetDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(Tgpu)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(Tgpu)));
    dtarget_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dT_sz, sizeof(Tgpu)));

    miopenGetSigmoidFocalLossForwardWorkspaceSize(
        GetHandle(), inputDesc, targetDesc, outputDesc, reduction, &workSpaceSizeInBytes);
    workspace_dev = std::make_unique<GPUMem>(ctx, workSpaceSizeInBytes, sizeof(std::byte));

    input       = std::vector<Tgpu>(in_sz);
    target      = std::vector<Tgpu>(target_sz);
    output      = std::vector<Tgpu>(out_sz);
    outputHost  = std::vector<Tref>(out_sz);
    doutput     = std::vector<Tgpu>(dO_sz, static_cast<Tgpu>(1));
    dinput      = std::vector<Tgpu>(dI_sz);
    dinputHost  = std::vector<Tref>(dI_sz);
    dtarget     = std::vector<Tgpu>(dT_sz);
    dtargetHost = std::vector<Tref>(dT_sz);

    for(int i = 0; i < in_sz; i++)
    {
        input[i]  = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-0.5), static_cast<Tgpu>(0.5));
        target[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-0.5), static_cast<Tgpu>(0.5));
    }

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (in) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
    {
        std::cerr << "Error copying (in) to GPU, size: " << target_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
    {
        std::cerr << "Error copying (dO) to GPU, size: " << doutput_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(dinput_dev->ToGPU(GetStream(), dinput.data()) != 0)
    {
        std::cerr << "Error copying (dI) to GPU, size: " << dinput_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(dtarget_dev->ToGPU(GetStream(), dtarget.data()) != 0)
    {
        std::cerr << "Error copying (dT) to GPU, size: " << dtarget_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSigmoidFocalLossForward(GetHandle(),
                                      workspace_dev->GetMem(),
                                      workSpaceSizeInBytes,
                                      inputDesc,
                                      input_dev->GetMem(),
                                      targetDesc,
                                      target_dev->GetMem(),
                                      outputDesc,
                                      output_dev->GetMem(),
                                      alpha,
                                      gamma,
                                      reduction);
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
            std::cout << "Wall-clock Time Sigmoid Focal Loss Fwd Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Sigmoid Focal Loss Fwd Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    auto status = mloSigmoidFocalLossFwdRunHost<Tgpu, Tref>(input.data(),
                                                            inputDesc,
                                                            target.data(),
                                                            targetDesc,
                                                            outputHost.data(),
                                                            outputDesc,
                                                            alpha,
                                                            gamma,
                                                            reduction);
    return status;
}

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSigmoidFocalLossBackward(GetHandle(),
                                       inputDesc,
                                       input_dev->GetMem(),
                                       targetDesc,
                                       target_dev->GetMem(),
                                       doutputDesc,
                                       doutput_dev->GetMem(),
                                       dinputDesc,
                                       dinput_dev->GetMem(),
                                       dtargetDesc,
                                       dtarget_dev->GetMem(),
                                       alpha,
                                       gamma,
                                       reduction);

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
            std::cout << "Wall-clock Time Sigmoid Focal Loss Bwd Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Sigmoid Focal Loss Bwd Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
    {
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }
    if(dtarget_dev->FromGPU(GetStream(), dtarget.data()) != 0)
    {
        std::cerr << "Error copying (dT_dev) from GPU, size: " << dtarget_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    auto status = mloSigmoidFocalLossBwdRunHost<Tgpu, Tref>(input.data(),
                                                            inputDesc,
                                                            target.data(),
                                                            targetDesc,
                                                            doutput.data(),
                                                            doutputDesc,
                                                            dinputHost.data(),
                                                            dinputDesc,
                                                            dtargetHost.data(),
                                                            dtargetDesc,
                                                            alpha,
                                                            gamma,
                                                            reduction);

    return status;
}

template <typename Tgpu, typename Tref>
Tref SigmoidFocalLossDriver<Tgpu, Tref>::GetTolerance()
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
int SigmoidFocalLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    if(miopen::deref(inputDesc).GetType() == miopenHalf &&
       reduction != MIOPEN_LOSS_REDUCTION_NONE && abs(outputHost[0]) > MAX_FP16)
    {
        std::cout << "Float16 overflow - CPU output: " << outputHost[0] << std::endl;
    }

    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Sigmoid Focal Loss FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Sigmoid Focal Loss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SigmoidFocalLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance = GetTolerance();
    auto dinputError     = miopen::rms_range(dinputHost, dinput);
    auto dtargetError    = miopen::rms_range(dtargetHost, dtarget);

    if(!std::isfinite(dinputError) || dinputError > tolerance)
    {
        std::cout << "Backward Sigmoid Focal Loss Input Gradient FAILED: " << dinputError << " > "
                  << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else if(!std::isfinite(dtargetError) || dtargetError > tolerance)
    {
        std::cout << "Backward Sigmoid Focal Loss Target Gradient FAILED: " << dtargetError << " > "
                  << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Sigmoid Focal Loss Verifies OK on CPU reference (dinput: "
                  << dinputError << ", dtarget: " << dtargetError << "< " << tolerance << ')'
                  << std::endl;
    }

    return miopenStatusSuccess;
}
