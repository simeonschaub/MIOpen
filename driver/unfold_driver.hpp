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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACTORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_UNFOLD_DRIVER_HPP
#define GUARD_MIOPEN_UNFOLD_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloUnfoldHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

template <typename Tgpu, typename Tref>
class UnfoldDriver : public Driver
{
public:
    UnfoldDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetTensorLengthsFromCmdLine();
    std::vector<int32_t> GetVectorInt32tFromCmdLine(std::string long_name);

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~UnfoldDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;

    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;

    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;

    std::vector<Tref> output_host;

    std::vector<Tref> doutput_host;

    std::vector<int32_t> kernel_size;
    std::vector<int32_t> stride;
    std::vector<int32_t> padding;
    std::vector<int32_t> dilation;
};

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_length = GetTensorLengthsFromCmdLine();

    kernel_size          = GetVectorInt32tFromCmdLine("kernelSize");
    stride               = GetVectorInt32tFromCmdLine("stride");
    padding              = GetVectorInt32tFromCmdLine("padding");
    dilation             = GetVectorInt32tFromCmdLine("dilation");
    int spatial_dim_size = input_length.size() - 2;
    const int N          = input_length[0];
    const int C          = input_length[1];

    int P = 1, L = 1;
    std::vector<int> ls;
    for(int i = 0; i < spatial_dim_size; ++i)
    {
        P *= kernel_size[i];
        int l = (input_length[i + 2] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) /
                    stride[i] +
                1;
        L *= l;
        ls.push_back(l);
    }
    std::vector<int> output_length = {N, (C * P), L};
    SetTensorNd(inputDesc, input_length, data_type);
    SetTensorNd(outputDesc, output_length, data_type);
    SetTensorNd(doutputDesc, output_length, data_type);
    SetTensorNd(dinputDesc, input_length, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run Unfold Forward (Default=1) or both Forward and Backward (0)", "int");
    inflags.AddInputFlag(
        "DimLengths", 'D', "2,5,3,4", "The dimensional lengths of the input tensor", "string");
    inflags.AddInputFlag("kernelSize", 'k', "2,3", "Kernel Size (Default=2,3)", "str");
    inflags.AddInputFlag("stride", 's', "1,1", "Stride (Default=1,1)", "str");
    inflags.AddInputFlag("padding", 'p', "0,0", "Stride (Default=0,0)", "str");
    inflags.AddInputFlag("dilation", 'd', "1,1", "Stride (Default=1,1)", "str");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> UnfoldDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine()
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
std::vector<int32_t> UnfoldDriver<Tgpu, Tref>::GetVectorInt32tFromCmdLine(std::string long_name)
{
    std::string lengthsStr = inflags.GetValueStr(long_name);

    std::vector<int32_t> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        lengths.push_back(static_cast<int32_t>(len));

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);
    int len              = std::stoi(sliceStr);

    lengths.push_back(static_cast<int32_t>(len));

    return (lengths);
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz  = GetTensorSize(inputDesc);
    size_t output_sz = GetTensorSize(outputDesc);

    size_t doutput_sz = GetTensorSize(doutputDesc);
    size_t dinput_sz  = GetTensorSize(dinputDesc);

    uint32_t ctx = 0;

    input_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));

    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, doutput_sz, sizeof(Tgpu)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dinput_sz, sizeof(Tgpu)));

    input  = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0.0f));
    output = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0.0f));

    doutput = std::vector<Tgpu>(doutput_sz, static_cast<Tgpu>(1.0f));
    dinput  = std::vector<Tgpu>(dinput_sz, static_cast<Tgpu>(0.0f));

    output_host = std::vector<Tref>(output_sz, static_cast<Tref>(0.0f));

    doutput_host = std::vector<Tref>(doutput_sz, static_cast<Tref>(0.0f));

    int status;

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    status = input_dev->ToGPU(GetStream(), input.data());

    for(int i = 0; i < doutput_sz; i++)
    {
        doutput[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= doutput_dev->ToGPU(GetStream(), doutput.data());
    status |= dinput_dev->ToGPU(GetStream(), dinput.data());

    if(status != 0)
        std::cout << "Unfold Driver Error copying data to GPU\n" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenUnfoldForward(GetHandle(),
                            inputDesc,
                            input_dev->GetMem(),
                            outputDesc,
                            output_dev->GetMem(),
                            kernel_size.data(),
                            kernel_size.size(),
                            stride.data(),
                            stride.size(),
                            padding.data(),
                            padding.size(),
                            dilation.data(),
                            dilation.size());

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
            std::cout << "Wall-clock Time Unfold Forward Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Unfold Forward Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloUnFoldFwd4DRunHost(input.data(),
                          inputDesc,
                          output_host.data(),
                          outputDesc,
                          kernel_size,
                          stride,
                          padding,
                          dilation);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref UnfoldDriver<Tgpu, Tref>::GetTolerance()
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
int UnfoldDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_output    = miopen::rms_range(output_host, output);

    if(!std::isfinite(error_output) || error_output > tolerance)
    {
        std::cout << "Forward Unfold FAILED: {" << error_output << "} > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Unfold Verifies OK on CPU reference ({" << error_output << "} < "
                  << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int UnfoldDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_UNFOLD_DRIVER_HPP
