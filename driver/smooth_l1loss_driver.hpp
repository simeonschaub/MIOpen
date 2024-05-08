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

template <typename Tgpu, typename Tcheck>
int32_t mloSmoothL1LossReducedBackwardRunHost(const miopenTensorDescriptor_t iDesc,
                                              const miopenTensorDescriptor_t tDesc,
                                              const miopenTensorDescriptor_t diDesc,
                                              const miopenTensorDescriptor_t dtDesc,
                                              const Tgpu* input,
                                              const Tgpu* target,
                                              const Tgpu* dO,
                                              Tcheck* dI,
                                              Tcheck* dT,
                                              const float beta,
                                              const float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv  = get_inner_expanded_tv(miopen::deref(iDesc));
    auto T_tv  = get_inner_expanded_tv(miopen::deref(tDesc));
    auto dI_tv = get_inner_expanded_tv(miopen::deref(diDesc));
    auto dT_tv = get_inner_expanded_tv(miopen::deref(dtDesc));

    auto size = miopen::deref(iDesc).GetElementSize();

    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);

        size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

        float sub  = input[Iidx] - target[Tidx];
        float grad = 0.0f;

        if(fabs(sub) < beta)
            grad = sub / beta * dO[0] / divisor;
        else
            grad = (sub >= 0 ? 1.0f : -1.0f) * dO[0] / divisor;

        if(dI)
            dI[TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4])] = grad;
        if(dT)
            dT[TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4])] = -grad;
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
    std::vector<int> GetTensorLengthsFromCmdLine();

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
    std::vector<Tref> workspacehost;
    std::vector<Tref> dIhost;
    std::vector<Tref> dThost;

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
    // forw           = inflags.GetValueInt("forw");
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

    SetTensorNd(diDesc, length, in_strides, data_type);
    SetTensorNd(dtDesc, length, tar_strides, data_type);

    if(reduction == "none")
    {
        divisor = std::numeric_limits<float>::quiet_NaN();
        SetTensorNd(doDesc, length, in_strides, data_type);
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(doDesc, out_lens, data_type);
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
        miopenGetSmoothL1LossReducedForwardWorkspaceSize(
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
    dI_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dT_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    dO_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in        = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    tar       = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    out       = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    workspace = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dI        = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dT        = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    dO        = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    outhost       = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    workspacehost = std::vector<Tref>(in_sz, static_cast<Tref>(0));
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

    dO[0] = static_cast<Tgpu>(0.5);

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
        std::cerr << "Error copying (tar) to GPU, size: " << tar_dev->GetSize() << std::endl;

    if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
        std::cerr << "Error copying (out grad) to GPU, size: " << dO_dev->GetSize() << std::endl;

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
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopen::deref(GetHandle()).ResetKernelTime();
        if(std::isnan(divisor)) {}
        else
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
        if(std::isnan(divisor))
        {
            std::cout << "-1\n";
        }
        else
        {
            int iter = inflags.GetValueInt("iter");
            if(WALL_CLOCK)
                std::cout << "Wall-clock Time Backward SmoothL1Loss Elapsed: "
                          << t.gettime_ms() / iter << " ms\n";

            float kernel_average_time =
                iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
            std::cout << "GPU Kernel Time Backward SmoothL1Loss Elapsed: " << kernel_average_time
                      << " ms\n";
        }
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
    if(std::isnan(divisor)) {}
    else
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

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SMOOTH_L1LOSS_DRIVER_HPP
