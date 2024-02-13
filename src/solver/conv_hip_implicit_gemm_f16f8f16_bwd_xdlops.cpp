/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/config.h>
#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/env.hpp>
#include <vector>
#include <cstdint>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_F16F8F16_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType, typename OutComputeType, typename WeiComputeType>
using DeviceOpF8Bwd = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
    3,
    ck::tensor_layout::convolution::NDHWGK,
    ck::tensor_layout::convolution::GKZYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NDHWGC,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    OutComputeType,
    WeiComputeType>;

template <typename DataType, typename OutComputeType, typename WeiComputeType>
using DeviceOpF8BwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpF8Bwd<DataType, OutComputeType, WeiComputeType>>;

namespace {

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);
        Di = 1;
        Do = 1;
        Z  = 1;

        input  = {G, N, C, Di, Hi, Wi};
        output = {G, N, K, Do, Ho, Wo};
        weight = {G, K, C, Z, Y, X};

        // CK strides are in GNCDHW order
        if(problem.IsLayoutNHWC())
        {
            in_strides  = {N * Di * Hi * Wi * C, Di * Hi * Wi * C, 1, Hi * Wi * C, Wi * C, C};
            out_strides = {N * Do * Ho * Wo * K, Do * Ho * Wo * K, 1, Ho * Wo * K, Wo * K, K};
            wei_strides = {K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};
        }
        else
        {
            assert(problem.IsLayoutDefault()); // already checked in IsApplicable
            // for default layout, we produce packed strides for NHWC layout
            // because we transpose to NHWC layout before calling CK kernel
            in_strides  = {C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};
            out_strides = {K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};
            wei_strides = {K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};
        }

        strides  = {1,
                   ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {1,
                    ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {0,
                    ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {0,
                    ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr, Data_t in, ConstData_t w, ConstData_t out) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {},
                                             in,
                                             output,
                                             out_strides,
                                             weight,
                                             wei_strides,
                                             {},
                                             {},
                                             input,
                                             in_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr, const ConvDataTensors& tensors) const
    {
        return MakeArgPtr(conv_ptr, tensors.out, tensors.w, tensors.in);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Di;
    int Ho;
    int Wo;
    int Do;
    int Y;
    int X;
    int Z;
    std::array<ck::index_t, 6> input;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> output;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> weight;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 3> strides;
    std::array<ck::index_t, 3> dilation;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};
} // namespace

template <typename DataType, typename OutComputeType, typename WeiComputeType>
void PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops::Init(const ProblemDescription& problem)
{
    valid_kernels =
        FillValidKernelsIDs<DeviceOpF8BwdPtrs<DataType, OutComputeType, WeiComputeType>, CKArgs>(
            problem);
    index     = 0;
    kernel_id = valid_kernels[index];
}

template <typename DataType, typename OutComputeType, typename WeiComputeType>
bool PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpF8BwdPtrs<DataType, OutComputeType, WeiComputeType>, CKArgs>(
        problem, kernel_id);
}

template <typename DataType, typename OutComputeType, typename WeiComputeType>
bool ConvHipImplicitGemmF16F8F16BwdXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpF8BwdPtrs<DataType, OutComputeType, WeiComputeType>, CKArgs>(
        problem);
}
#endif

void PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops::HeuristicInit(
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";

#if MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.GetIn().GetCastType() == miopenBFloat8 &&
       problem.GetWeights().GetCastType() == miopenFloat8)
        Init<ck::half_t, ck::bf8_t, ck::f8_t>(problem);
#endif
}

bool PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops::SetNextValue(
    const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit(problem);
        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
        return false;
}

bool PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.GetIn().GetCastType() == miopenBFloat8 &&
       problem.GetWeights().GetCastType() == miopenFloat8)
        return CheckIsSupportCKArgs<ck::half_t, ck::bf8_t, ck::f8_t>(problem);
#endif
    return false;
}

bool PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops
ConvHipImplicitGemmF16F8F16BwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext&, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops pp;
    pp.HeuristicInit(problem);
    return pp;
}

bool ConvHipImplicitGemmF16F8F16BwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemmF16F8F16BwdXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                      const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops
ConvHipImplicitGemmF16F8F16BwdXdlops::Search(const ExecutionContext& ctx,
                                             const ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmF16F8F16BwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(miopen::IsDisabled(ENV(MIOPEN_DEBUG_F16F8F16_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS)))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(problem.HasAtLeastOne64BitTensor())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsTensorsCasted())
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;
    if(!problem.IsFp16())
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;
    if(problem.GetIn().GetCastType() == miopenBFloat8 &&
       problem.GetWeights().GetCastType() == miopenFloat8)
        return CheckCKApplicability<ck::half_t, ck::bf8_t, ck::f8_t>(problem);
#endif
    return false;
}

ConvSolution ConvHipImplicitGemmF16F8F16BwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmF16F8F16BwdXdlops& config) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.IsLayoutNHWC()){
        return InitInvokerFactoryNHWC<DeviceOpF8BwdPtrs<ck::half_t, ck::bf8_t, ck::f8_t>,
                                    CKArgs,
                                    miopen::conv::DataInvokeParams>(ctx, problem, config.kernel_id);
    } else {
        return InitInvokerFactoryBwdNCHW<3,
                                    DeviceOpF8BwdPtrs<ck::half_t, ck::bf8_t, ck::f8_t>,
                                    CKArgs,
                                    miopen::conv::DataInvokeParams>(ctx, problem, config.kernel_id);
    }
#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
