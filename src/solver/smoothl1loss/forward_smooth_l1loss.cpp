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

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/smoothl1loss/invoke_params.hpp>
#include <miopen/smoothl1loss/solvers.hpp>
#include <miopen/smooth_l1loss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_5d.hpp>

#define LOCAL_SIZE_CONTIGUOUS 256
#define LOCAL_SIZE_NONCONTIGUOUS 256

namespace miopen {

namespace solver {

namespace smoothl1loss {

bool SmoothL1LossUnreducedForwardSolver::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::smoothl1loss::ProblemDescription& problem) const
{
    if(problem.GetReduction() != MIOPEN_LOSS_NO_REDUCTION)
        return false;
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsRightStride())
        return false;
    return true;
}

std::size_t SmoothL1LossUnreducedForwardSolver::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::smoothl1loss::ProblemDescription& /*problem*/) const
{
    return 0;
}

bool SmoothL1LossUnreducedForwardContiguous::IsApplicable(
    const ExecutionContext& context, const miopen::smoothl1loss::ProblemDescription& problem) const
{
    SmoothL1LossUnreducedForwardSolver::IsApplicable(context, problem);
    if(!problem.IsSameStride() && !problem.IsAllContiguous())
        return false;
    return true;
}

ConvSolution SmoothL1LossUnreducedForwardContiguous::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::smoothl1loss::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetIDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetIDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetODesc().GetType());
    auto size         = problem.GetODesc().GetElementSize();

    {
        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenSmoothL1Loss.cpp";
        kernel.kernel_name = "SmoothL1LossUnreducedForwardContiguous";
        xlocalsize         = LOCAL_SIZE_CONTIGUOUS;
        xgridsize          = AlignUp(size, xlocalsize);

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::smoothl1loss::InvokeParams>();

            auto size = params.iDesc->GetElementSize();

            kernel(params.i, params.t, params.o, params.beta, size);
        };
    };

    return result;
}

bool SmoothL1LossUnreducedForward5d::IsApplicable(
    const ExecutionContext& context, const miopen::smoothl1loss::ProblemDescription& problem) const
{
    SmoothL1LossUnreducedForwardSolver::IsApplicable(context, problem);
    if(problem.GetIDesc().GetSize() > 5)
        return false;
    return true;
}

ConvSolution SmoothL1LossUnreducedForward5d::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::smoothl1loss::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetIDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetIDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetODesc().GetType());
    auto size         = problem.GetODesc().GetElementSize();

    {
        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenSmoothL1Loss.cpp";
        kernel.kernel_name = "SmoothL1LossUnreducedForward5d";
        xlocalsize         = LOCAL_SIZE_NONCONTIGUOUS;
        xgridsize          = AlignUp(size, xlocalsize);

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::smoothl1loss::InvokeParams>();

            auto I_tv = get_inner_expanded_tv(deref(params.iDesc));
            auto T_tv = get_inner_expanded_tv(deref(params.tDesc));
            auto O_tv = get_inner_expanded_tv(deref(params.oDesc));

            kernel(params.i, params.t, params.o, params.beta, I_tv, T_tv, O_tv);
        };
    };

    return result;
}

} // namespace smoothl1loss

} // namespace solver

} // namespace miopen
