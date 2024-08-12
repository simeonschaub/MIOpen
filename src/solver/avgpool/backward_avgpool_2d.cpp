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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <miopen/avgpool/solvers.hpp>

#include <miopen/avgpool/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/avgpool.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE_BWD_2D 1024

namespace miopen {

namespace solver {

namespace avgpool {

bool AvgPoolBackward2d::IsApplicable(const ExecutionContext& context,
                                     const miopen::avgpool::BwdProblemDescription& problem) const
{
    if(problem.GetInputGradDesc().GetNumDims() != 4 ||
       problem.GetOutputGradDesc().GetNumDims() != 4)
    {
        return false;
    }
    return true;
}

ConvSolution
AvgPoolBackward2d::GetSolution(const ExecutionContext& context,
                               const miopen::avgpool::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetOutputGradDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetInputGradDesc().GetType());
    auto dtype        = problem.GetInputGradDesc().GetType();
    size_t N_total    = problem.GetNtotal();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE_BWD_2D}, {N_total}, "MIOpenAvgPool.cpp", "AvgPoolBackward2d", build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::avgpool::BwdInvokeParams>();

            decltype(auto) kernel = handle_.Run(kernels.front());

            auto input_grad_tv  = get_inner_expanded_tv<4>(deref(params.inputGradDesc));
            auto output_grad_tv = get_inner_expanded_tv<4>(deref(params.outputGradDesc));

            auto N  = deref(params.inputGradDesc).GetLengths()[0];
            auto C  = deref(params.inputGradDesc).GetLengths()[1];
            auto H  = deref(params.inputGradDesc).GetLengths()[2];
            auto W  = deref(params.inputGradDesc).GetLengths()[3];
            auto OH = deref(params.outputGradDesc).GetLengths()[2];
            auto OW = deref(params.outputGradDesc).GetLengths()[3];

            kernel(params.output_grad,
                   params.input_grad,
                   N,
                   C,
                   H,
                   W,
                   OH,
                   OW,
                   params.kinfor,
                   params.stride,
                   params.padding,
                   params.count_include_pad,
                   params.divisor_override,
                   output_grad_tv,
                   input_grad_tv);
        };
    };

    return result;
}

} // namespace avgpool

} // namespace solver

} // namespace miopen
