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

#include <miopen/common.hpp>
#include <miopen/datatype.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/l1loss/invoke_params.hpp>
#include <miopen/l1loss/problem_description.hpp>
#include <miopen/l1loss/solvers.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <cstddef>
#include <vector>

#define LOCAL_SIZE_FWD 256
#define LOCAL_SIZE_REDUCE 1024

namespace miopen {

namespace solver {

namespace l1loss {

bool L1LossForward5d::IsImprovementOverROCm(
    const ExecutionContext& /*context*/, const miopen::l1loss::FwdProblemDescription& problem) const
{
    if(problem.GetReduction() == MIOPEN_LOSS_REDUCTION_NONE)
    {
        return false;
    }

    return true;
}

bool L1LossForward5d::IsApplicable(const ExecutionContext& /*context*/,
                                   const miopen::l1loss::FwdProblemDescription& problem) const
{
    if(!IsImprovementOverROCm({}, problem))
    {
        return false;
    }

    return true;
}

ConvSolution
L1LossForward5d::GetSolution(const ExecutionContext& /*context*/,
                             const miopen::l1loss::FwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype      = problem.GetODesc().GetType();
    auto io_dtype   = miopen::GetDataType(dtype);
    auto input_size = problem.GetIDesc().GetElementSize();

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
                              {"REDUCE_SIZE", LOCAL_SIZE_REDUCE}};

    {
        /* Phase 1: Calculate loss elementwise. (TIO to FLOAT_ACCUM) */
        size_t xlocalsize = LOCAL_SIZE_FWD;
        size_t xgridsize  = AlignUp(input_size, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenL1Loss.cpp";
        kernel.kernel_name = "L1LossReducedForward5d";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    {
        /* Phase 2: Reduce sum (FLOAT_ACCUM to FLOAT_ACCUM) */
        auto _size = input_size;

        do
        {
            size_t xlocalsize = LOCAL_SIZE_REDUCE;
            size_t xgridsize  = AlignUp(_size, xlocalsize);

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenReduceSum.cpp";
            kernel.kernel_name = "ReduceSumFLOATACCUM";

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(1);
            kernel.l_wk.push_back(1);
            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(1);
            kernel.g_wk.push_back(1);

            result.construction_params.push_back(kernel);
            _size = (_size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE;
        } while(_size > LOCAL_SIZE_REDUCE);

        /* Reduce sum (FLOAT_ACCUM to TIO) */
        size_t xlocalsize = LOCAL_SIZE_REDUCE;
        size_t xgridsize  = AlignUp(_size, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenReduceSum.cpp";
        kernel.kernel_name = "ReduceSum";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [input_size, dtype](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::l1loss::InvokeParams>();

            auto elapsed = 0.f;
            HipEventPtr start, stop;

            const bool profiling = handle_.IsProfilingEnabled();
            if(profiling)
            {
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            {
                /* Phase 1: Calculate loss elementwise. */
                auto I_tv      = get_inner_expanded_tv<5>(deref(params.iDesc));
                auto T_tv      = get_inner_expanded_tv<5>(deref(params.tDesc));
                size_t divisor = (params.reduction == MIOPEN_LOSS_REDUCTION_SUM) ? 1 : input_size;

                decltype(auto) kernel = handle_.Run(kernels.front());
                kernel(params.i, params.t, params.workspace, divisor, I_tv, T_tv);
            }

            {
                /* Phase 2: Reduce. */
                auto _size      = input_size;
                auto reduce_in  = params.workspace;
                auto reduce_out = static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                                      input_size * get_data_size(dtype));

                for(size_t i = 1; i < kernels.size() - 1; ++i)
                {
                    decltype(auto) kernel = handle_.Run(kernels[i]);

                    kernel(reduce_in, reduce_out, _size);
                    std::swap(reduce_in, reduce_out);

                    _size = (_size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE;
                }

                decltype(auto) kernel = handle_.Run(kernels.back());
                kernel(reduce_in, params.o, _size);

                if(profiling)
                {
                    hipEventRecord(stop.get(), handle_.GetStream());
                    hipEventSynchronize(stop.get());
                    hipEventElapsedTime(&elapsed, start.get(), stop.get());

                    // Clean up
                    hipEventDestroy(start.get());
                    hipEventDestroy(stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);

                    handle_.EnableProfiling(true);
                }
            }
        };
    };

    return result;
}

std::size_t
L1LossForward5d::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                  const miopen::l1loss::FwdProblemDescription& problem) const
{
    if(problem.GetReduction() == MIOPEN_LOSS_REDUCTION_NONE)
    {
        return 0;
    }

    size_t input_size = problem.GetIDesc().GetElementSize();
    return (input_size + (input_size + LOCAL_SIZE_REDUCE - 1) / LOCAL_SIZE_REDUCE) *
           get_data_size(miopenFloat);
}

} // namespace l1loss

} // namespace solver

} // namespace miopen
