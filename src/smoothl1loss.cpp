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
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/smoothl1loss/invoke_params.hpp>
#include <miopen/smoothl1loss/problem_description.hpp>
#include <miopen/smoothl1loss/solvers.hpp>
#include <miopen/smoothl1loss.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

size_t GetSmoothL1LossForwardWorkspaceSize(Handle& handle,
                                           const TensorDescriptor& iDesc,
                                           const TensorDescriptor& oDesc,
                                           const miopenLossReductionMode_t reduction)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = smoothl1loss::ForwardProblemDescription{iDesc, iDesc, oDesc, reduction};

    const auto solvers = solver::SolverContainer<solver::smoothl1loss::SmoothL1LossForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t SmoothL1LossForward(Handle& handle,
                                   Data_t workspace,
                                   size_t workspaceSizeInBytes,
                                   const TensorDescriptor& iDesc,
                                   ConstData_t i,
                                   const TensorDescriptor& tDesc,
                                   ConstData_t t,
                                   const TensorDescriptor& oDesc,
                                   Data_t o,
                                   float beta,
                                   const miopenLossReductionMode_t reduction)
{
    const auto problem = smoothl1loss::ForwardProblemDescription{iDesc, tDesc, oDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp           = smoothl1loss::FwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.iDesc          = &iDesc;
        tmp.tDesc          = &tDesc;
        tmp.oDesc          = &oDesc;
        tmp.i              = i;
        tmp.t              = t;
        tmp.o              = o;
        tmp.beta           = beta;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SmoothL1LossForward"};
    const auto solvers = solver::SolverContainer<solver::smoothl1loss::SmoothL1LossForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SmoothL1LossBackward(Handle& handle,
                                    const TensorDescriptor& iDesc,
                                    ConstData_t i,
                                    const TensorDescriptor& tDesc,
                                    ConstData_t t,
                                    const TensorDescriptor& dODesc,
                                    ConstData_t dO,
                                    const TensorDescriptor& dIDesc,
                                    Data_t dI,
                                    const TensorDescriptor& dTDesc,
                                    Data_t dT,
                                    float beta,
                                    const miopenLossReductionMode_t reduction)
{
    const auto problem =
        smoothl1loss::BackwardProblemDescription{iDesc, tDesc, dODesc, dIDesc, dTDesc, reduction};

    const auto invoke_params = [&]() {
        auto tmp   = smoothl1loss::BwdInvokeParams{};
        tmp.type   = InvokeType::Run;
        tmp.iDesc  = &iDesc;
        tmp.tDesc  = &tDesc;
        tmp.dODesc = &dODesc;
        tmp.dIDesc = &dIDesc;
        tmp.dTDesc = &dTDesc;
        tmp.i      = i;
        tmp.t      = t;
        tmp.dO     = dO;
        tmp.dI     = dI;
        tmp.dT     = dT;
        tmp.beta   = beta;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SmoothL1LossBackward"};
    const auto solvers = solver::SolverContainer<solver::smoothl1loss::SmoothL1LossBackward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
