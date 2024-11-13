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
#include <miopen/rnn/solvers.hpp>

namespace miopen {

namespace rnn_base {

void RNNModularSingleStreamBWWeights::Compute(const Handle& handle,
                                              ConstData_t x,
                                              ConstData_t hx,
                                              Data_t dw,
                                              Data_t workSpace,
                                              size_t workSpaceSize,
                                              ConstData_t reserveSpace,
                                              size_t /*reserveSpaceSize*/) const
{

    if(rnnDesc.nLayers == 0 || max_seq_len == 0)
        return;

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    rnnAlgoModules.PrepareWriteBuffers(handle, dw);

    for(int layer_i = 0; layer_i < rnnDesc.nLayers; layer_i++)
    {
        if(layer_i == 0)
            rnnAlgoModules.PhisXInputWeights(handle, dw, workSpace, x);
        else
            rnnAlgoModules.HiddenXInputWeights(handle, dw, workSpace, reserveSpace, layer_i);

        rnnAlgoModules.BiasUpdate(handle, dw, workSpace, layer_i, workSpaceSize);

        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            rnnAlgoModules.PhisHStateWeights(
                handle, dw, workSpace, hx, layer_i, max_seq_len, seq_dir);

            rnnAlgoModules.HiddenHStateWeights(
                handle, dw, workSpace, reserveSpace, layer_i, max_seq_len, seq_dir);
        }
    }
}


void RNNDynamicModularSingleStreamBWWeights::Compute(const Handle& handle,
                                                     ConstData_t x,
                                                     ConstData_t hx,
                                                     Data_t dw,
                                                     Data_t workSpace,
                                                     size_t workSpaceSize,
                                                     ConstData_t reserveSpace,
                                                     size_t /*reserveSpaceSize*/) const
{
    const auto args_ext = rnnAlgoModules.createRuntimeArgsExt(
        runtimeArgsBWWeights{&handle, x, hx, dw, workSpace, reserveSpace});

    if(rnnDesc.nLayers == 0 || max_seq_len == 0)
        return;

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    rnnAlgoModules.PrepareWriteBuffers(handle, dw);

    rnnAlgoModules.realXProp(handle, args_ext);

    for(int layer_i = 0; layer_i < rnnDesc.nLayers; layer_i++)
    {
        if(layer_i == 0)
            rnnAlgoModules.PhisXInputWeights(handle, dw, workSpace, args_ext.tempX);
        else
            rnnAlgoModules.HiddenXInputWeights(handle, dw, workSpace, reserveSpace, layer_i);

        rnnAlgoModules.BiasUpdate(handle, dw, workSpace, layer_i, workSpaceSize);

        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            rnnAlgoModules.PhisHStateWeights(
                handle, dw, workSpace, hx, layer_i, max_seq_len, seq_dir);

            rnnAlgoModules.HiddenHStateWeights(
                handle, dw, workSpace, reserveSpace, layer_i, max_seq_len, seq_dir);
        }
    }
}

} // namespace rnn_base
} // namespace miopen
