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

#include <miopen/errors.hpp>
#include <miopen/graphapi/conv_bias_res_add_activ_forward_executor.hpp>
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace miopen {

namespace graphapi {

GraphPatternExecutor::~GraphPatternExecutor() = default;

size_t GraphExecutorFind20::getWorkspaceSize() const { return mSolution.GetWorkspaceSize(); }

namespace {

const std::string_view base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                     "abcdefghijklmnopqrstuvwxyz"
                                     "0123456789+/";

std::string base64Encode(const char* data, size_t length)
{
    // Calculate the exact size of the resulting Base64 string
    size_t outputSize = ((length + 2) / 3) * 4;
    std::string encodedString;
    encodedString.reserve(outputSize); // Preallocate memory for the result

    size_t i = 0;
    unsigned char charArray3[3];
    unsigned char charArray4[4];

    while(length-- != 0U)
    {
        charArray3[i++] = *(data++);
        if(i == 3)
        {
            charArray4[0] = (charArray3[0] & 0xfc) >> 2;
            charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
            charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
            charArray4[3] = charArray3[2] & 0x3f;

            std::transform(std::begin(charArray4),
                           std::end(charArray4),
                           std::back_inserter(encodedString),
                           [](unsigned char c) { return base64Chars[c]; });
            i = 0;
        }
    }

    if(i != 0U)
    {
        for(size_t j = i; j < 3; j++)
        {
            charArray3[j] = '\0';
        }

        charArray4[0] = (charArray3[0] & 0xfc) >> 2;
        charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
        charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
        charArray4[3] = charArray3[2] & 0x3f;

        for(size_t j = 0; j < i + 1; j++)
        {
            encodedString += base64Chars[charArray4[j]];
        }

        while(i++ < 3)
        {
            encodedString += '=';
        }
    }

    return encodedString;
}

std::string base64Decode(const std::string& encodedString)
{
    size_t length = encodedString.size();
    if(length % 4 != 0)
    {
        throw std::invalid_argument("Invalid Base64 input");
    }

    std::string decodedString;
    unsigned char charArray4[4], charArray3[3];
    size_t i = 0;

    for(char c : encodedString)
    {
        if(c == '=')
            break;

        auto it = std::find(base64Chars.begin(), base64Chars.end(), c);
        if(it == base64Chars.end())
        {
            throw std::invalid_argument("Invalid character in Base64 string");
        }

        charArray4[i++] = static_cast<unsigned char>(std::distance(base64Chars.begin(), it));
        if(i == 4)
        {
            charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
            charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
            charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

            decodedString.append(std::begin(charArray3), std::end(charArray3));
            i = 0;
        }
    }

    if(i != 0U)
    {
        for(size_t j = i; j < 4; j++)
        {
            charArray4[j] = 0;
        }

        charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
        charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
        charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

        for(size_t j = 0; j < i - 1; j++)
        {
            decodedString += charArray3[j];
        }
    }

    return decodedString;
}

} // namespace

nlohmann::json GraphExecutorFind20::getJson()
{
    std::map<int64_t, miopenTensorArgumentId_t> id2ArgumentMap{};

    for(const auto& [tensorId, tensorInfo] : *mTensorInfoMap)
    {
        id2ArgumentMap.try_emplace(tensorId, tensorInfo.mEnumId);
    }

    std::size_t size{0U};
    auto status = miopenGetSolutionSize(&mSolution, &size);
    MIOPEN_THROW_IF(status != miopenStatusSuccess,
                    "Serialization size for Solution wasn't obtained");

    std::vector<char> serializedSolution(size);
    status = ::miopenSaveSolution(&mSolution, serializedSolution.data());
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Solution failed to be serialized");

    std::string base64edSolution =
        base64Encode(serializedSolution.data(), serializedSolution.size());

    return {
        {GraphPatternExecutor::JsonFields::Name, name},
        {GraphExecutorFind20::JsonFields::Solution, base64edSolution},
        {GraphExecutorFind20::JsonFields::Id2ArgumentMap, id2ArgumentMap},
    };
}

GraphExecutorFind20::GraphExecutorFind20(const nlohmann::json& json)
{
    auto base64edSolution   = json.at(GraphExecutorFind20::JsonFields::Solution).get<std::string>();
    auto serializedSolution = base64Decode(base64edSolution);

    miopenSolution_t solutionDescriptor;
    auto status = miopenLoadSolution(
        &solutionDescriptor, serializedSolution.data(), serializedSolution.size());
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Failed to deserialize Solution");

    // Ensure miopenDestroySolution() will be called
    std::unique_ptr<miopenSolution_t, std::function<void(miopenSolution_t*)>>
        exceptionSafeSolutionStore(&solutionDescriptor,
                                   [](miopenSolution_t* sol) { miopenDestroySolution(*sol); });

    mSolution = std::move(deref(solutionDescriptor));

    auto id2ArgumentMap = json.at(GraphExecutorFind20::JsonFields::Id2ArgumentMap)
                              .get<std::map<int64_t, miopenTensorArgumentId_t>>();

    mTensorInfoMap = std::make_shared<TensorInfoMap>();
    for(const auto [tensorId, argumentId] : id2ArgumentMap)
    {
        mTensorInfoMap->try_emplace(tensorId, argumentId, nullptr);
    }
}

void GraphExecutorFind20::execute(miopenHandle_t handle, const VariantPack& vpk)
{

    std::vector<miopenTensorArgument_t> tens_args;

    auto num = vpk.getTensorIds().size();
    assert(num == vpk.getDataPtrs().size());

    /// \todo  verify that variant pack has all the expected input and output
    /// tensors --amberhassaan May, 2024
    for(std::size_t i = 0; i < num; ++i)
    {
        auto tens_id  = vpk.getTensorIds()[i];
        auto* gpu_ptr = vpk.getDataPtrs()[i];
        assert(gpu_ptr);

        auto it = mTensorInfoMap->find(tens_id);
        MIOPEN_THROW_IF(it == mTensorInfoMap->cend(),
                        "couldn't find a variant pack tensor id in the map");

        auto& v = it->second;

        /// \todo use this code with C++20 --amberhassaan May, 2024
        /*
        miopenTensorArgument_t targ{
          .id = v.mEnumId,
          // .descriptor = &(v.mTensDesc),
          .descriptor = nullptr,
          .buffer = gpu_ptr
        };
        */
        miopenTensorArgument_t targ{};
        targ.id         = v.mEnumId;
        targ.descriptor = nullptr;
        targ.buffer     = gpu_ptr;

        tens_args.emplace_back(targ);
    }

    auto s = miopenRunSolution(handle,
                               &mSolution,
                               tens_args.size(),
                               tens_args.data(),
                               vpk.getWorkspace(),
                               getWorkspaceSize());

    MIOPEN_THROW_IF(s != miopenStatusSuccess, "Run Solution failed");
    if(s == miopenStatusSuccess)
    {
        MIOPEN_LOG_I2("Graph API Find 2.0 Solution Ran");
    }
}

void to_json(nlohmann::json& json, const Engine& engine)
{
    MIOPEN_THROW_IF(!engine.mExecutor, "Cannot serialize an Engine without an Executor");

    json = nlohmann::json{
        {Engine::JsonFields::Executor, engine.mExecutor->getJson()},
        {Engine::JsonFields::GlobalIndex, engine.mGlobalIndex},
        {Engine::JsonFields::SmCount, engine.mSmCount},
    };
}

void from_json(const nlohmann::json& json, Engine& engine)
{
    static const std::map<
        std::string,
        std::function<std::shared_ptr<GraphPatternExecutor>(const nlohmann::json& json)>>
        name2Maker{
            {GraphExecutorFind20::name,
             std::make_shared<GraphExecutorFind20, const nlohmann::json&>},
            {ConvBiasResAddActivForwardExecutor::name,
             std::make_shared<ConvBiasResAddActivForwardExecutor, const nlohmann::json&>},
        };

    auto jExecutor    = json.at(Engine::JsonFields::Executor);
    auto executorName = jExecutor.at(GraphPatternExecutor::JsonFields::Name).get<std::string>();
    auto maker        = name2Maker.at(executorName);

    engine.mExecutor = maker(jExecutor);
    engine.mGraph    = nullptr;
    json.at(Engine::JsonFields::GlobalIndex).get_to(engine.mGlobalIndex);
    json.at(Engine::JsonFields::SmCount).get_to(engine.mSmCount);
}

EngineBuilder& EngineBuilder::setGraph(OpGraph* g)
{
    assert(g);
    mGraph    = checkPtr(g);
    mGraphSet = true;
    return *this;
}

EngineBuilder& EngineBuilder::setGlobalIndex(int64_t globalIndex)
{
    MIOPEN_THROW_IF(globalIndex < 0, "globalIndex must be >= 0");
    mGlobalIndex = globalIndex;
    mIndexSet    = true;
    return *this;
}

EngineBuilder& EngineBuilder::setSmCount(int32_t smCount)
{
    MIOPEN_THROW_IF(smCount <= 0, "SM count must be positive");
    mSmCount = smCount;
    return *this;
}

EngineBuilder& EngineBuilder::setExecutor(const std::shared_ptr<GraphPatternExecutor>& e)
{
    assert(e.get());
    mExecutor = e;
    mExecSet  = true;
    return *this;
}

Engine EngineBuilder::build()
{
    MIOPEN_THROW_IF(!mGraphSet || !mExecSet || !mIndexSet,
                    "must set graph, index and executor attributes");
    Engine e;
    e.mGraph       = mGraph;
    e.mGlobalIndex = mGlobalIndex;
    e.mExecutor    = mExecutor;
    e.mSmCount     = mSmCount;
    return e;
}

void BackendEngineDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_ENGINE_OPERATION_GRAPH:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t& apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendOperationGraphDescriptor& operationGraphDescriptor =
                dynamic_cast<BackendOperationGraphDescriptor&>(backendDescriptor);
            mBuilder.setGraph(operationGraphDescriptor.getOperationGraph());
            mOpGraphDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_GLOBAL_INDEX:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder.setGlobalIndex(*static_cast<int64_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET:
        if(attributeType == MIOPEN_TYPE_INT32 && elementCount == 1)
        {
            mBuilder.setSmCount(*static_cast<int32_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendEngineDescriptor::finalize()
{
    if(mFinalized || mBuilder.mGraph == nullptr)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    const auto& engines = mBuilder.mGraph->getEngines();

    if(static_cast<size_t>(mBuilder.mGlobalIndex) >= engines.size())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    const auto& candidate_engine = engines.at(mBuilder.mGlobalIndex);
    mBuilder.setExecutor(candidate_engine.getExecutor());
    mEngine = mBuilder.build();

    mFinalized = true;
}

void BackendEngineDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_ENGINE_OPERATION_GRAPH:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mOpGraphDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_GLOBAL_INDEX:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mEngine.getGlobalIndex();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET:
        if(attributeType == MIOPEN_TYPE_INT32 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int32_t*>(arrayOfElements) = mEngine.getSmCount();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case MIOPEN_ATTR_ENGINE_KNOB_INFO:
    case MIOPEN_ATTR_ENGINE_LAYOUT_INFO:
    case MIOPEN_ATTR_ENGINE_NUMERICAL_NOTE:
        /// \todo figure out what we can return here --Sergei May, 2024
        *elementCount = 0;
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
