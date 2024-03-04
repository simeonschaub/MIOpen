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
#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "../conv2d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_WITH_MIOPENDRIVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPENDRIVER_MODE_CONV)

namespace miopendriver_regression_half {
void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

auto GetTestCases(void)
{
    const std::string cmd           = "MIOpenDriver ";
    const std::string drv_mode_conv = " poolfp16";

    // clang-format off
    return std::vector<std::string>{
        {cmd + drv_mode_conv + " -M 0 --input 1x64x41x40x70 -y 41 -x 40 -Z 70 -m avg -F 1 -t 1 -i 1"},
        {cmd + drv_mode_conv + " -M 0 --input 1x64x41x40x100 -y 4 -x 4 -Z 100 -m max -F 1 -t 1 -i 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class ConfigWithHalf_miopendriver_regression_half
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::Default>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

static bool SkipTest(const std::string& float_arg)
{
    if(IsTestSupportedForDevice() && miopen::IsEnabled(ENV(MIOPEN_TEST_WITH_MIOPENDRIVER)) &&
       (miopen::IsUnset(ENV(MIOPEN_TEST_ALL))       // standalone run
        || (miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) // or full tests enabled
            && miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG)) == float_arg)))
        return false;
    return true;
}

void Run2dDriver(const std::string& float_arg)
{
    if(SkipTest(float_arg))
    {
        GTEST_SKIP();
    }
    std::vector<std::string> params = ConfigWithHalf_miopendriver_regression_half::GetParam();

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });
        testing::internal::CaptureStderr();
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

} // namespace miopendriver_regression_half
using namespace miopendriver_regression_half;

TEST_P(ConfigWithHalf_miopendriver_regression_half, HalfTest_miopendriver_regression_half)
{
    Run2dDriver("--half");
};

INSTANTIATE_TEST_SUITE_P(MiopendriverRegressionHalf,
                         ConfigWithHalf_miopendriver_regression_half,
                         testing::Values(GetTestCases()));
