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

#include <miopen/env.hpp>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include <random.hpp>

class TestBase : public ::testing::Test
{
protected:
    std::map<std::string, std::string> original_env_vars;

    void SetUp() override
    {
        prng::reset_seed();
        save_env_vars();
    }

    void TearDown() override { restore_env_vars(); }

private:
    void save_env_vars()
    {
        std::vector<std::string> env_vars = {"CMAKE_CURRENT_BINARY_DIR",
                                             "MIOPEN_TEST_MLIR",
                                             "MIOPEN_TEST_COMPOSABLEKERNEL",
                                             "CODECOV_TEST",
                                             "MIOPEN_TEST_DBSYNC",
                                             "MIOPEN_TEST_CONV",
                                             "MIOPEN_TEST_DEEPBENCH",
                                             "MIOPEN_DEBUG_TUNING_ITERATIONS_MAX",
                                             "MIOPEN_TEST_WITH_MIOPENDRIVER"};

        for(const auto& var : env_vars)
        {
            auto value             = miopen::env::getEnvironmentVariable(var);
            original_env_vars[var] = value.value_or("");
        }
    }

    void restore_env_vars()
    {
        for(const auto& [key, value] : original_env_vars)
        {
            if(value.empty())
            {
                miopen::env::clearEnvironmentVariable(key);
            }
            else
            {
                miopen::env::setEnvironmentVariable(key, value);
            }
        }
    }
};
