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

#include "bn.hpp"
/* typename XDataType,
   typename DxDataType,
   typename DyDataType,
   typename AccDataType,
   typename ScaleDataType,
   typename DscaleDbiasDataType,
   typename MeanVarDataType> */

struct GPU_BN_CK_BWD_Large_FP16_2D
    : BNBwdTest<half_float::half, float, float, float, half_float::half, float, float, BN2DTestCase>
{
};

struct GPU_BN_OCL_BWD_Large_FP16_2D : BNBwdTest<half_float::half,
                                                half_float::half,
                                                half_float::half,
                                                float,
                                                float,
                                                float,
                                                float,
                                                BN2DTestCase>
{
};

struct GPU_BN_OCL_BWD_Large_FP16_3D : BNBwdTest<half_float::half,
                                                half_float::half,
                                                half_float::half,
                                                float,
                                                float,
                                                float,
                                                float,
                                                BN3DTestCase>
{
};

struct GPU_BN_CK_BWD_Large_BFP16_2D
    : BNBwdTest<bfloat16, float, float, float, bfloat16, float, float, BN2DTestCase>
{
};

struct GPU_BN_OCL_BWD_Large_BFP16_2D
    : BNBwdTest<bfloat16, bfloat16, bfloat16, float, float, float, float, BN2DTestCase>
{
};

struct GPU_BN_OCL_BWD_Large_BFP16_3D
    : BNBwdTest<bfloat16, bfloat16, bfloat16, float, float, float, float, BN3DTestCase>
{
};

struct GPU_BN_BWD_Small_FP32_2D
    : BNBwdTest<float, float, float, float, float, float, float, BN2DTestCase>
{
};

struct GPU_BN_BWD_Large_FP32_2D
    : BNBwdTest<float, float, float, float, float, float, float, BN2DTestCase>
{
};

struct GPU_BN_BWD_Large_FP32_3D
    : BNBwdTest<float, float, float, float, float, float, float, BN3DTestCase>
{
};

struct GPU_BN_BWD_Small_FP64_2D
    : BNBwdTest<double, double, double, double, double, double, double, BN2DTestCase>
{
};

struct GPU_BN_BWD_Large_FP64_2D
    : BNBwdTest<double, double, double, double, double, double, double, BN2DTestCase>
{
};

// fp16
TEST_P(GPU_BN_CK_BWD_Large_FP16_2D, DISABLED_BnV2LargeBWDCKfp16_2D) {}
TEST_P(GPU_BN_OCL_BWD_Large_FP16_2D, BnV2LargeBWDOCLfp16_2D) {}
TEST_P(GPU_BN_OCL_BWD_Large_FP16_3D, BnV2LargeBWDOCLfp16_3D) {}

// // // bfp16
TEST_P(GPU_BN_CK_BWD_Large_BFP16_2D, DISABLED_BnV2LargeBWDCKbfp16_2D) {}
TEST_P(GPU_BN_OCL_BWD_Large_BFP16_2D, BnV2LargeBWDOCLbfp16_2D) {}
TEST_P(GPU_BN_OCL_BWD_Large_BFP16_3D, BnV2LargeBWDOCLbfp16_3D) {}

// fp32 (float)
TEST_P(GPU_BN_BWD_Small_FP32_2D, BnV1SmallBWDCKfp32_2D) {}
TEST_P(GPU_BN_BWD_Large_FP32_2D, BnV2LargeBWDCKfp32_2D) {}
TEST_P(GPU_BN_BWD_Large_FP32_3D, BnV2LargeBWDCKfp32_3D) {}

// fp64
TEST_P(GPU_BN_BWD_Small_FP64_2D, DISABLED_BnV1SmallBWDCKfp64_2D) {}
TEST_P(GPU_BN_BWD_Large_FP64_2D, DISABLED_BnV2LargeBWDCKfp64_2D) {}

// fp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_CK_BWD_Large_FP16_2D,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_OCL_BWD_Large_FP16_2D,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_OCL_BWD_Large_FP16_3D,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN3DTestCase>());

// // bfp16
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_CK_BWD_Large_BFP16_2D,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_OCL_BWD_Large_BFP16_2D,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_OCL_BWD_Large_BFP16_3D,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN3DTestCase>());

// // fp32
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Small_FP32_2D,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV1})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Large_FP32_2D,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Large_FP32_3D,
                         testing::Combine(testing::ValuesIn(Network3DBN<BN3DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCDHW}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN3DTestCase>());
// fp64
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Small_FP64_2D,
                         testing::Combine(testing::ValuesIn(Network2DSmall<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV1})),
                         TestNameGenerator<BN2DTestCase>());

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BN_BWD_Large_FP64_2D,
                         testing::Combine(testing::ValuesIn(Network2DLarge<BN2DTestCase>()),
                                          testing::ValuesIn({miopenTensorNCHW, miopenTensorNHWC}),
                                          testing::ValuesIn({testBNAPIV2})),
                         TestNameGenerator<BN2DTestCase>());
