import os
import re

# Path to the folder containing test files
FOLDER_PATH = "../../test/gtest"

# Ignore list: Add test names or file paths you want to exclude
#"../../test/gtest/ignore_this_test.cpp" or "graphapi_convolution.cpp"
IGNORE_LIST = {
    "CPU_MIOpenDriverRegressionBigTensorTest_FP32",
    "../../test/gtest/reduce_custom_fp32.cpp"# Exclude this specific test suite
}

# Valid enums and Regex for validation
VALID_HW_TYPES = {"CPU", "GPU"}
VALID_DATATYPES = {"FP8", "FP16", "FP32", "FP64", "BFP16", "BFP8", "I64", "I32", "I16", "I8", "NONE"}
TESTSUITE_REGEX = re.compile(
    r"^(CPU|GPU)_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_(" + "|".join(VALID_DATATYPES) + r")$"
)
TEST_P_REGEX = re.compile(r"TEST_P\(([^,]+),\s*([^)]+)\)")
INSTANTIATE_TEST_REGEX = re.compile(r"INSTANTIATE_TEST_SUITE_P\(([^,]+),\s*([^,]+),")
TEST_TYPE_REGEX = re.compile(r"^(Smoke|Full|Perf|Unit)([A-Za-z0-9]*)?$")


def analyze_tests(folder_path):
    invalid_tests = []
    unmatched_tests = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)

                if file_path in IGNORE_LIST:
                    print(f"Skipping ignored file: {file_path}")
                    continue

                with open(file_path, "r") as f:
                    content = f.read()

                # Extract TEST_P and INSTANTIATE_TEST_SUITE_P
                test_p_matches = TEST_P_REGEX.findall(content)
                instantiate_matches = INSTANTIATE_TEST_REGEX.findall(content)

                test_p_suites = {suite: info for suite, info in test_p_matches}
                instantiated_suites = {suite: test_type for test_type, suite in instantiate_matches}

                # Validate TEST_P suites
                for suite, info in test_p_suites.items():
                    if suite in IGNORE_LIST:
                        print(f"Skipping ignored test suite: {suite}")
                        continue

                    if not TESTSUITE_REGEX.match(suite):
                        invalid_tests.append(
                            f"{file_path}: Invalid TESTSUITE_NAME '{suite}' in TEST_P."
                        )
                        
                    if suite not in instantiated_suites:
                        unmatched_tests.append(
                            f"{file_path}: Test '{suite}.{info}' does not have a matching INSTANTIATE_TEST_SUITE_P."
                        )

                # Validate instantiated suites
                for suite, test_type in instantiated_suites.items():
                    if suite in IGNORE_LIST:
                        print(f"Skipping ignored instantiated suite: {suite}")
                        continue

                    if suite not in test_p_suites:
                        unmatched_tests.append(
                            f"{file_path}: INSTANTIATE_TEST_SUITE_P references non-existent TESTSUITE_NAME '{suite}'."
                        )
                    if not TEST_TYPE_REGEX.match(test_type):
                        invalid_tests.append(
                            f"{file_path}: Invalid TEST_TYPE '{test_type}' in INSTANTIATE_TEST_SUITE_P."
                        )

    return invalid_tests, unmatched_tests


def main():
    invalid_tests, unmatched_tests = analyze_tests(FOLDER_PATH)

    print("----------------------------")

    if not invalid_tests and not unmatched_tests:
        print("All tests meet the criteria.")
    else:
        if invalid_tests:
            print("\nInvalid tests:")
            for test in invalid_tests:
                print(f"  {test}")

        if unmatched_tests:
            print("\nUnmatched tests:")
            for test in unmatched_tests:
                print(f"  {test}")


if __name__ == "__main__":
    main()
