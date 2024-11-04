#!/bin/bash

FOLDER_PATH="../../test/gtest"

declare -a unmatched_tests

find "$FOLDER_PATH" -type f -name "*.cpp" | while read -r file; do
  echo "Processing file: $file"
  
  TEST_P_LINES=$(grep -oP 'TEST_P\(\K\w+,\s*\w+' "$file" | awk -F, '{gsub(/ /,""); print $1 "." $2}')
  INSTANTIATE_TEST_LINES=$(grep -oP 'INSTANTIATE_TEST_SUITE_P\(\w+,\s*\K\w+' "$file")

  readarray -t TEST_P_ARRAY <<< "$TEST_P_LINES"
  readarray -t INSTANTIATE_TEST_ARRAY <<< "$INSTANTIATE_TEST_LINES"

  for test in "${TEST_P_ARRAY[@]}"; do
    suite=$(echo "$test" | cut -d. -f1)
    
    echo " checking test: ${test}"
    
    if [[ ! " ${INSTANTIATE_TEST_ARRAY[@]} " =~ " ${suite} " ]]; then
      unmatched_tests+=("$file: Test '$test' does not have a matching INSTANTIATE_TEST_SUITE_P.")
    fi
  done
done

echo "----------------------------"

if [[ ${#unmatched_tests[@]} -eq 0 ]]; then
  echo "All tests meet the criteria."
else
  echo -e "\nUnmatched tests:"
  for unmatched in "${unmatched_tests[@]}"; do
    echo "  $unmatched"
  done
fi
