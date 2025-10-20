#!/bin/bash
# Run all training tests in sequence
#
# Usage:
#   bash scripts/run_all_tests.sh

set -e  # Exit on error

echo "================================================================================"
echo "RUNNING ALL TRAINING TESTS"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to repo root
cd "$(dirname "$0")/.."

# # Test 1: Hard examples integration (quick)
# echo "================================================================================"
# echo "TEST 1: Hard Examples Integration (Quick)"
# echo "================================================================================"
# if uv run scripts/test_hard_examples_integration.py --steps 50 --batch_size 16; then
#     echo -e "${GREEN}✓ TEST 1 PASSED${NC}"
#     TEST1_RESULT="PASS"
# else
#     echo -e "${RED}✗ TEST 1 FAILED${NC}"
#     TEST1_RESULT="FAIL"
# fi
# echo ""

# # Test 2: Hard examples integration (stress test)
# echo "================================================================================"
# echo "TEST 2: Hard Examples Integration (Stress Test)"
# echo "================================================================================"
# if uv run scripts/test_hard_examples_integration.py --steps 200 --batch_size 32 --buffer_size 100; then
#     echo -e "${GREEN}✓ TEST 2 PASSED${NC}"
#     TEST2_RESULT="PASS"
# else
#     echo -e "${RED}✗ TEST 2 FAILED${NC}"
#     TEST2_RESULT="FAIL"
# fi
# echo ""

# Test 3: Mock training (short)
echo "================================================================================"
echo "TEST 3: Mock Training (Short)"
echo "================================================================================"
if uv run scripts/test_training_mock.py --num_train_steps 50 --log_interval 10; then
    echo -e "${GREEN}✓ TEST 3 PASSED${NC}"
    TEST3_RESULT="PASS"
else
    echo -e "${RED}✗ TEST 3 FAILED${NC}"
    TEST3_RESULT="FAIL"
fi
echo ""

# Test 4: Mock training (longer with hard examples)
echo "================================================================================"
echo "TEST 4: Mock Training (Longer with Hard Examples)"
echo "================================================================================"
if uv run scripts/test_training_mock.py \
    --num_train_steps 150 \
    --log_interval 10 \
    --hard_example_log_interval 50 \
    --max_hard_examples_buffer 50; then
    echo -e "${GREEN}✓ TEST 4 PASSED${NC}"
    TEST4_RESULT="PASS"
else
    echo -e "${RED}✗ TEST 4 FAILED${NC}"
    TEST4_RESULT="FAIL"
fi
echo ""

# Summary
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo -e "Test 1 (Hard Examples - Quick):      ${TEST1_RESULT}"
echo -e "Test 2 (Hard Examples - Stress):     ${TEST2_RESULT}"
echo -e "Test 3 (Mock Training - Short):      ${TEST3_RESULT}"
echo -e "Test 4 (Mock Training - Long):       ${TEST4_RESULT}"
echo ""

# Check if all passed
if [ "$TEST1_RESULT" = "PASS" ] && \
   [ "$TEST2_RESULT" = "PASS" ] && \
   [ "$TEST3_RESULT" = "PASS" ] && \
   [ "$TEST4_RESULT" = "PASS" ]; then
    echo -e "${GREEN}================================================================================"
    echo "🎉 ALL TESTS PASSED 🎉"
    echo -e "================================================================================${NC}"
    exit 0
else
    echo -e "${RED}================================================================================"
    echo "❌ SOME TESTS FAILED"
    echo -e "================================================================================${NC}"
    echo ""
    echo "See output above for details."
    echo "Check TEST_README.md for troubleshooting tips."
    exit 1
fi
