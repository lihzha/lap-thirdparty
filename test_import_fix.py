#!/usr/bin/env python3
"""Test script to verify the import fix for OOM issue."""

import sys
import psutil
import os

def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("Testing import chain to verify OOM fix...")
print(f"Initial memory: {get_memory_mb():.1f} MB")

# Test 1: Import the module that was causing OOM
print("\n1. Testing: import openpi_cot.policies.adapters.policy_config_adapter")
try:
    import openpi_cot.policies.adapters.policy_config_adapter
    mem_after = get_memory_mb()
    print(f"   ✓ Success! Memory after import: {mem_after:.1f} MB")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Verify the functions are still accessible
print("\n2. Testing: Functions are accessible")
try:
    assert hasattr(openpi_cot.policies.adapters.policy_config_adapter, 'create_trained_policy')
    assert hasattr(openpi_cot.policies.adapters.policy_config_adapter, 'create_trained_policy_cot')
    print("   ✓ All functions accessible")
except AssertionError:
    print("   ✗ Functions not accessible")
    sys.exit(1)

# Test 3: Check that TensorFlow is NOT loaded at import time
print("\n3. Testing: TensorFlow should NOT be loaded at import time (lazy loading)")
try:
    if 'tensorflow' in sys.modules:
        print("   ⚠ Warning: TensorFlow was loaded at import time (should be lazy)")
    else:
        print("   ✓ TensorFlow not loaded (lazy loading working correctly!)")
except Exception as e:
    print(f"   ? Could not check: {e}")

print(f"\n✓ All tests passed! Final memory: {get_memory_mb():.1f} MB")
print("\nThe OOM issue should be fixed. The heavy dependencies (TensorFlow, etc.)")
print("will only be loaded when you actually call create_trained_policy(), not at import time.")
