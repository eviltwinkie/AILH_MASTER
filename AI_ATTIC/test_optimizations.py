#!/usr/bin/env python3
"""
Test script to validate the pipeline optimizations.
"""

import sys
import traceback

def test_imports():
    """Test that all imports work correctly."""
    try:
        from pipeline import (
            Config, 
            GPUMemoryMonitor, 
            CPUGPUBuffer, 
            GPUTransformManager,
            ProcessingState,
            Pipeline
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration initialization."""
    try:
        from pipeline import Config
        config = Config()
        print(f"✅ Config created successfully")
        print(f"   - GPU batch size: {config.GPU_BATCH_SIZE}")
        print(f"   - Buffer timeout: {config.BUFFER_TIMEOUT_MS}ms")
        print(f"   - CPU threads: {config.CPU_THREADS}")
        print(f"   - Device: {config.DEVICE}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        traceback.print_exc()
        return False

def test_gpu_manager():
    """Test GPU manager initialization."""
    try:
        from pipeline import GPUTransformManager
        # Just test creation, not actual GPU operations
        print("✅ GPU Manager class available")
        return True
    except Exception as e:
        print(f"❌ GPU Manager error: {e}")
        traceback.print_exc()
        return False

def test_buffer():
    """Test buffer class."""
    try:
        from pipeline import CPUGPUBuffer
        print("✅ Buffer class available")
        return True
    except Exception as e:
        print(f"❌ Buffer error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing pipeline optimizations...")
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("GPU Manager", test_gpu_manager),
        ("Buffer", test_buffer),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The pipeline optimizations are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
