#!/usr/bin/env python3
"""
Quick test to verify SystemMonitor functionality without running full training.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from AI_DEV.dataset_trainer import SystemMonitor

def test_system_monitor():
    """Test SystemMonitor class."""
    print("Testing SystemMonitor class...\n")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device("cuda")
    print(f"✓ Using device: {device}")
    
    # Create monitor
    monitor = SystemMonitor(device, enabled=True)
    print("✓ SystemMonitor initialized")
    
    # Get stats
    stats = monitor.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"   GPU Util: {stats.get('gpu_util', -1):.1f}%")
    print(f"   VRAM: {stats.get('vram_used_gb', -1):.2f} / {stats.get('vram_total_gb', -1):.2f} GB ({stats.get('vram_percent', -1):.1f}%)")
    print(f"   CPU Util: {stats.get('cpu_percent', -1):.1f}%")
    print(f"   RAM: {stats.get('ram_used_gb', -1):.2f} / {stats.get('ram_total_gb', -1):.2f} GB ({stats.get('ram_percent', -1):.1f}%)")
    
    # Format string
    formatted = monitor.format_stats()
    print(f"\n📝 Formatted: {formatted}")
    
    # Test with some GPU work
    print("\n🔥 Creating GPU tensor to increase utilization...")
    x = torch.randn(10000, 10000, device=device)
    y = torch.matmul(x, x.T)
    torch.cuda.synchronize()
    
    stats2 = monitor.get_stats()
    print(f"\n📊 After GPU Work:")
    print(f"   VRAM: {stats2.get('vram_used_gb', -1):.2f} / {stats2.get('vram_total_gb', -1):.2f} GB ({stats2.get('vram_percent', -1):.1f}%)")
    print(f"   Formatted: {monitor.format_stats()}")
    
    print("\n✅ SystemMonitor test passed!")
    return True

def check_config():
    """Check current training configuration."""
    from AI_DEV.dataset_trainer import Config
    
    print("\n" + "="*70)
    print("Current Training Configuration")
    print("="*70)
    
    cfg = Config()
    
    print(f"\n📦 Batch Configuration:")
    print(f"   batch_size: {cfg.batch_size}")
    print(f"   val_batch_size: {cfg.val_batch_size}")
    
    print(f"\n⚙️  DataLoader Configuration:")
    print(f"   num_workers: {cfg.num_workers}")
    print(f"   prefetch_factor: {cfg.prefetch_factor}")
    print(f"   persistent_workers: {cfg.persistent_workers}")
    print(f"   pin_memory: {cfg.pin_memory}")
    
    print(f"\n🚀 Optimization Settings:")
    print(f"   use_compile: {cfg.use_compile}")
    print(f"   compile_mode: {cfg.compile_mode}")
    print(f"   use_channels_last: {cfg.use_channels_last}")
    
    print(f"\n📈 Profiling:")
    print(f"   profile_performance: {cfg.profile_performance}")
    print(f"   profile_gpu_util: {cfg.profile_gpu_util}")
    
    # Calculate prefetch queue size
    total_prefetch = cfg.num_workers * cfg.prefetch_factor
    print(f"\n💾 Prefetch Queue:")
    print(f"   Total batches queued: {total_prefetch} batches")
    print(f"   Queue memory (approx): {total_prefetch * cfg.batch_size * 128 * 16 * 4 / (1024**3):.2f} GB")
    print(f"   (Assuming mel spectrograms: 128 mels × 16 frames × 4 bytes)")

if __name__ == "__main__":
    print("="*70)
    print("GPU Utilization Optimization Test Suite")
    print("="*70)
    
    # Test monitoring
    success = test_system_monitor()
    
    if success:
        # Show config
        check_config()
        
        print("\n" + "="*70)
        print("✅ All tests passed! Ready to run training with monitoring.")
        print("="*70)
        print("\nRun training with:")
        print("  cd /DEVELOPMENT/ROOT_AILH/REPOS/AILH_MASTER/AI_DEV")
        print("  python train_binary.py")
        print("\nMonitor GPU in another terminal:")
        print("  watch -n 1 nvidia-smi")
        print("="*70)
    else:
        print("\n❌ Tests failed")
        sys.exit(1)
