#!/usr/bin/env python3
"""
批量训练所有三个场景的脚本
运行方式: python train_all_scenarios.py
"""

import subprocess
import sys
import time

def train_all_scenarios():
    scenarios = [
        {"id": 1, "name": "Highway + Merge (高速+汇入)"},
        {"id": 2, "name": "Roundabout (环岛)"},
        {"id": 3, "name": "Parking (停车)"},
    ]
    
    print("=" * 70)
    print("开始批量训练所有场景")
    print("=" * 70)
    print(f"总共需要训练 {len(scenarios)} 个场景")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(scenarios)}] 开始训练场景 {scenario['id']}: {scenario['name']}")
        print(f"{'=' * 70}\n")
        
        start_time = time.time()
        
        try:
            cmd = ["python", "train_highway_ppo.py", "--scenario", str(scenario['id'])]
            result = subprocess.run(cmd, check=True)
            
            elapsed_time = time.time() - start_time
            print(f"\n✓ 场景 {scenario['id']} 训练完成！耗时 {elapsed_time/60:.1f} 分钟")
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ 场景 {scenario['id']} 训练失败！错误码: {e.returncode}")
            return False
        except KeyboardInterrupt:
            print(f"\n! 训练被中断 (Ctrl+C)，已完成场景 {i-1}/{len(scenarios)}")
            return False
        
        # 场景间稍作等待
        if i < len(scenarios):
            print(f"\n等待 10 秒后开始下一个场景...")
            time.sleep(10)
    
    print(f"\n{'=' * 70}")
    print("✓ 所有场景训练完成！")
    print(f"{'=' * 70}\n")
    print("总结:")
    for scenario in scenarios:
        print(f"  ✓ 场景 {scenario['id']}: {scenario['name']}")
    
    print("\n下一步: 使用以下命令测试各场景")
    print("  python test_highway_ppo.py --scenario 1")
    print("  python test_highway_ppo.py --scenario 2")
    print("  python test_highway_ppo.py --scenario 3")
    
    return True

if __name__ == "__main__":
    try:
        success = train_all_scenarios()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)
