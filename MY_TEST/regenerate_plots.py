#!/usr/bin/env python3
"""
重新生成已有训练的图表
Regenerate plots from existing training runs
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import uniform_filter1d

def plot_training_curves(run_dir, scenario_name, script_dir):
    """从TensorBoard日志绘制训练曲线"""
    try:
        # 创建输出目录
        plot_dir = script_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # 使用EventAccumulator读取TensorBoard日志
        event_acc = EventAccumulator(run_dir)
        event_acc.Reload()
        
        print(f"\n📊 Reading TensorBoard logs from: {run_dir}")
        
        # 获取所有标签
        tags = event_acc.Tags()
        print(f"Available tags: {tags}")
        
        # 获取标量数据
        if 'scalars' not in tags or not tags['scalars']:
            print(f"⚠️  Warning: No scalar data found in {run_dir}")
            return
            
        scalars = tags['scalars']
        print(f"Scalar keys found: {scalars}")
        
        # 提取所有数据
        data_dict = {}
        for scalar_name in scalars:
            try:
                events = event_acc.Scalars(scalar_name)
                if events:
                    steps = np.array([e.step for e in events])
                    values = np.array([e.value for e in events])
                    data_dict[scalar_name] = (steps, values)
                    print(f"  ✓ {scalar_name}: {len(events)} data points")
            except Exception as e:
                print(f"  ✗ Failed to read {scalar_name}: {e}")
        
        if not data_dict:
            print(f"⚠️  Warning: No scalar data could be extracted from {run_dir}")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Progress - {scenario_name.upper()}', fontsize=16, fontweight='bold')
        
        plot_count = 0
        
        # 1. 策略损失 (Policy Loss)
        policy_loss_keys = [k for k in data_dict.keys() if 'policy_loss' in k.lower()]
        if policy_loss_keys:
            key = policy_loss_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[0, 0].plot(steps, values, linewidth=2, color='#FF6B6B')
                axes[0, 0].set_title('Policy Loss', fontweight='bold')
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Policy Loss")
        else:
            axes[0, 0].text(0.5, 0.5, 'No Policy Loss Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # 2. 价值函数损失 (Value Loss)
        value_loss_keys = [k for k in data_dict.keys() if 'value_loss' in k.lower()]
        if value_loss_keys:
            key = value_loss_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[0, 1].plot(steps, values, linewidth=2, color='#4ECDC4')
                axes[0, 1].set_title('Value Loss', fontweight='bold')
                axes[0, 1].set_xlabel('Training Steps')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Value Loss")
        else:
            axes[0, 1].text(0.5, 0.5, 'No Value Loss Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. 总损失 (Entropy Loss or Clip Fraction)
        entropy_keys = [k for k in data_dict.keys() if 'entropy_loss' in k.lower()]
        clip_keys = [k for k in data_dict.keys() if 'clip_fraction' in k.lower()]
        
        if entropy_keys:
            key = entropy_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[1, 0].plot(steps, values, linewidth=2, color='#95E1D3')
                axes[1, 0].set_title('Entropy Loss', fontweight='bold')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Entropy Loss")
        elif clip_keys:
            key = clip_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[1, 0].plot(steps, values, linewidth=2, color='#95E1D3')
                axes[1, 0].set_title('Clip Fraction', fontweight='bold')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Fraction')
                axes[1, 0].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Clip Fraction")
        else:
            axes[1, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. 平均奖励 (Mean Episode Reward)
        reward_keys = [k for k in data_dict.keys() if 'ep_rew_mean' in k.lower()]
        if reward_keys:
            key = reward_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[1, 1].plot(steps, values, linewidth=2, color='#F7DC6F')
                axes[1, 1].fill_between(steps, values, alpha=0.3, color='#F7DC6F')
                axes[1, 1].set_title('Mean Episode Reward', fontweight='bold')
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Mean Episode Reward")
        else:
            axes[1, 1].text(0.5, 0.5, 'No Reward Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # 保存图表到当前目录
        plot_path = os.path.join(plot_dir, f'training_curves_{scenario_name}_run_{os.path.basename(run_dir)}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to: {plot_path}")
        
        plt.close()
        
        # 绘制详细的奖励曲线
        reward_keys = [k for k in data_dict.keys() if 'ep_rew_mean' in k.lower()]
        if reward_keys:
            fig, ax = plt.subplots(figsize=(12, 6))
            key = reward_keys[0]
            steps, values = data_dict[key]
            
            if len(steps) > 0:
                # 平滑曲线
                if len(values) > 20:
                    smoothed = uniform_filter1d(values, size=20)
                elif len(values) > 5:
                    smoothed = uniform_filter1d(values, size=5)
                else:
                    smoothed = values
                
                ax.plot(steps, values, alpha=0.3, color='#4ECDC4', label='Raw Reward')
                ax.plot(steps, smoothed, linewidth=2.5, color='#FF6B6B', label='Smoothed Reward')
                ax.fill_between(steps, values, alpha=0.1, color='#4ECDC4')
                ax.set_title(f'Episode Reward Over Training - {scenario_name.upper()}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Training Steps', fontsize=12)
                ax.set_ylabel('Mean Episode Reward', fontsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                reward_plot_path = os.path.join(plot_dir, f'reward_curve_{scenario_name}_run_{os.path.basename(run_dir)}.png')
                plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
                print(f"✅ Reward curve saved to: {reward_plot_path}")
                plt.close()
        
        print(f"📊 Total plots generated: {plot_count + 1 if reward_keys else plot_count}")
        
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 获取当前脚本的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义场景映射
    scenarios = {
        'highway_merge': 'highway_merge',
        'parking': 'parking',
        'roundabout': 'roundabout',
    }
    
    # 扫描所有已有的训练目录
    print("🔍 Scanning for existing training runs...\n")
    
    base_dir = os.path.join(script_dir, 'highway_ppo')
    
    for scenario_key, scenario_name in scenarios.items():
        scenario_dir = os.path.join(base_dir, scenario_key)
        
        if not os.path.exists(scenario_dir):
            print(f"⚠️  Directory not found: {scenario_dir}")
            continue
        
        # 查找所有run目录
        run_dirs = sorted(glob.glob(os.path.join(scenario_dir, 'run_*', 'PPO_1')))
        
        if not run_dirs:
            print(f"⚠️  No training runs found in: {scenario_dir}")
            continue
        
        print(f"📁 Found {scenario_key}: {len(run_dirs)} run(s)")
        
        for run_dir in run_dirs:
            run_name = os.path.basename(os.path.dirname(run_dir))
            print(f"\n{'=' * 60}")
            print(f"Processing: {scenario_name} / {run_name}")
            print(f"{'=' * 60}")
            
            plot_training_curves(run_dir, scenario_name, script_dir)
    
    print(f"\n{'=' * 60}")
    print("✅ All plots regenerated successfully!")
    print(f"{'=' * 60}")
