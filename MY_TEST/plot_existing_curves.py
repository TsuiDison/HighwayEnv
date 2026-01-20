#!/usr/bin/env python3
"""
从已有的训练运行中生成训练曲线图表
可以用于查看之前训练的结果
"""

import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import uniform_filter1d
import argparse

def find_events_file(run_dir):
    """查找TensorBoard events文件，支持子目录"""
    # 首先检查直接在run_dir中的events文件
    events_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if events_files:
        return events_files[0]
    
    # 如果没有找到，递归搜索子目录
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                return os.path.join(root, file)
    
    return None

def plot_training_curves(run_dir, scenario_name, output_dir):
    """从TensorBoard日志绘制训练曲线"""
    try:
        print(f"Reading TensorBoard data from: {run_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找events文件
        events_file = find_events_file(run_dir)
        if not events_file:
            print(f"⚠️  Warning: No TensorBoard events file found in {run_dir}")
            return 0
        
        print(f"Found events file: {events_file}")
        
        # 使用EventAccumulator读取TensorBoard日志
        event_acc = EventAccumulator(os.path.dirname(events_file))
        event_acc.Reload()
        
        # 获取标量数据
        tags = event_acc.Tags()
        if 'scalars' not in tags or not tags['scalars']:
            print(f"⚠️  Warning: No scalar data found in {events_file}")
            return 0
            
        scalars = tags['scalars']
        print(f"Available metrics: {scalars}")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Progress - {scenario_name.upper()}', fontsize=16, fontweight='bold')
        
        plot_count = 0
        
        # 1. 策略损失 (Policy Loss)
        if any('policy_loss' in key.lower() for key in scalars):
            policy_loss_key = [key for key in scalars if 'policy_loss' in key.lower()][0]
            policy_loss_events = event_acc.Scalars(policy_loss_key)
            steps = [event.step for event in policy_loss_events]
            values = [event.value for event in policy_loss_events]
            axes[0, 0].plot(steps, values, linewidth=2, color='#FF6B6B')
            axes[0, 0].set_title('Policy Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            plot_count += 1
        else:
            axes[0, 0].text(0.5, 0.5, 'No Policy Loss Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # 2. 价值函数损失 (Value Loss)
        if any('value_loss' in key.lower() for key in scalars):
            value_loss_key = [key for key in scalars if 'value_loss' in key.lower()][0]
            value_loss_events = event_acc.Scalars(value_loss_key)
            steps = [event.step for event in value_loss_events]
            values = [event.value for event in value_loss_events]
            axes[0, 1].plot(steps, values, linewidth=2, color='#4ECDC4')
            axes[0, 1].set_title('Value Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            plot_count += 1
        else:
            axes[0, 1].text(0.5, 0.5, 'No Value Loss Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. Clip Fraction
        if any('clip_fraction' in key.lower() for key in scalars):
            clip_frac_key = [key for key in scalars if 'clip_fraction' in key.lower()][0]
            clip_frac_events = event_acc.Scalars(clip_frac_key)
            steps = [event.step for event in clip_frac_events]
            values = [event.value for event in clip_frac_events]
            axes[1, 0].plot(steps, values, linewidth=2, color='#95E1D3')
            axes[1, 0].set_title('Clip Fraction', fontweight='bold')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Fraction')
            axes[1, 0].grid(True, alpha=0.3)
            plot_count += 1
        else:
            axes[1, 0].text(0.5, 0.5, 'No Clip Fraction Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. 平均奖励 (Mean Episode Reward)
        if any('ep_rew_mean' in key.lower() for key in scalars):
            reward_key = [key for key in scalars if 'ep_rew_mean' in key.lower()][0]
            reward_events = event_acc.Scalars(reward_key)
            steps = [event.step for event in reward_events]
            values = [event.value for event in reward_events]
            axes[1, 1].plot(steps, values, linewidth=2, color='#F7DC6F')
            axes[1, 1].fill_between(steps, values, alpha=0.3, color='#F7DC6F')
            axes[1, 1].set_title('Mean Episode Reward', fontweight='bold')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True, alpha=0.3)
            plot_count += 1
        else:
            axes[1, 1].text(0.5, 0.5, 'No Reward Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # 保存图表
        run_name = os.path.basename(run_dir)
        plot_path = os.path.join(output_dir, f'training_curves_{scenario_name}_{run_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to: {plot_path}")
        plt.close()
        
        # 绘制详细的奖励曲线
        if any('ep_rew_mean' in key.lower() for key in scalars):
            fig, ax = plt.subplots(figsize=(12, 6))
            reward_key = [key for key in scalars if 'ep_rew_mean' in key.lower()][0]
            reward_events = event_acc.Scalars(reward_key)
            steps = [event.step for event in reward_events]
            values = [event.value for event in reward_events]
            
            # 平滑曲线
            if len(values) > 10:
                smoothed = uniform_filter1d(values, size=10)
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
            reward_plot_path = os.path.join(output_dir, f'reward_curve_{scenario_name}_{run_name}.png')
            plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Reward curve saved to: {reward_plot_path}")
            plt.close()
        
        return plot_count
        
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from existing runs')
    parser.add_argument('--scenario', type=int, default=None, help='Scenario to plot (1/2/3)')
    parser.add_argument('--run', type=int, default=None, help='Specific run ID to plot')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    
    scenario_names = {1: "highway_merge", 2: "roundabout", 3: "parking"}
    
    # 查找所有可用的运行
    runs_found = 0
    
    for scenario_id, scenario_name in scenario_names.items():
        if args.scenario and scenario_id != args.scenario:
            continue
        
        base_dir = os.path.join(script_dir, "highway_ppo", scenario_name)
        if not os.path.exists(base_dir):
            continue
        
        run_dirs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
        
        if not run_dirs:
            print(f"No runs found for scenario {scenario_id} ({scenario_name})")
            continue
        
        for run_dir in run_dirs:
            run_id = os.path.basename(run_dir)
            
            if args.run and int(run_id.split('_')[-1]) != args.run:
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: Scenario {scenario_id} ({scenario_name}) - {run_id}")
            print(f"{'='*60}")
            
            count = plot_training_curves(run_dir, scenario_name, output_dir)
            runs_found += count
    
    if runs_found == 0:
        print("❌ No training runs found or no plots generated")
    else:
        print(f"\n✅ Successfully generated plots for {runs_found} runs")
        print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()
