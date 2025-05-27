#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试采样策略的简单脚本
"""

import random
import numpy as np

def generate_sampling_indices(total_count, num_samples, strategy="random", tail_ratio=0.8):
    """
    根据指定策略生成采样索引
    
    Args:
        total_count: 总数据量
        num_samples: 需要采样的数量
        strategy: 采样策略 ("random", "tail_heavy", "exponential_decay")
        tail_ratio: tail_heavy策略中从后半部分采样的比例，或exponential_decay的衰减强度
    
    Returns:
        sorted list of indices
    """
    if strategy == "random":
        # 原有的随机采样
        return sorted(random.sample(range(total_count), num_samples))
    
    elif strategy == "tail_heavy":
        # 偏向后半部分的采样
        mid_point = total_count // 2
        
        # 计算从前半部分和后半部分分别采样多少个
        tail_samples = int(num_samples * tail_ratio)
        head_samples = num_samples - tail_samples
        
        # 确保不超出范围
        head_samples = min(head_samples, mid_point)
        tail_samples = min(tail_samples, total_count - mid_point)
        
        # 如果调整后的样本数不够，从另一部分补充
        if head_samples + tail_samples < num_samples:
            remaining = num_samples - head_samples - tail_samples
            if mid_point > head_samples:
                # 从前半部分补充
                head_samples = min(head_samples + remaining, mid_point)
            else:
                # 从后半部分补充
                tail_samples = min(tail_samples + remaining, total_count - mid_point)
        
        indices = []
        
        # 从前半部分采样
        if head_samples > 0:
            head_indices = random.sample(range(0, mid_point), head_samples)
            indices.extend(head_indices)
        
        # 从后半部分采样
        if tail_samples > 0:
            tail_indices = random.sample(range(mid_point, total_count), tail_samples)
            indices.extend(tail_indices)
        
        return sorted(indices)
    
    elif strategy == "exponential_decay":
        # 从后往前指数衰减的采样分布
        # tail_ratio 控制衰减强度，值越大衰减越快（更偏向后面）
        decay_rate = tail_ratio * 5  # 将0-1的范围映射到0-5的衰减率
        
        # 计算每个位置的权重（从后往前衰减）
        weights = []
        for i in range(total_count):
            # 位置越靠后权重越大
            position_from_end = total_count - 1 - i
            weight = np.exp(-decay_rate * position_from_end / total_count)
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 根据权重进行采样
        indices = np.random.choice(total_count, size=num_samples, replace=False, p=weights)
        return sorted(indices.tolist())
    
    elif strategy == "linear_decay":
        # 从后往前线性衰减的采样分布
        # tail_ratio 控制衰减强度
        
        # 计算每个位置的权重（线性衰减）
        weights = []
        for i in range(total_count):
            # 位置越靠后权重越大
            position_from_end = total_count - 1 - i
            # 线性衰减：最后位置权重为1，前面按比例递减
            weight = 1.0 - tail_ratio * position_from_end / total_count
            weight = max(weight, 0.01)  # 确保最小权重不为0
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 根据权重进行采样
        indices = np.random.choice(total_count, size=num_samples, replace=False, p=weights)
        return sorted(indices.tolist())
    
    else:
        raise ValueError(f"未知的采样策略: {strategy}")

def test_sampling():
    """测试采样策略"""
    total_count = 10000
    num_samples = 1000
    
    print("测试采样策略...")
    print(f"总数据量: {total_count}")
    print(f"采样数量: {num_samples}")
    print("="*50)
    
    # 测试随机采样
    print("1. 随机采样策略:")
    random_indices = generate_sampling_indices(total_count, num_samples, "random")
    analyze_distribution(random_indices, total_count, num_samples)
    print()
    
    # 测试tail_heavy采样 (80%后半部分)
    print("2. tail_heavy采样策略 (80%后半部分):")
    tail_heavy_indices = generate_sampling_indices(total_count, num_samples, "tail_heavy", 0.8)
    analyze_distribution(tail_heavy_indices, total_count, num_samples)
    print()
    
    # 测试指数衰减采样
    print("3. 指数衰减采样策略 (decay_rate=0.8):")
    exp_decay_indices = generate_sampling_indices(total_count, num_samples, "exponential_decay", 0.8)
    analyze_distribution(exp_decay_indices, total_count, num_samples)
    print()
    
    # 测试线性衰减采样
    print("4. 线性衰减采样策略 (decay_rate=0.8):")
    linear_decay_indices = generate_sampling_indices(total_count, num_samples, "linear_decay", 0.8)
    analyze_distribution(linear_decay_indices, total_count, num_samples)
    print()
    
    # 测试不同衰减强度的指数衰减
    print("5. 不同衰减强度的指数衰减:")
    for decay in [0.3, 0.5, 0.8, 0.9]:
        print(f"   衰减强度 {decay}:")
        indices = generate_sampling_indices(total_count, num_samples, "exponential_decay", decay)
        analyze_distribution(indices, total_count, num_samples, show_details=False)
    print()
    
    # 可视化分布
    print("6. 分布可视化 (显示每个区间的采样密度):")
    visualize_distribution(total_count, num_samples)

def analyze_distribution(indices, total_count, num_samples, show_details=True):
    """分析采样分布"""
    # 按区间统计
    quarters = [
        (0, total_count // 4, "第1四分位"),
        (total_count // 4, total_count // 2, "第2四分位"), 
        (total_count // 2, 3 * total_count // 4, "第3四分位"),
        (3 * total_count // 4, total_count, "第4四分位")
    ]
    
    for start, end, name in quarters:
        count = sum(1 for idx in indices if start <= idx < end)
        percentage = count / num_samples * 100
        print(f"   {name}({start}-{end-1}): {count}个 ({percentage:.1f}%)")
    
    if show_details:
        print(f"   前10个索引: {indices[:10]}")
        print(f"   后10个索引: {indices[-10:]}")
        print(f"   最小索引: {min(indices)}, 最大索引: {max(indices)}")
        print(f"   平均索引: {np.mean(indices):.1f}")

def visualize_distribution(total_count, num_samples):
    """可视化不同采样策略的分布"""
    strategies = [
        ("random", 0.5, "随机"),
        ("exponential_decay", 0.5, "指数衰减(0.5)"),
        ("exponential_decay", 0.8, "指数衰减(0.8)"),
        ("linear_decay", 0.8, "线性衰减(0.8)")
    ]
    
    bins = 20  # 将数据分为20个区间
    bin_size = total_count // bins
    
    for strategy, param, name in strategies:
        indices = generate_sampling_indices(total_count, num_samples, strategy, param)
        
        # 统计每个区间的采样数量
        bin_counts = [0] * bins
        for idx in indices:
            bin_idx = min(idx // bin_size, bins - 1)
            bin_counts[bin_idx] += 1
        
        print(f"   {name}:")
        # 简单的文本图表
        max_count = max(bin_counts) if bin_counts else 1
        for i, count in enumerate(bin_counts):
            bar_length = int(count / max_count * 30) if max_count > 0 else 0
            bar = "█" * bar_length
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size - 1, total_count - 1)
            print(f"     [{start_idx:5d}-{end_idx:5d}]: {count:3d} {bar}")
        print()

if __name__ == "__main__":
    test_sampling() 