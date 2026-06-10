import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
from functools import reduce
from collections import Counter

def logistic_map(N, r, x0, M=1000):
    """
    Logistic 映射序列生成
    x(n+1) = r * x(n) * (1 - x(n))
    """
    x = x0
    for _ in range(M):
        x = r * x * (1.0 - x)
    
    seq = np.zeros(N)
    for i in range(N):
        x = r * x * (1.0 - x)
        seq[i] = x
    return seq

def tent_map(N, mu, x0, M=1000):
    """
    Tent 映射序列生成
    x(n+1) = mu * x(n) (x < 0.5) else mu * (1 - x(n))
    """
    x = x0
    for _ in range(M):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1.0 - x)
            
    seq = np.zeros(N)
    for i in range(N):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1.0 - x)
        seq[i] = x
    return seq

def henon_map(N, a, b, x0, y0, M=1000):
    """
    Henon 映射序列生成
    x(n+1) = 1 - a * x(n)^2 + y(n)
    y(n+1) = b * x(n)
    提取 x 维度序列
    """

    max_attempts = 10
    for attempt in range(max_attempts):
        x, y = x0, y0
        try:
            for _ in range(M):
                x_next = 1.0 - a * x**2 + y
                y_next = b * x
                x, y = x_next, y_next
                if not math.isfinite(x) or not math.isfinite(y):
                    raise OverflowError

            seq = np.zeros(N)
            for i in range(N):
                x_next = 1.0 - a * x**2 + y
                y_next = b * x
                x, y = x_next, y_next
                if not math.isfinite(x):
                    raise OverflowError
                seq[i] = x
            return seq
        except OverflowError:
            x0 = random.uniform(-0.5, 0.5)
            y0 = random.uniform(-0.5, 0.5)
            continue
    return np.zeros(N)

def generate_permutation_table(seq):
    """
    基于稳定排序构造置乱表 P
    保证 P[原始索引] = 新索引
    """
    sorted_indices = np.argsort(seq, kind='stable')
    N = len(seq)
    P = np.zeros(N, dtype=int)
    P[sorted_indices] = np.arange(N)
    return P

def decompose_cycles(P):
    """
    利用标记数组将置换分解为多个不相交的循环
    """
    N = len(P)
    visited = np.zeros(N, dtype=bool)
    cycles = []
    
    for i in range(N):
        if not visited[i]:
            cycle_length = 0
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cur = P[cur]
                cycle_length += 1
            cycles.append(cycle_length)
            
    return cycles

def calculate_order(cycles):
    """
    计算所有循环长度的最小公倍数(LCM)得到置换阶
    由于阶数可能极大，返回 math.log10(order)
    """
    order = reduce(math.lcm, cycles, 1)
    return math.log10(order)

def run_single_experiment(map_type, N):
    """执行单次实验并返回各项指标"""
    if map_type == 'Logistic':
        r = random.uniform(3.90, 3.999999)
        x0 = random.uniform(0.001, 0.999)
        seq = logistic_map(N, r, x0)
    elif map_type == 'Tent':
        mu = random.uniform(1.98, 1.999999)
        x0 = random.uniform(0.001, 0.999)
        seq = tent_map(N, mu, x0)
    else: # Henon
        a, b = 1.4, 0.3
        x0 = random.uniform(-0.5, 0.5)
        y0 = random.uniform(-0.5, 0.5)
        seq = henon_map(N, a, b, x0, y0)
        
    P = generate_permutation_table(seq)
    cycles = decompose_cycles(P)
    
    log_order = calculate_order(cycles)
    cycle_count = len(cycles)
    max_cycle = max(cycles)
    
    return log_order, cycle_count, max_cycle, cycles

def run_batch_experiments(N_list, repeat=50):
    """批量测试并统计均值"""
    maps = ['Logistic', 'Tent', 'Henon']
    results = {m: [] for m in maps}
    
    print(f"{'映射':<10} | {'N':<6} | {'平均 log10(阶)':<15} | {'平均循环数':<10} | {'平均最大循环'}")
    print("-" * 65)
    
    for m in maps:
        for N in N_list:
            log_orders = []
            cycle_counts = []
            max_cycles = []
            for _ in range(repeat):
                l_ord, c_count, m_cycle, _ = run_single_experiment(m, N)
                log_orders.append(l_ord)
                cycle_counts.append(c_count)
                max_cycles.append(m_cycle)
            
            avg_log_order = np.mean(log_orders)
            avg_cycle_count = np.mean(cycle_counts)
            avg_max_cycle = np.mean(max_cycles)
            
            results[m].append(avg_log_order)
            print(f"{m:<10} | {N:<6} | {avg_log_order:<15.4f} | {avg_cycle_count:<10.2f} | {avg_max_cycle:.2f}")
            
    return results

def test_sensitivity(N=1024):
    """初值敏感性实验"""
    epsilon = 1e-14
    maps = ['Logistic', 'Tent', 'Henon']
    diff_ratios = []
    
    # Logistic
    r = 3.99
    x0 = 0.123456789
    s1 = logistic_map(N, r, x0)
    s2 = logistic_map(N, r, x0 + epsilon)
    p1, p2 = generate_permutation_table(s1), generate_permutation_table(s2)
    diff_ratios.append(np.sum(p1 != p2) / N * 100)
    
    # Tent
    mu = 1.99
    x0 = 0.123456789
    s1 = tent_map(N, mu, x0)
    s2 = tent_map(N, mu, x0 + epsilon)
    p1, p2 = generate_permutation_table(s1), generate_permutation_table(s2)
    diff_ratios.append(np.sum(p1 != p2) / N * 100)
    
    # Henon
    a, b = 1.4, 0.3
    x0, y0 = 0.123456789, 0.123456789
    s1 = henon_map(N, a, b, x0, y0)
    s2 = henon_map(N, a, b, x0 + epsilon, y0)
    p1, p2 = generate_permutation_table(s1), generate_permutation_table(s2)
    diff_ratios.append(np.sum(p1 != p2) / N * 100)
    
    return maps, diff_ratios


def plot_average_order(N_list, results):
    plt.figure(figsize=(10, 6))
    markers = {'Logistic': 'o-', 'Tent': 's-', 'Henon': '^-'}
    for m in results:
        plt.plot(N_list, results[m], markers[m], label=m, linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('N (Log Scale)')
    plt.ylabel('Average log10(Order)')
    plt.title('Comparison of Chaos Maps: Average Permutation Order vs N')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cycle_distribution():
    _, _, _, cycles = run_single_experiment('Logistic', 1024)
    counter = Counter(cycles)
    lengths = [str(k) for k in sorted(counter.keys())]
    counts = [counter[int(k)] for k in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, color='tab:blue')
    plt.xlabel('Cycle Length')
    plt.ylabel('Count')
    plt.title('Example Cycle-Length Distribution (Logistic, N=1024)')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_sensitivity(maps, diff_ratios):
    plt.figure(figsize=(8, 6))
    plt.bar(maps, diff_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylim(99.0, 100.1) # 放大差异观察区
    plt.ylabel('Different Positions (%)')
    plt.title('Sensitivity Experiment: Permutation Difference Ratio')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    N_list = [256, 512, 1024, 2048, 4096, 8192]

    print("\n执行批量实验统计")
    results = run_batch_experiments(N_list, repeat=50)
    
    print("\n生成平均阶-N 曲线图")
    plot_average_order(N_list, results)
    
    print("\n生成典型样本循环长度分布图")
    plot_cycle_distribution()
    
    print("\n执行初值敏感性实验")
    maps, diff_ratios = test_sensitivity(N=1024)
    for m, ratio in zip(maps, diff_ratios):
        print(f"  {m} 映射位置改变比例: {ratio:.3f}%")
    plot_sensitivity(maps, diff_ratios)

if __name__ == "__main__":
    main()