import numpy as np
import h5py
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.patches import Rectangle, Patch


# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# --- 基础参数 ---
WORD_SIZE = 16
mask = 0xFFFF
REAL_MASTER_KEY = [0x2211, 0x4433, 0x6655, 0x8877]

# --- 滑动窗口参数 ---
TOTAL_TRACES = 1000
WINDOW_STEP = 10
TRACE_MIN = 30
TRACE_MAX = 220


def simon_f(x):
    s1 = ((x << 1) & mask) | (x >> 15)
    s8 = ((x << 8) & mask) | (x >> 8)
    s2 = ((x << 2) & mask) | (x >> 14)
    return (s1 & s8) ^ s2


def get_hw_byte(n):
    return bin(n & 0xFF).count('1')


def calculate_correlations(hypo_hw, traces):
    h = hypo_hw - np.mean(hypo_hw)
    t = traces - np.mean(traces, axis=0)
    numerator = np.dot(h, t)
    denominator = np.sqrt(np.sum(h ** 2) * np.sum(t ** 2, axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = numerator / (denominator + 1e-8)
    return np.nan_to_num(corr)


def normalize_traces(traces):
    mean = traces.mean(axis=0, keepdims=True)
    std = traces.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (traces - mean) / std


def moving_average(traces, window_size=7):
    return np.array([
        np.convolve(trace, np.ones(window_size) / window_size, mode='same')
        for trace in traces
    ])


def attack_single_window_detailed(trace_window, textin_window):

    num_traces = len(trace_window)
    trace_cleaned = normalize_traces(moving_average(trace_window, window_size=7))
    curr_in_L = textin_window[:, 3].astype(np.uint16) << 8 | textin_window[:, 2].astype(np.uint16)
    curr_in_R = textin_window[:, 1].astype(np.uint16) << 8 | textin_window[:, 0].astype(np.uint16)

    recovered_master_key = []
    round_correct = []
    rank_info = []  # 存储每轮正确密钥的排名
    score_info = []  # 存储每轮所有候选密钥的评分

    for r in range(4):
        target_base = simon_f(curr_in_L) ^ curr_in_R

        # 步骤 1：快速宽口径筛选（高低字节各取前 4 候选）
        candidate_pool = []
        for is_high in [False, True]:
            byte_corrs = []
            for g in range(256):
                hypo_hw = np.array([
                    get_hw_byte((target_base[i] >> (8 if is_high else 0)) ^ g)
                    for i in range(num_traces)
                ])
                byte_corrs.append(np.max(np.abs(calculate_correlations(hypo_hw, trace_cleaned))))
            candidate_pool.append(np.argsort(byte_corrs)[-4:])

        all_scores = {}  # 存储所有候选密钥的评分

        for high_b in candidate_pool[1]:
            for low_b in candidate_pool[0]:
                p_key = (high_b << 8) | low_b

                c_low = np.max(np.abs(calculate_correlations(
                    np.array([get_hw_byte(target_base[i] ^ low_b) for i in range(num_traces)]),
                    trace_cleaned)))
                c_high = np.max(np.abs(calculate_correlations(
                    np.array([get_hw_byte((target_base[i] >> 8) ^ high_b) for i in range(num_traces)]),
                    trace_cleaned)))
                current_corr = (c_low + c_high) / 2

                next_corr = 0
                if r < 4:
                    inter_x = (simon_f(curr_in_L) ^ p_key) & mask
                    next_L = (curr_in_R ^ inter_x) & mask
                    next_R = curr_in_L
                    next_target_base = simon_f(next_L) ^ next_R
                    test_corrs = []
                    for g_test in range(32):
                        h_test = np.array([get_hw_byte(next_target_base[i] ^ g_test) for i in range(num_traces)])
                        test_corrs.append(np.max(np.abs(calculate_correlations(h_test, trace_cleaned))))
                    next_corr = np.max(test_corrs)

                joint_score = current_corr * 0.8 + next_corr * 0.2
                all_scores[p_key] = joint_score

        # 按评分排序
        sorted_keys = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        best_p_key = sorted_keys[0][0]

        # 找到正确密钥的排名
        correct_key = REAL_MASTER_KEY[r]
        correct_rank = next((i + 1 for i, (k, s) in enumerate(sorted_keys) if k == correct_key), len(sorted_keys))
        rank_info.append(correct_rank)
        score_info.append(all_scores)

        is_correct = (best_p_key == REAL_MASTER_KEY[r])
        round_correct.append(is_correct)
        recovered_master_key.append(best_p_key)
        curr_in_L, curr_in_R = (curr_in_R ^ ((simon_f(curr_in_L) ^ best_p_key) & mask)) & mask, curr_in_L

    full_key_correct = all(round_correct)
    return recovered_master_key, round_correct, full_key_correct, rank_info, score_info


def plot_success_rate_vs_traces(all_stats, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    trace_counts = [s['n_traces'] for s in all_stats]
    full_sr = [s['full_sr'] for s in all_stats]

    # 子图1：完整密钥成功率
    ax1.plot(trace_counts, full_sr, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Complete Key')
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Success')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Success')
    ax1.set_xlabel('Number of Attack Traces', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Complete Key Recovery Success Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([-5, 105])

    # 子图2：各轮密钥成功率
    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2']
    for r in range(4):
        round_sr = [s['round_sr'][r] for s in all_stats]
        ax2.plot(trace_counts, round_sr, 'o-', linewidth=2, markersize=6,
                 color=colors[r], label=f'Round {r} Key (K{r})', alpha=0.8)

    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Number of Attack Traces', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Round Key Success Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_ylim([-5, 105])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_rank_heatmap(rank_data_all, trace_range, window_step, save_path=None):
    print("\n  生成热力图...")

    trace_counts = list(range(trace_range[0], trace_range[1] + 1, window_step))

    success_rate_matrix = np.zeros((4, len(trace_counts)))

    for r in range(4):
        for i, n in enumerate(trace_counts):
            ranks = rank_data_all[r][n]
            success_rate_matrix[r, i] = (np.array(ranks) == 1).sum() / len(ranks) * 100

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(success_rate_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_yticks(range(4))
    ax.set_yticklabels([f'K{r} (0x{REAL_MASTER_KEY[r]:04X})' for r in range(4)], fontsize=11)
    ax.set_xticks(range(len(trace_counts)))
    ax.set_xticklabels(trace_counts, fontsize=10)
    ax.set_xlabel('Number of Attack Traces', fontsize=12, fontweight='bold')
    ax.set_ylabel('Round Key', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap - Rank=1 Success Rate (%) Across All Windows', fontsize=14, fontweight='bold')

    # 在每个格子上标注数值
    for r in range(4):
        for i in range(len(trace_counts)):
            text_color = "white" if success_rate_matrix[r, i] < 50 else "black"
            text = ax.text(i, r, f'{success_rate_matrix[r, i]:.0f}%',
                           ha="center", va="center", color=text_color,
                           fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_rank_cdf(rank_data_all, trace_range, window_step, save_path=None):
    print("\n  生成累积分布函数图...")

    trace_counts = list(range(trace_range[0], trace_range[1] + 1, window_step))
    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2']

    n_plots = len(trace_counts)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, n_plot in enumerate(trace_counts):
        ax = axes[idx]

        for r in range(4):
            ranks = np.array(rank_data_all[r][n_plot])
            sorted_ranks = np.sort(ranks)
            cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks) * 100

            ax.plot(sorted_ranks, cdf, linewidth=2.5, color=colors[r],
                    label=f'K{r} (0x{REAL_MASTER_KEY[r]:04X})',
                    marker='o', markersize=4, markevery=max(1, len(sorted_ranks) // 10))

        ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.3)
        ax.set_xlabel('Rank of Correct Key', fontsize=10, fontweight='bold')
        ax.set_ylabel('Cumulative Probability (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'CDF - {n_plot} Traces', fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim([0.5, max(10, ax.get_xlim()[1])])
        ax.set_ylim([0, 105])

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Cumulative Distribution Function - Rank Distribution',
                 fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_complementary_key_symmetry(trace_raw_all, textin_raw_all, n_traces=50, save_path=None):

    print("\n正在生成互补密钥统计对称性图...")

    tw = trace_raw_all[:n_traces]
    xw = textin_raw_all[:n_traces]

    trace_cleaned = normalize_traces(moving_average(tw, window_size=7))
    curr_in_L = xw[:, 3].astype(np.uint16) << 8 | xw[:, 2].astype(np.uint16)
    curr_in_R = xw[:, 1].astype(np.uint16) << 8 | xw[:, 0].astype(np.uint16)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for r in range(4):
        target_base = simon_f(curr_in_L) ^ curr_in_R

        key_scores = {}
        for key_guess in range(0x10000):
            hypo_hw_low = np.array([get_hw_byte(target_base[i] ^ (key_guess & 0xFF)) for i in range(n_traces)])
            hypo_hw_high = np.array([get_hw_byte((target_base[i] >> 8) ^ (key_guess >> 8)) for i in range(n_traces)])

            corr_low = np.max(np.abs(calculate_correlations(hypo_hw_low, trace_cleaned)))
            corr_high = np.max(np.abs(calculate_correlations(hypo_hw_high, trace_cleaned)))
            key_scores[key_guess] = (corr_low + corr_high) / 2

        keys = list(key_scores.keys())
        scores = list(key_scores.values())

        correct_key = REAL_MASTER_KEY[r]
        complement_key = correct_key ^ 0xFFFF

        axes[r].scatter(keys, scores, c=scores, cmap='viridis', s=1, alpha=0.5)
        axes[r].scatter([correct_key], [key_scores[correct_key]], c='red', s=100,
                        marker='*', label=f'Correct: 0x{correct_key:04X}', zorder=5)
        axes[r].scatter([complement_key], [key_scores[complement_key]], c='blue', s=100,
                        marker='x', label=f'Complement: 0x{complement_key:04X}', zorder=5)

        axes[r].set_xlabel('Key Space (0x0000 - 0xFFFF)', fontsize=10, fontweight='bold')
        axes[r].set_ylabel('Joint Score', fontsize=10, fontweight='bold')
        axes[r].set_title(f'Round {r} Key - Complementary Symmetry', fontsize=12, fontweight='bold')
        axes[r].legend(fontsize=9)
        axes[r].grid(True, alpha=0.3)

        # 更新状态
        curr_in_L, curr_in_R = (curr_in_R ^ ((simon_f(curr_in_L) ^ correct_key) & mask)) & mask, curr_in_L

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_joint_score_ranking(score_data, save_path=None):
    print("\n  生成联合评分排序对比图...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for r in range(4):
        all_scores = score_data[r]
        sorted_keys = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        keys_hex = [f'0x{k:04X}' for k, _ in sorted_keys]
        scores = [s for _, s in sorted_keys]

        correct_key = REAL_MASTER_KEY[r]
        correct_in_top10 = any(k == correct_key for k, _ in sorted_keys)

        colors_bar = ['#2E86AB' if k != f'0x{correct_key:04X}' else '#E63946' for k in keys_hex]

        bars = axes[r].barh(range(len(keys_hex)), scores, color=colors_bar, alpha=0.8)
        axes[r].set_yticks(range(len(keys_hex)))
        axes[r].set_yticklabels(keys_hex, fontsize=9)
        axes[r].invert_yaxis()
        axes[r].set_xlabel('Joint Score', fontsize=10, fontweight='bold')
        axes[r].set_ylabel('Key Candidate', fontsize=10, fontweight='bold')
        axes[r].set_title(f'Round {r} - Top 10 Key Candidates (Correct: 0x{correct_key:04X})',
                          fontsize=12, fontweight='bold')
        axes[r].grid(True, axis='x', alpha=0.3)

        if correct_in_top10:
            legend_elements = [Patch(facecolor='#E63946', label='Correct Key'),
                               Patch(facecolor='#2E86AB', label='Other Candidates')]
            axes[r].legend(handles=legend_elements, fontsize=9, loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    import os

    file_path = "E:/科研/侧信道/实验代码/数据/simon32_64_raw_traces_train.hdf5"

    # 创建输出目录（在当前脚本所在目录下）
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"图像将保存到: {output_dir}")

    print("=" * 80)
    print("SIMON32/64 CPA 攻击")
    print(f"总读取能量迹: {TOTAL_TRACES} 条")
    print("=" * 80)

    try:
        print(f"\n正在尝试打开文件: {file_path}")

        if not os.path.exists(file_path):
            print(f"\n错误: 文件不存在！")
            print(f"请检查路径: {file_path}")
            print(f"\n提示: 请将代码中的 file_path 变量修改为您的实际文件路径")
            exit(1)

        print(f"文件存在，正在读取 {TOTAL_TRACES} 条能量迹...")
        with h5py.File(file_path, 'r') as f:
            print(f"HDF5文件已打开，可用的组: {list(f.keys())}")
            group = f['fkgroup']
            trace_raw_all = np.array(group['traces'][:TOTAL_TRACES])
            textin_raw_all = np.array(group['pt'][:TOTAL_TRACES])
        print(f"读取完成 | 能量迹形状: {trace_raw_all.shape} | 明文形状: {textin_raw_all.shape}")

        trace_counts = list(range(TRACE_MIN, TRACE_MAX + 1, WINDOW_STEP))
        all_stats = []

        # 初始化数据收集结构
        rank_data_all = {r: {n: [] for n in trace_counts} for r in range(4)}
        score_data_first_window = None  # 保存第一个窗口的评分数据用于联合评分图

        for n_idx, n_traces in enumerate(trace_counts):
            win_starts = list(range(0, TOTAL_TRACES - n_traces + 1, WINDOW_STEP))
            total_wins = len(win_starts)
            success_count = 0
            round_success = [0, 0, 0, 0]

            print(f"\n>>> 能量迹数量: {n_traces:3d} | 窗口数: {total_wins:3d} "
                  f"| 范围: [{win_starts[0]}:{win_starts[0] + n_traces}] ~ "
                  f"[{win_starts[-1]}:{win_starts[-1] + n_traces}]")

            for idx in trange(total_wins, desc=f"  n_traces={n_traces}", leave=False):
                s = win_starts[idx]
                e = s + n_traces
                tw = trace_raw_all[s:e]
                xw = textin_raw_all[s:e]

                # 使用详细版本获取rank和score信息
                _, r_correct, full_ok, rank_info, score_info = attack_single_window_detailed(tw, xw)

                if full_ok:
                    success_count += 1
                for r in range(4):
                    if r_correct[r]:
                        round_success[r] += 1
                    # 保存每个窗口的rank数据
                    rank_data_all[r][n_traces].append(rank_info[r])

                # 保存第一个窗口的评分数据用于绘制联合评分图
                if score_data_first_window is None and idx == 0 and n_idx == 0:
                    score_data_first_window = score_info

            full_sr = success_count / total_wins * 100
            round_sr = [round_success[r] / total_wins * 100 for r in range(4)]

            print(f"    完整密钥成功率: {success_count}/{total_wins} = {full_sr:.1f}%")
            for r in range(4):
                mark = "✓" if round_sr[r] == 100.0 else ("△" if round_sr[r] >= 50.0 else "✗")
                print(f"      第{r}轮密钥成功率: {round_success[r]:>3}/{total_wins} = {round_sr[r]:5.1f}%  {mark}")

            all_stats.append({
                "n_traces": n_traces,
                "total_windows": total_wins,
                "success": success_count,
                "full_sr": full_sr,
                "round_success": round_success,
                "round_sr": round_sr
            })

        print("\n" + "=" * 80)
        print("汇总统计表（成功率 = 正确恢复次数 / 总窗口数）")
        print("=" * 80)
        col_header = f"{'迹数':>6} | {'窗口数':>6} | {'全密钥成功率':>12} | {'K0':>7} | {'K1':>7} | {'K2':>7} | {'K3':>7}"
        print(col_header)
        print("-" * len(col_header))
        for stat in all_stats:
            row = (f"{stat['n_traces']:>6} | "
                   f"{stat['total_windows']:>6} | "
                   f"{stat['full_sr']:>11.1f}% | "
                   f"{stat['round_sr'][0]:>6.1f}% | "
                   f"{stat['round_sr'][1]:>6.1f}% | "
                   f"{stat['round_sr'][2]:>6.1f}% | "
                   f"{stat['round_sr'][3]:>6.1f}%")
            print(row)
        print("=" * 80)
        print("✓ = 100%   △ = >=50%   ✗ = <50%")

        print("\n" + "=" * 80)
        print("开始生成可视化图像...")
        print("=" * 80)

        # 1. 成功率 vs 攻击迹数
        print("\n[1/5] 绘制成功率 vs 攻击迹数曲线...")
        fig1 = plot_success_rate_vs_traces(all_stats,
                                           save_path=os.path.join(output_dir, '1_success_rate_vs_traces.png'))
        plt.close(fig1)
        print("  ✓ 已保存: 1_success_rate_vs_traces.png")

        # 3. Rank热力图
        print("\n[2/5] 绘制Rank热力图...")
        fig3 = plot_rank_heatmap(rank_data_all, (TRACE_MIN, TRACE_MAX), WINDOW_STEP,
                                 save_path=os.path.join(output_dir, '2_rank_heatmap.png'))
        plt.close(fig3)
        print("  ✓ 已保存: 2_rank_heatmap.png")

        # 5. Rank累积分布函数
        print("\n[3/5] 绘制Rank累积分布函数...")
        fig5 = plot_rank_cdf(rank_data_all, (TRACE_MIN, TRACE_MAX), WINDOW_STEP,
                             save_path=os.path.join(output_dir, '3_rank_cdf.png'))
        plt.close(fig5)
        print("  ✓ 已保存: 3_rank_cdf.png")

        # 4. 互补密钥统计对称性图
        print("\n[4/5] 绘制互补密钥统计对称性图...")
        fig4 = plot_complementary_key_symmetry(trace_raw_all, textin_raw_all,
                                                n_traces=min(50, TOTAL_TRACES),
                                                save_path=os.path.join(output_dir, '4_complementary_symmetry.png'))
        plt.close(fig4)
        print("  ✓ 已保存: 4_complementary_symmetry.png")

        # 7. 联合评分排序对比图
        print("\n[5/5] 绘制联合评分排序对比图...")
        if score_data_first_window is not None:
            fig7 = plot_joint_score_ranking(score_data_first_window,
                                            save_path=os.path.join(output_dir, '5_joint_score_ranking.png'))
            plt.close(fig7)
            print("  ✓ 已保存: 5_joint_score_ranking.png")
        else:
            print("  ✗ 警告: 没有评分数据，跳过联合评分图")

        print("\n" + "=" * 80)
        print("所有可视化图像生成完成！")
        print(f"保存位置: {output_dir}")
        print("\n性能优化: 所有数据在攻击阶段一次性收集，绘图时直接使用，无需重复计算！")
        print("=" * 80)

    except KeyError as e:
        print(f"\n错误: HDF5文件结构不匹配！")
        print(f"找不到键: {e}")
        print(f"请检查HDF5文件中是否包含 'fkgroup/traces' 和 'fkgroup/pt' 数据集")
    except Exception as e:
        import traceback

        print(f"\n运行出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        traceback.print_exc()