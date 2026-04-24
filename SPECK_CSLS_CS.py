import numpy as np
import h5py
import time
import tracemalloc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import trange
from scipy.ndimage import uniform_filter1d
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings


warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', category=np.RankWarning)
warnings.filterwarnings('ignore', message='.*Precision loss occurred.*')
warnings.filterwarnings('ignore', message='.*all input arrays have length 1.*')
warnings.filterwarnings('ignore', module='scipy.stats._distn_infrastructure')


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

epsilon = 1e-8

NUM_ROUNDS = 22
BLOCK_SIZE = 32
KEY_SIZE = 64
WORD_SIZE = 16

# SHIFTs for SPECK
ALPHA = 7
BETA = 2

mod_mask = (2 ** WORD_SIZE) - 1
mod_mask_sub = (2 ** WORD_SIZE)

template = [0x11, 0x22, 0xdd, 0x0, 0xdc, 0xa8, 0x9c, 0x34]


log_file = f"speck_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_message(message):

    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def mean(X):
    return np.sum(X, axis=0) / len(X)


def std_dev(X, X_bar):
    return np.sqrt(np.sum((X - X_bar) ** 2, axis=0))


def cov(X, X_bar, Y, Y_bar):
    return np.sum((X - X_bar) * (Y - Y_bar).reshape(-1, 1), axis=0)

file_path = "speck32_64_raw_traces3.hdf5"
if os.path.exists(file_path):
    print("文件存在!")
else:
    print("文件不存在,请检查文件路径或文件名!")

with h5py.File(file_path, 'r') as f:
    trace_array = np.array(f['fkgroup']['traces'][:5000])
    textin_array = np.array(f['fkgroup']['pt'][:5000])


textin_cleaned = []
trace_cleaned = []
for (i, pt) in enumerate(textin_array):
    if len(pt) != 4:
        continue
    textin_cleaned.append(pt)
    trace_cleaned.append(trace_array[i])
trace_cleaned = np.array(trace_cleaned)
textin_cleaned = np.array(textin_cleaned)
print(f"清洗后数据量: {len(trace_cleaned)}")



def normalize_traces(traces):
    mean = traces.mean(axis=1, keepdims=True)
    std = traces.std(axis=1, keepdims=True)
    return (traces - mean) / std



def moving_average(traces, window_size=5):
    return np.array([
        np.convolve(trace, np.ones(window_size) / window_size, mode='same')
        for trace in traces
    ])


trace_cleaned = normalize_traces(trace_cleaned)
trace_cleaned = moving_average(trace_cleaned, window_size=5)



def ER16(x, y, k):
    rs_x = ((x << (16 - ALPHA)) + (x >> ALPHA)) & mod_mask
    add_sxy = (rs_x + y) & mod_mask
    new_x = k ^ add_sxy
    ls_y = ((y >> (16 - BETA)) + (y << BETA)) & mod_mask
    new_y = new_x ^ ls_y
    return new_x, new_y, rs_x, add_sxy



def leakage_model(plaintext, key, arg=None):
    """第一轮泄露"""
    Ct_0 = (int(plaintext[1]) << 8) + int(plaintext[0])
    Ct_1 = (int(plaintext[3]) << 8) + int(plaintext[2])
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, key)
    inter = Ct_0
    return bin(inter).count("1")


def leakage_model1(plaintext, keybyte, keybyte2, key, arg=None):
    """第二轮泄露"""
    Ct_0 = (int(plaintext[1]) << 8) + int(plaintext[0])
    Ct_1 = (int(plaintext[3]) << 8) + int(plaintext[2])
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, (keybyte2 << 8) + keybyte)
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, key)
    inter = Ct_0
    return bin(inter).count("1")


def leakage_model2(plaintext, keybyte, keybyte2, keybyte3, keybyte4, key, arg=None):
    """第三轮泄露"""
    Ct_0 = (int(plaintext[1]) << 8) + int(plaintext[0])
    Ct_1 = (int(plaintext[3]) << 8) + int(plaintext[2])
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, (keybyte2 << 8) + keybyte)
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, (keybyte4 << 8) + keybyte3)
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, key)
    inter = Ct_0
    return bin(inter).count("1")


def leakage_model3(plaintext, keybyte, keybyte2, keybyte3, keybyte4, keybyte5, keybyte6, key, arg=None):
    """第四轮泄露"""
    Ct_0 = (int(plaintext[1]) << 8) + int(plaintext[0])
    Ct_1 = (int(plaintext[3]) << 8) + int(plaintext[2])
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, (keybyte2 << 8) + keybyte)
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, (keybyte4 << 8) + keybyte3)
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, (keybyte6 << 8) + keybyte5)
    Ct_1, Ct_0, leftmove, add = ER16(Ct_1, Ct_0, key)
    inter = Ct_0
    return bin(inter).count("1")



def calculate_correlations(traces, plaintexts, model_callback, model_args=(), leftmost=True, other_keybyte=0x00):
    """计算余弦相似度攻击"""
    maxcos = [0] * 256
    index = [0] * 256
    cos_output_array = np.empty((256, traces.shape[1]))

    # 对轨迹按列标准化
    traces_centered = traces - np.mean(traces, axis=0, keepdims=True)
    traces_norm = traces_centered / (np.linalg.norm(traces_centered, axis=0, keepdims=True) + epsilon)

    for kguess in trange(0, 256, desc="密钥猜测", leave=False):
        key = (kguess + (other_keybyte << 8)) if leftmost else ((kguess << 8) + other_keybyte)
        hws = np.array([[model_callback(pt, *model_args, key)] for pt in plaintexts])

        # 中心化并标准化泄露向量
        hws_centered = hws - np.mean(hws)
        hws_norm = hws_centered / (np.linalg.norm(hws_centered) + epsilon)

        # 余弦相似度
        cos_similarities = np.dot(hws_norm.T, traces_norm)
        cos_output_array[kguess] = cos_similarities[0]

        smoothed = uniform_filter1d(np.abs(cos_similarities), size=11)
        maxcos[kguess] = np.max(smoothed)
        index[kguess] = np.argmax(np.abs(cos_similarities))

    best_guess = int(np.argmax(maxcos))
    return best_guess, maxcos, cos_output_array


def calculate_guessing_entropy(correlation, correct_key):

    sorted_indices = np.argsort(correlation)[::-1]
    rank = np.where(sorted_indices == correct_key)[0][0] + 1
    return rank


def calculate_statistics(ge_list):

    if not ge_list:
        return {}

    ge_array = np.array(ge_list)
    return {
        'mean': np.mean(ge_array),
        'median': np.median(ge_array),
        'std': np.std(ge_array),
        'min': np.min(ge_array),
        'max': np.max(ge_array),
        'ge_equal_1': np.sum(ge_array == 1),
        'ge_equal_2': np.sum(ge_array == 2),
        'ge_equal_3': np.sum(ge_array == 3),
        'ge_equal_4': np.sum(ge_array == 4),
        'ge_equal_5': np.sum(ge_array == 5),
        'success_rate_1': np.sum(ge_array == 1) / len(ge_array),
        'success_rate_5': np.sum(ge_array <= 5) / len(ge_array),
        'data': ge_array
    }


def attack_all_keybytes_with_ge(traces, plaintexts, step_size):

    keybytes = []
    correlations_all = []
    ge_values = []


    keybyte, correlation, _ = calculate_correlations(traces, plaintexts, leakage_model,
                                                     leftmost=True, other_keybyte=0x00)
    keybytes.append(keybyte)
    correlations_all.append(correlation)
    ge1 = calculate_guessing_entropy(correlation, template[0])
    ge_values.append(ge1)


    keybyte2, correlation2, _ = calculate_correlations(traces, plaintexts, leakage_model,
                                                       leftmost=False, other_keybyte=keybyte)
    keybytes.append(keybyte2)
    correlations_all.append(correlation2)
    ge2 = calculate_guessing_entropy(correlation2, template[1])
    ge_values.append(ge2)


    keybyte3, correlation3, _ = calculate_correlations(traces, plaintexts, leakage_model1,
                                                       model_args=(keybyte, keybyte2),
                                                       leftmost=True, other_keybyte=0x00)
    keybytes.append(keybyte3)
    correlations_all.append(correlation3)
    ge3 = calculate_guessing_entropy(correlation3, template[2])
    ge_values.append(ge3)


    keybyte4, correlation4, _ = calculate_correlations(traces, plaintexts, leakage_model1,
                                                       model_args=(keybyte, keybyte2),
                                                       leftmost=False, other_keybyte=keybyte3)
    keybytes.append(keybyte4)
    correlations_all.append(correlation4)
    ge4 = calculate_guessing_entropy(correlation4, template[3])
    ge_values.append(ge4)


    keybyte5, correlation5, _ = calculate_correlations(traces, plaintexts, leakage_model2,
                                                       model_args=(keybyte, keybyte2, keybyte3, keybyte4),
                                                       leftmost=True, other_keybyte=0x00)
    keybytes.append(keybyte5)
    correlations_all.append(correlation5)
    ge5 = calculate_guessing_entropy(correlation5, template[4])
    ge_values.append(ge5)


    keybyte6, correlation6, _ = calculate_correlations(traces, plaintexts, leakage_model2,
                                                       model_args=(keybyte, keybyte2, keybyte3, keybyte4),
                                                       leftmost=False, other_keybyte=keybyte5)
    keybytes.append(keybyte6)
    correlations_all.append(correlation6)
    ge6 = calculate_guessing_entropy(correlation6, template[5])
    ge_values.append(ge6)


    keybyte7, correlation7, _ = calculate_correlations(traces, plaintexts, leakage_model3,
                                                       model_args=(
                                                       keybyte, keybyte2, keybyte3, keybyte4, keybyte5, keybyte6),
                                                       leftmost=True, other_keybyte=0x00)
    keybytes.append(keybyte7)
    correlations_all.append(correlation7)
    ge7 = calculate_guessing_entropy(correlation7, template[6])
    ge_values.append(ge7)


    keybyte8, correlation8, _ = calculate_correlations(traces, plaintexts, leakage_model3,
                                                       model_args=(
                                                       keybyte, keybyte2, keybyte3, keybyte4, keybyte5, keybyte6),
                                                       leftmost=False, other_keybyte=keybyte7)
    keybytes.append(keybyte8)
    correlations_all.append(correlation8)
    ge8 = calculate_guessing_entropy(correlation8, template[7])
    ge_values.append(ge8)

    return keybytes, correlations_all, ge_values


def attack_with_step_sizes(trace_cleaned, textin_cleaned, step_sizes):

    results = {step: [] for step in step_sizes}
    ge_results = {step: [] for step in step_sizes}
    success_counts = {step: 0 for step in step_sizes}
    total_counts = {step: 0 for step in step_sizes}

    log_message("=" * 80)
    log_message("开始SPECK密码分析与猜测熵计算")
    log_message(f"正确密钥模板: {[hex(x) for x in template]}")
    log_message("=" * 80)

    for step in step_sizes:
        step_ge_values = [[] for _ in range(8)]

        log_message(f"\n{'=' * 20} 攻击步长: {step} {'=' * 20}")

        for start in range(0, len(trace_cleaned) - step + 1, 100):
            total_counts[step] += 1
            end = start + step
            traces_step = trace_cleaned[start:end]
            plaintexts_step = textin_cleaned[start:end]

            keybytes, correlations, ge_values = attack_all_keybytes_with_ge(traces_step, plaintexts_step, step)

            for i, ge in enumerate(ge_values):
                step_ge_values[i].append(ge)

            if keybytes == template:
                success_counts[step] += 1

            results[step].append(keybytes)
            ge_results[step].append(ge_values)

            log_message(f"范围 {start:4d}-{end:4d}: 猜测密钥={[hex(x) for x in keybytes]}, GE={ge_values}")

        # 计算统计结果
        success_rate = success_counts[step] / total_counts[step]
        log_message(f"\n步长 {step} 总体结果:")
        log_message(f"  总攻击次数: {total_counts[step]}")
        log_message(f"  完全正确次数: {success_counts[step]}")
        log_message(f"  完全正确率: {success_rate:.4f}")

        log_message(f"\n步长 {step} 各密钥字节GE统计:")
        for i in range(8):
            stats = calculate_statistics(step_ge_values[i])
            log_message(f"  密钥字节 {i + 1} (正确值: {hex(template[i])}):")
            log_message(f"    平均GE: {stats['mean']:.2f}")
            log_message(f"    中位GE: {stats['median']:.2f}")
            log_message(f"    GE标准差: {stats['std']:.2f}")
            log_message(f"    GE范围: [{stats['min']}, {stats['max']}]")
            log_message(f"    GE=1次数: {stats['ge_equal_1']} ({stats['success_rate_1']:.4f})")

        log_message("-" * 60)

    return results, ge_results, success_counts, total_counts


def plot_success_rate_trend(success_counts, total_counts, step_sizes):

    plt.figure(figsize=(12, 6))

    success_rates = [success_counts[step] / total_counts[step] for step in step_sizes]

    plt.plot(step_sizes, success_rates, marker='o', linewidth=2, markersize=8,
             color='#2E86AB', label='Attack Success Rate')

    # 添加趋势线 (仅当有足够数据点时)
    if len(step_sizes) >= 4:
        try:
            z = np.polyfit(step_sizes, success_rates, min(3, len(step_sizes) - 1))
            p = np.poly1d(z)
            plt.plot(step_sizes, p(step_sizes), "--", alpha=0.5, color='red', label='Trend Line')
        except np.RankWarning:
            pass

    plt.xlabel('Number of Power Traces', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate', fontsize=12, fontweight='bold')
    plt.title('Attack Success Rate vs Number of Traces', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)


    if max(success_rates) >= 0.5:
        threshold_idx = next((i for i, rate in enumerate(success_rates) if rate >= 0.5), None)
        if threshold_idx is not None:
            plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
            plt.axvline(x=step_sizes[threshold_idx], color='green', linestyle='--', alpha=0.5)
            plt.text(step_sizes[threshold_idx], 0.5, f'  50% @ {step_sizes[threshold_idx]} traces',
                     fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('1_success_rate_trend.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ 图1已保存: 1_success_rate_trend.png")


def plot_convergence_comparison(ge_results, step_sizes):

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for byte_idx in range(8):
        ax = axes[byte_idx]

        mean_ge = []
        std_ge = []

        for step in step_sizes:
            ge_list = [ge_values[byte_idx] for ge_values in ge_results[step]]
            mean_ge.append(np.mean(ge_list))
            std_ge.append(np.std(ge_list))

        mean_ge = np.array(mean_ge)
        std_ge = np.array(std_ge)

        ax.plot(step_sizes, mean_ge, marker='o', linewidth=2,
                color=colors[byte_idx], label=f'KeyByte {byte_idx + 1}')
        ax.fill_between(step_sizes, mean_ge - std_ge, mean_ge + std_ge,
                        alpha=0.2, color=colors[byte_idx])

        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Perfect (GE=1)')
        ax.set_xlabel('Number of Traces', fontsize=10)
        ax.set_ylabel('Guessing Entropy', fontsize=10)
        ax.set_title(f'KeyByte {byte_idx + 1} (0x{template[byte_idx]:02x})',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)


        ax.set_ylim([0, min(max(mean_ge) * 1.2, 256)])

    plt.suptitle('Convergence Speed Comparison of 8 Key Bytes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('2_convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ 图2已保存: 2_convergence_comparison.png")


def plot_statistical_stability(ge_results, step_sizes):

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 6, figure=fig)


    ax1 = fig.add_subplot(gs[0, :3])
    for byte_idx in range(8):
        variance_values = []
        for step in step_sizes:
            ge_list = [ge_values[byte_idx] for ge_values in ge_results[step]]
            variance_values.append(np.var(ge_list))
        ax1.plot(step_sizes, variance_values, marker='o', label=f'KB{byte_idx + 1}')

    ax1.set_xlabel('Number of Traces', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Variance of GE', fontsize=11, fontweight='bold')
    ax1.set_title('Variance Stability Analysis', fontsize=12, fontweight='bold')
    ax1.legend(ncol=4, fontsize=9)
    ax1.grid(True, alpha=0.3)


    ax2 = fig.add_subplot(gs[:3, 3:6])
    cv_values = []
    for byte_idx in range(8):
        cv_list = []
        for step in step_sizes:
            ge_list = [ge_values[byte_idx] for ge_values in ge_results[step]]
            mean_val = np.mean(ge_list)
            std_val = np.std(ge_list)
            cv = (std_val / mean_val) if mean_val > 0 else 0
            cv_list.append(cv)
        cv_values.append(np.mean(cv_list))

    ax2.bar(range(1, 9), cv_values, color=plt.cm.viridis(np.linspace(0, 1, 8)))
    ax2.set_xlabel('Key Byte', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation', fontsize=11, fontweight='bold')
    ax2.set_title('CV by Key Byte', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')


    ax3 = fig.add_subplot(gs[1, :3])
    for byte_idx in range(8):
        ci_widths = []
        for step in step_sizes:
            ge_list = [ge_values[byte_idx] for ge_values in ge_results[step]]
            if len(ge_list) > 1:
                try:
                    # 计算标准误
                    sem_val = stats.sem(ge_list)
                    # 检查sem是否有效
                    if not np.isnan(sem_val) and not np.isinf(sem_val) and sem_val > 0:
                        ci = stats.t.interval(0.95, len(ge_list) - 1,
                                              loc=np.mean(ge_list),
                                              scale=sem_val)

                        if not (np.isnan(ci[0]) or np.isnan(ci[1]) or np.isinf(ci[0]) or np.isinf(ci[1])):
                            ci_widths.append(ci[1] - ci[0])
                        else:
                            ci_widths.append(0)
                    else:
                        ci_widths.append(0)
                except:
                    ci_widths.append(0)
            else:
                ci_widths.append(0)
        ax3.plot(step_sizes, ci_widths, marker='s', label=f'KB{byte_idx + 1}')

    ax3.set_xlabel('Number of Traces', fontsize=11, fontweight='bold')
    ax3.set_ylabel('95% CI Width', fontsize=11, fontweight='bold')
    ax3.set_title('Confidence Interval Width (Narrower = More Stable)',
                  fontsize=12, fontweight='bold')
    ax3.legend(ncol=4, fontsize=9)
    ax3.grid(True, alpha=0.3)


    ax4 = fig.add_subplot(gs[2, :3])
    for byte_idx in range(8):
        se_values = []
        for step in step_sizes:
            ge_list = [ge_values[byte_idx] for ge_values in ge_results[step]]
            if len(ge_list) > 1:
                try:
                    se = stats.sem(ge_list)
                    # 检查se是否有效
                    if not np.isnan(se) and not np.isinf(se):
                        se_values.append(se)
                    else:
                        se_values.append(0)
                except:
                    se_values.append(0)
            else:
                se_values.append(0)
        ax4.plot(step_sizes, se_values, marker='d', label=f'KB{byte_idx + 1}')

    ax4.set_xlabel('Number of Traces', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Standard Error', fontsize=11, fontweight='bold')
    ax4.set_title('Standard Error Trend (Lower = More Reliable)',
                  fontsize=12, fontweight='bold')
    ax4.legend(ncol=8, fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Statistical Stability Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('3_statistical_stability.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ 图3已保存: 3_statistical_stability.png")


def plot_variance_theory_validation(ge_results, step_sizes):

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for byte_idx in range(8):
        ax = axes[byte_idx]

        actual_variance = []
        theoretical_variance = []
        sample_sizes = []
        valid_steps = []

        base_var = None
        base_n = None

        for step in step_sizes:
            ge_list = [ge_values[byte_idx] for ge_values in ge_results[step]]


            if len(ge_list) > 1:


                n = step

                sample_sizes.append(n)
                valid_steps.append(step)


                actual_var = np.var(ge_list, ddof=1)
                actual_variance.append(actual_var)


                if base_var is None:
                    base_var = actual_var
                    base_n = n
                    theoretical_variance.append(actual_var)
                else:

                    theoretical_var = base_var * (base_n / n)
                    theoretical_variance.append(theoretical_var)


        if len(valid_steps) > 0:


            ax.plot(valid_steps,
                    actual_variance,
                    marker='o',
                    linewidth=2,
                    label='Actual Variance',
                    color='blue')


            if len(theoretical_variance) > 0:
                ax.plot(valid_steps,
                        theoretical_variance,
                        marker='s',
                        linewidth=2,
                        linestyle='--',
                        label='Reference 1/n Trend',
                        color='red',
                        alpha=0.7)


            if len(actual_variance) > 1 and len(theoretical_variance) > 1:
                try:
                    r_squared = np.corrcoef(actual_variance, theoretical_variance)[0, 1] ** 2

                    if not np.isnan(r_squared):
                        ax.text(0.05,
                                0.95,
                                f'$R^2$ = {r_squared:.4f}',
                                transform=ax.transAxes,
                                fontsize=10,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round',
                                facecolor='wheat',
                                alpha=0.5))
                except:
                    pass

        ax.set_xlabel('Number of Traces', fontsize=10)
        ax.set_ylabel('Variance', fontsize=10)

        ax.set_title(f'KeyByte {byte_idx + 1} (0x{template[byte_idx]:02x})',
                     fontsize=11,
                     fontweight='bold')

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Variance Theory Validation: Actual vs Reference Trend',
                 fontsize=14,
                 fontweight='bold')

    plt.tight_layout()

    plt.savefig('4_variance_theory_validation.png',
                dpi=300,
                bbox_inches='tight')

    plt.show()

    print("✓ 图4已保存: 4_variance_theory_validation.png")


def plot_convergence_zones(ge_results, step_sizes):

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for byte_idx in range(8):
        ax = axes[byte_idx]

        mean_ge = []
        for step in step_sizes:
            ge_list = [ge_values[byte_idx] for ge_values in ge_results[step]]
            mean_ge.append(np.mean(ge_list))

        mean_ge = np.array(mean_ge)

        initial_ge = mean_ge[0]
        fast_threshold = initial_ge * 0.5
        medium_threshold = initial_ge * 0.2

        fast_zone_end = next((i for i, ge in enumerate(mean_ge) if ge <= fast_threshold), len(mean_ge))
        medium_zone_end = next((i for i, ge in enumerate(mean_ge) if ge <= medium_threshold), len(mean_ge))

        if fast_zone_end > 0:
            ax.plot(step_sizes[:fast_zone_end + 1], mean_ge[:fast_zone_end + 1],
                    linewidth=3, color='red', label='Fast Convergence')

        if medium_zone_end > fast_zone_end:
            ax.plot(step_sizes[fast_zone_end:medium_zone_end + 1],
                    mean_ge[fast_zone_end:medium_zone_end + 1],
                    linewidth=3, color='orange', label='Medium Convergence')

        if medium_zone_end < len(mean_ge) - 1:
            ax.plot(step_sizes[medium_zone_end:], mean_ge[medium_zone_end:],
                    linewidth=3, color='green', label='Slow Convergence')

        ax.axhline(y=fast_threshold, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=medium_threshold, color='orange', linestyle='--', alpha=0.3)
        ax.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='Target (GE=1)')

        if fast_zone_end > 0:
            ax.text(step_sizes[fast_zone_end // 2], mean_ge[fast_zone_end // 2],
                    'FAST', fontsize=9, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlabel('Number of Traces', fontsize=10)
        ax.set_ylabel('Mean Guessing Entropy', fontsize=10)
        ax.set_title(f'KeyByte {byte_idx + 1} (0x{template[byte_idx]:02x})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Convergence Zone Analysis: Fast → Medium → Slow',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('5_convergence_zones.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ 图5已保存: 5_convergence_zones.png")

if __name__ == "__main__":
    log_message(f"SPECK密码分析日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("=" * 80)

    step_sizes = list(range(50, 101, 10)) + list(range(200, 1501, 100))

    print("\n开始攻击分析...")
    results, ge_results, success_counts, total_counts = attack_with_step_sizes(
        trace_cleaned, textin_cleaned, step_sizes
    )

    print("\n" + "=" * 80)
    print("开始生成可视化图表...")
    print("=" * 80 + "\n")

    plot_success_rate_trend(success_counts, total_counts, step_sizes)
    plot_convergence_comparison(ge_results, step_sizes)
    plot_statistical_stability(ge_results, step_sizes)
    plot_variance_theory_validation(ge_results, step_sizes)
    plot_convergence_zones(ge_results, step_sizes)

    log_message("\n" + "=" * 80)
    log_message("最终统计报告")
    log_message("=" * 80)

    for step, keybyte_lists in results.items():
        log_message(f"\n步长 {step} 密钥猜测统计:")
        keybyte_counts = [{} for _ in range(8)]

        for keybytes in keybyte_lists:
            for i, keybyte in enumerate(keybytes):
                if keybyte in keybyte_counts[i]:
                    keybyte_counts[i][keybyte] += 1
                else:
                    keybyte_counts[i][keybyte] = 1

        for i, counts in enumerate(keybyte_counts):
            log_message(f"  密钥字节 {i + 1} (正确值: {hex(template[i])}):")
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for keybyte, count in sorted_counts[:5]:
                is_correct = "✓" if keybyte == template[i] else "✗"
                log_message(f"    {hex(keybyte)}: {count}次 {is_correct}")

    log_message("\n" + "=" * 80)
    log_message("综合猜测熵(GE)分析")
    log_message("=" * 80)

    for step in step_sizes:
        log_message(f"\n步长 {step} 的GE综合分析:")
        all_ge_values = []
        for ge_list in ge_results[step]:
            all_ge_values.extend(ge_list)

        if all_ge_values:
            overall_stats = calculate_statistics(all_ge_values)
            log_message(f"  总体平均GE: {overall_stats['mean']:.2f}")
            log_message(f"  总体中位GE: {overall_stats['median']:.2f}")
            log_message(f"  GE=1的比例: {overall_stats['success_rate_1']:.4f}")
            log_message(f"  GE≤5的比例: {overall_stats['success_rate_5']:.4f}")

    print("\n" + "=" * 80)
    print("✓ 所有分析完成!")
    print(f"✓ 日志文件: {log_file}")
    print("  1. 1_success_rate_trend.png - 成功率整体趋势")
    print("  2. 2_convergence_comparison.png - 收敛速度对比")
    print("  3. 3_statistical_stability.png - 统计稳定性分析")
    print("  4. 4_variance_theory_validation.png - 方差验证理论")
    print("  5. 5_convergence_zones.png - 收敛区间划分")
    print("=" * 80)

    log_message(f"\n分析完成!结果已保存到: {log_file}")
    log_message("=" * 80)