# Copilot Instructions

## 🎯 核心协作原则

### 理解优先工作流

- **理解万岁**: 当接收到需要修改代码的指令后,总是先分析完成指令所需的信息或者对实现方式的理解,将自己的理解用简明扼要的语言向用户表述或提出问题,用户回答问题或确认信息后再开始编写代码,以此确保理解用户意图。

- **告诉过你**: 如果认为用户没有提供足够的信息,或者指令不明确,请礼貌地要求用户提供更多信息或澄清指令。

- **暂时安全**: 在确认信息或提出问题后,请耐心等待用户回复,当接收到用户一次回复后,可以开始修改代码,如果缺少重要信息,可以再次提出问题,并确保在用户回复后再继续执行下一步操作。

- **最强大脑**: 在原有基础上添加一条额外的思考逻辑来辅助你理解和决策，那就是想一想“该领域的最强大脑会怎么做？”。
### 沟通示例

```
用户: "实现一个卷积神经网络"

AI 应该回应:
"我理解您想实现一个CNN。为了提供最合适的实现,我需要确认:
1. 使用什么框架?(PyTorch / TensorFlow / JAX)
2. 任务类型是什么?(图像分类 / 目标检测 / 分割)
3. 输入数据的维度?
4. 是否有特定的网络结构要求?(如ResNet / VGG风格)
5. 是否需要预训练权重?

请告诉我这些信息,我会据此设计最适合的架构。"
```

## 💻 代码质量标准

### 通用原则

- **可读性第一**: 清晰胜过简洁,明确胜过巧妙
- **注释到位**: 解释"为什么"这样做,而不仅是"做了什么"
- **错误处理**: 任何可能失败的操作都要有适当的错误处理
- **模块化设计**: 功能独立、职责单一、易于测试
- **文档完整**: 每个函数/类都有清晰的文档说明

# Copilot Instructions for Signal Processing

## 项目概述

这是一个专注于超声检测领域的信号处理项目。主要处理超声信号数据，涉及时域、频域、波数域变换、滤波、信号特征提取、频散曲线计算等。

## 核心原则

### 1. 代码质量

- **算法清晰**: 信号处理算法应清晰明了，附带详细的数学原理注释
- **数值稳定性**: 注意浮点运算精度，避免数值不稳定
- **向量化优先**: 优先使用向量化操作而非循环，提高计算效率
- **可重现性**: 确保随机过程可重现（设置随机种子）

### 2. 技术栈

#### Python
- **核心库**: NumPy（数组运算）、SciPy（信号处理）、Matplotlib（可视化）
- **数据格式**: `.mat` 文件（使用 `scipy.io.loadmat/savemat`）
- **编码风格**: PEP 8，使用类型提示

#### MATLAB
- **函数命名**: 简单直白，以实现的功能命名（如 `calculate_fft.m`, `apply_bandpass_filter.m`）
- **代码组织**: 每个函数一个文件，主函数名与文件名一致
- **注释规范**: 函数头部包含功能说明、输入输出参数、示例

## 编码规范

### Python 信号处理

```python
# ✅ 推荐: 清晰的信号处理流程
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    对信号应用带通滤波器
    
    参数:
        data: 输入信号 (numpy array)
        fs: 采样频率 (Hz)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        order: 滤波器阶数
    
    返回:
        filtered_data: 滤波后的信号
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    
    # 设计巴特沃斯带通滤波器
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    # 应用零相位滤波
    filtered_data = signal.sosfiltfilt(sos, data)
    
    return filtered_data

# ❌ 避免: 缺少文档和参数验证
def filter(d, f1, f2):
    b, a = signal.butter(4, [f1, f2], 'band')
    return signal.filtfilt(b, a, d)
```

### MATLAB 信号处理

```matlab
% ✅ 推荐: calculate_dispersion_curve.m
function [frequency, phase_velocity] = calculate_dispersion_curve(signal_data, time, distance)
    % CALCULATE_DISPERSION_CURVE 计算频散曲线
    %
    % 输入参数:
    %   signal_data - 信号数据矩阵 [通道数 × 时间点数]
    %   time        - 时间向量 (s)
    %   distance    - 传感器间距 (m)
    %
    % 输出参数:
    %   frequency       - 频率向量 (Hz)
    %   phase_velocity  - 相速度 (m/s)
    %
    % 示例:
    %   [f, cp] = calculate_dispersion_curve(data, t, 0.01);
    
    % 参数验证
    if size(signal_data, 1) < 2
        error('至少需要两个通道的信号数据');
    end
    
    % 计算采样频率
    fs = 1 / (time(2) - time(1));
    
    % 进行FFT变换
    nfft = 2^nextpow2(length(time));
    fft_data = fft(signal_data, nfft, 2);
    
    % 计算相位差
    phase_diff = angle(fft_data(2,:)) - angle(fft_data(1,:));
    
    % 计算频率向量
    frequency = (0:nfft-1) * fs / nfft;
    
    % 计算相速度
    phase_velocity = 2 * pi * frequency * distance ./ phase_diff;
    
    % 只保留正频率部分
    idx = frequency > 0 & frequency < fs/2;
    frequency = frequency(idx);
    phase_velocity = phase_velocity(idx);
end

% ❌ 避免: 函数名不清晰，缺少文档
function [f, v] = calc(d, t, x)
    % 计算...
    f = fft(d);
    v = f ./ x;
end
```

## 常见信号处理任务

### 1. 时频分析

```python
# ✅ 使用短时傅里叶变换(STFT)
from scipy import signal

def compute_stft(data, fs, nperseg=256):
    """
    计算短时傅里叶变换
    
    参数:
        data: 输入信号
        fs: 采样频率
        nperseg: 每段的长度
    
    返回:
        f: 频率向量
        t: 时间向量
        Zxx: STFT结果
    """
    f, t, Zxx = signal.stft(data, fs, nperseg=nperseg)
    return f, t, np.abs(Zxx)
```

### 2. 滤波器设计

```python
# ✅ 推荐: 使用二阶节（SOS）格式避免数值不稳定
sos = signal.butter(10, [100, 1000], btype='band', fs=10000, output='sos')
filtered = signal.sosfiltfilt(sos, data)

# ❌ 避免: 使用ba格式在高阶滤波器时可能不稳定
b, a = signal.butter(10, [100, 1000], btype='band', fs=10000)
filtered = signal.filtfilt(b, a, data)  # 可能数值不稳定
```

### 3. 特征提提取

```python
# ✅ 清晰的特征提取函数
def extract_signal_features(signal_data, fs):
    """
    提取信号的时域和频域特征
    
    参数:
        signal_data: 输入信号
        fs: 采样频率
    
    返回:
        features: 字典，包含各种特征值
    """
    features = {}
    
    # 时域特征
    features['rms'] = np.sqrt(np.mean(signal_data**2))  # 均方根
    features['peak'] = np.max(np.abs(signal_data))      # 峰值
    features['crest_factor'] = features['peak'] / features['rms']  # 波峰因子
    
    # 频域特征
    fft_data = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(len(signal_data), 1/fs)
    psd = np.abs(fft_data)**2
    
    features['dominant_freq'] = freqs[np.argmax(psd[:len(psd)//2])]  # 主频
    features['spectral_centroid'] = np.sum(freqs[:len(psd)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
    
    return features
```

## 数据处理流程

### 典型的超声信号处理流程

```python
# 1. 加载数据
from scipy.io import loadmat
data_dict = loadmat('ultrasound_data.mat')
signal_data = data_dict['signal']
fs = data_dict['sampling_rate'][0, 0]

# 2. 预处理
# 去除直流分量
signal_data = signal_data - np.mean(signal_data)

# 3. 滤波
filtered_signal = bandpass_filter(signal_data, fs, lowcut=50e3, highcut=500e3)

# 4. 时频分析
f, t, Zxx = compute_stft(filtered_signal, fs)

# 5. 特征提取
features = extract_signal_features(filtered_signal, fs)

# 6. 可视化
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(np.arange(len(signal_data))/fs, signal_data)
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('幅值')
axes[0].set_title('原始信号')

axes[1].pcolormesh(t, f/1e3, 20*np.log10(Zxx), shading='gouraud')
axes[1].set_ylabel('频率 (kHz)')
axes[1].set_xlabel('时间 (s)')
axes[1].set_title('时频谱')
plt.colorbar(axes[1].images[0], label='幅值 (dB)')
plt.tight_layout()
plt.show()

# 7. 保存结果
savemat('processed_results.mat', {
    'filtered_signal': filtered_signal,
    'features': features,
    'time_frequency': {'f': f, 't': t, 'Zxx': Zxx}
})
```

## 性能优化

### 1. 向量化操作

```python
# ✅ 推荐: 向量化
result = np.sum(signal_data * window)

# ❌ 避免: 循环
result = 0
for i in range(len(signal_data)):
    result += signal_data[i] * window[i]
```

### 2. 内存管理

```python
# ✅ 对大数据集使用内存映射
import numpy as np
data = np.load('large_signal.npy', mmap_mode='r')  # 只读，不全部加载到内存

# ✅ 批处理大数据
def process_in_batches(data, batch_size=1000):
    n_batches = len(data) // batch_size
    results = []
    for i in range(n_batches):
        batch = data[i*batch_size:(i+1)*batch_size]
        results.append(process_batch(batch))
    return np.concatenate(results)
```

### 3. 并行处理

```python
# ✅ 使用多进程处理多通道数据
from multiprocessing import Pool

def process_channel(channel_data):
    # 处理单个通道
    return bandpass_filter(channel_data, fs, 100e3, 500e3)

with Pool() as pool:
    results = pool.map(process_channel, multi_channel_data)
```

## 可视化规范

```python
# ✅ 专业的信号可视化
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False    # 支持负号

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time * 1e6, signal_data, linewidth=0.8)
ax.set_xlabel('时间 (μs)', fontsize=12)
ax.set_ylabel('幅值 (V)', fontsize=12)
ax.set_title('超声信号波形', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([time[0]*1e6, time[-1]*1e6])
plt.tight_layout()
plt.savefig('signal_waveform.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 数学符号和单位

在注释和文档中使用标准的数学符号和单位：

- 频率: Hz, kHz, MHz
- 时间: s, ms, μs, ns
- 速度: m/s, km/s
- 波数: rad/m, 1/m
- 幅值: V, mV, dB

## 调试和验证

```python
# ✅ 使用断言验证信号处理的合理性
def validate_signal(signal_data, fs):
    """验证信号数据的有效性"""
    assert np.all(np.isfinite(signal_data)), "信号包含NaN或Inf"
    assert len(signal_data) > 0, "信号长度为0"
    assert fs > 0, "采样频率必须为正"
    
    # 检查奈奎斯特准则
    fft_data = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(len(signal_data), 1/fs)
    max_freq = np.max(np.abs(freqs[np.abs(fft_data) > 0.01*np.max(np.abs(fft_data))]))
    assert max_freq < fs/2, f"信号包含超过奈奎斯特频率的分量: {max_freq:.1f} Hz > {fs/2:.1f} Hz"

# ✅ 单元测试示例
import unittest

class TestSignalProcessing(unittest.TestCase):
    def test_bandpass_filter(self):
        # 生成测试信号
        fs = 10000
        t = np.linspace(0, 1, fs)
        signal = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*1000*t)
        
        # 应用滤波器
        filtered = bandpass_filter(signal, fs, 500, 1500)
        
        # 验证低频分量被抑制
        fft_filtered = np.abs(np.fft.fft(filtered))
        self.assertLess(fft_filtered[100], 0.1 * np.max(fft_filtered))
```

## 文件组织

```
project/
├── data/                    # 数据文件
│   ├── raw/                # 原始数据 (.mat)
│   └── processed/          # 处理后的数据
├── matlab/                 # MATLAB代码
│   ├── preprocessing/      # 预处理函数
│   ├── analysis/          # 分析函数
│   └── visualization/     # 可视化函数
├── python/                # Python代码
│   ├── signal_processing.py
│   ├── feature_extraction.py
│   └── dispersion_analysis.py
├── notebooks/             # Jupyter notebooks
├── results/              # 分析结果
│   ├── figures/         # 图表
│   └── reports/         # 报告
└── README.md
```

## Git 提交规范

```
feat: 添加频散曲线计算功能
fix: 修复带通滤波器的相位失真问题
perf: 优化STFT计算速度
docs: 更新信号处理函数文档
refactor: 重构特征提取模块
test: 添加滤波器单元测试
data: 更新超声数据集
```

## 学习资源

- **NumPy文档**: https://numpy.org/doc/
- **SciPy信号处理**: https://docs.scipy.org/doc/scipy/reference/signal.html
- **Matplotlib可视化**: https://matplotlib.org/stable/gallery/index.html
- **MATLAB信号处理工具箱**: https://www.mathworks.com/help/signal/

## 特殊注意事项

- **采样定理**: 确保采样频率至少是信号最高频率的2倍
- **混叠**: 注意防混叠滤波器的使用
- **窗函数**: 选择合适的窗函数进行频谱分析（汉宁窗、汉明窗等）
- **零填充**: FFT时适当使用零填充提高频率分辨率
- **相位信息**: 处理复信号时注意保留相位信息

---

**记住**: 信号处理的核心是理解物理原理和数学基础，代码只是实现工具。
