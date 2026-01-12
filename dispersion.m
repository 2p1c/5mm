%% 数据预处理：加载并重塑蛇形扫描数据
% 加载原始数据
load("/Volumes/ESD-ISO/数据/260108/complete/data21.mat"); % 包含变量 x (1×2500) 和 y (465×2500)

% 计算采样率和时间向量
data_time = x; % 时间向量
fs = 6.25e6; % 采样率 (Hz)

% 设置点阵参数 - 矩形点阵
n_cols = 1;      % x方向列数
n_rows = 21;    % y方向行数
spacing = 5e-4;  % 物理间距 0.5mm = 0.0005m

% 生成坐标向量
data_x = (0:n_cols-1) * spacing; % x方向坐标 (m)
data_y = (0:n_rows-1) * spacing; % y方向坐标 (m)

% 验证数据点数
total_points = n_cols * n_rows;
assert(size(y, 1) == total_points, ...
    sprintf('数据点数不匹配: 期望 %d×%d=%d, 实际 %d', ...
    n_cols, n_rows, total_points, size(y, 1)));

% 将蛇形扫描数据重塑为 n_cols×n_rows×2500 的三维数组
% 扫描方式: 先扫描y方向(列方向),再移动到下一列x方向
data_xyt = zeros(n_cols, n_rows, length(data_time));

for col = 1:n_cols
    % 计算当前列在y数组中的起始和结束索引
    start_idx = (col-1) * n_rows + 1;
    end_idx = col * n_rows;
    
    % 提取当前列的数据
    col_data = y(start_idx:end_idx, :);
    
    % 根据列数决定是否翻转（偶数列从下到上扫描）
    if mod(col, 2) == 0
        col_data = flipud(col_data); % 翻转偶数列
    end
    
    % 存储到三维数组中: data_xyt(x位置, y位置, 时间)
    data_xyt(col, :, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d列 × %d行\n', n_cols, n_rows);
fprintf('  物理尺寸: %.2f mm × %.2f mm\n', ...
    (n_cols-1)*spacing*1e3, (n_rows-1)*spacing*1e3);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  数据形状: %s\n', mat2str(size(data_xyt)));

%% 随机选择一个点并分析其时域和频域特性
% 随机选择点的坐标
rand_x = randi(n_cols);
rand_y = randi(n_rows);

% 提取该点的时域信号
point_signal = squeeze(data_xyt(rand_x, rand_y, :));

% 计算频谱
nfft = 2^nextpow2(length(point_signal)); % FFT点数
freq_spectrum = fft(point_signal, nfft);
freq_vector = (0:nfft-1) * fs / nfft; % 频率向量

% 只保留正频率部分
half_idx = 1:nfft/2;
freq_vector_pos = freq_vector(half_idx);
amplitude_spectrum = abs(freq_spectrum(half_idx));

fprintf('\n随机点分析:\n');
fprintf('  选择的点: (%d, %d)\n', rand_x, rand_y);
fprintf('  物理坐标: (%.2f mm, %.2f mm)\n', ...
    data_x(rand_x)*1e3, data_y(rand_y)*1e3);
fprintf('  信号RMS: %.4e\n', rms(point_signal));
fprintf('  信号峰值: %.4e\n', max(abs(point_signal)));
[~, max_freq_idx] = max(amplitude_spectrum);
fprintf('  主频: %.2f MHz\n', freq_vector_pos(max_freq_idx)/1e6);

%% 对随机点信号施加滤波并对比
% 设置滤波参数
center_freq = 4e5;    % 中心频率 400 kHz
bandwidth = 6e5;      % 带宽 600 kHz
filter_order = 2;     % 滤波器阶数

% 显示滤波器信息
fprintf('\n滤波处理:\n');
Filter.printInfo(center_freq, bandwidth, filter_order, fs);

% 应用带通滤波器
filtered_signal = Filter.apply(point_signal, fs, center_freq, bandwidth, filter_order);

% 计算滤波后的频谱
filtered_spectrum = fft(filtered_signal, nfft);
filtered_amplitude = abs(filtered_spectrum(half_idx));

% 计算滤波效果指标
original_energy = sum(point_signal.^2);
filtered_energy = sum(filtered_signal.^2);
energy_ratio = filtered_energy / original_energy * 100;

fprintf('  滤波后信号能量保留: %.2f%%\n', energy_ratio);
fprintf('  滤波后信号RMS: %.4e\n', rms(filtered_signal));
fprintf('  滤波后信号峰值: %.4e\n', max(abs(filtered_signal)));

% 可视化:滤波前后对比
figure('Name', '滤波前后信号对比', 'Position', [100, 100, 1400, 800]);

% 时域信号对比
subplot(2, 2, 1);
plot(data_time * 1e6, point_signal, 'b-', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title(sprintf('原始时域信号 - 位置: (%d, %d)', rand_x, rand_y), 'FontSize', 12);
grid on;
legend('原始信号', 'Location', 'best');

subplot(2, 2, 2);
plot(data_time * 1e6, filtered_signal, 'r-', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title('滤波后时域信号', 'FontSize', 12);
grid on;
legend('滤波信号', 'Location', 'best');

% 频域信号对比
subplot(2, 2, 3);
plot(freq_vector_pos / 1e6, amplitude_spectrum, 'b-', 'LineWidth', 1);
hold on;
% 标注通带范围
lowcut = (center_freq - bandwidth/2) / 1e6;
highcut = (center_freq + bandwidth/2) / 1e6;
xline(lowcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', lowcut));
xline(highcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', highcut));
hold off;
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title('原始频谱', 'FontSize', 12);
grid on;
xlim([0, fs/6/1e6]);
legend('原始频谱', 'Location', 'best');

subplot(2, 2, 4);
plot(freq_vector_pos / 1e6, filtered_amplitude, 'r-', 'LineWidth', 1);
hold on;
xline(lowcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', lowcut));
xline(highcut, 'g--', 'LineWidth', 1.5, 'Label', sprintf('%.2f MHz', highcut));
hold off;
xlabel('频率 (MHz)', 'FontSize', 11);
ylabel('幅值', 'FontSize', 11);
title('滤波后频谱', 'FontSize', 12);
grid on;
xlim([0, fs/6/1e6]);
legend('滤波频谱', 'Location', 'best');

% 添加总标题
sgtitle('随机点信号滤波前后对比分析', 'FontSize', 14, 'FontWeight', 'bold');

%% 频散曲线计算（f-k域分析）
% 1. 提取中间一行数据进行空间-时间二维分析
% 根据实际列数选择合适的列
if n_cols == 1
    middle_col_index = 1;  % 只有1列时使用第1列
else
    middle_col_index = ceil(n_cols / 2);  % 多列时选择中间列
end

data_yt = permute(data_xyt(middle_col_index, :, :), [2, 3, 1]);  % [y空间 × 时间]

fprintf('\n频散曲线计算:\n');
fprintf('  使用第 %d 列数据 (共 %d 列)\n', middle_col_index, n_cols);
fprintf('  分析y方向的波传播特性\n');

% 2. 对整列数据应用滤波
fprintf('  对整列数据应用滤波...\n');
data_yt_filtered = zeros(size(data_yt));
for i = 1:n_rows
    data_yt_filtered(i, :) = Filter.apply(data_yt(i, :), fs, center_freq, bandwidth, filter_order);
end
fprintf('  滤波完成\n');

% 2.5 应用小波去噪
wavelet_name = 'db4';      % Daubechies 4小波
wavelet_level = 1;         % 分解层数
threshold_method = 'soft'; % 软阈值

fprintf('\n  应用小波去噪...\n');
Filter.printWaveletInfo(wavelet_name, wavelet_level, threshold_method);

for i = 1:n_rows
    data_yt_filtered(i, :) = Filter.waveletDenoise(data_yt_filtered(i, :), wavelet_name, wavelet_level, threshold_method);
end
fprintf('  小波去噪完成\n');

% 3. 设置FFT参数
% 使用零填充提高分辨率
nfft_space = 2^(nextpow2(n_rows) + 1);          % 空间维度FFT点数
nfft_time = 2^(nextpow2(length(data_time)) + 1); % 时间维度FFT点数

fprintf('  FFT点数: 空间=%d, 时间=%d\n', nfft_space, nfft_time);

% 4. 对原始数据和滤波数据分别进行二维傅里叶变换
% 原始数据
kf_spectrum_original = fftn(data_yt, [nfft_space, nfft_time]);
kf_shifted_original = fftshift(kf_spectrum_original, 1);

% 滤波数据
kf_spectrum_filtered = fftn(data_yt_filtered, [nfft_space, nfft_time]);
kf_shifted_filtered = fftshift(kf_spectrum_filtered, 1);

% 5. 生成频率和波数向量
% 频率向量 (Hz)
freq_vector_full = (0:nfft_time-1) * fs / nfft_time;

% 波数向量 (rad/m) - 沿y方向
delta_y = data_y(2) - data_y(1);  % y方向空间采样间隔
ky_vector = ((-round(nfft_space/2) + 1 : round(nfft_space/2)) / nfft_space) ...
            * 2*pi / delta_y;

% 6. 选择感兴趣的频率范围 (0 到 1 MHz)
max_freq = 1e6;  % 最大显示频率 (Hz)
[~, freq_max_index] = min(abs(freq_vector_full - max_freq));

% 截取数据
data_kf_original = kf_shifted_original(:, 1:freq_max_index);
data_kf_filtered = kf_shifted_filtered(:, 1:freq_max_index);
freq_display = freq_vector_full(1:freq_max_index);
ky_display = ky_vector;

fprintf('  显示频率范围: 0 - %.2f MHz\n', max_freq/1e6);
fprintf('  波数范围: %.2f - %.2f rad/mm\n', min(ky_vector)/1e3, max(ky_vector)/1e3);

% 7. 计算幅值谱
amp_original = abs(data_kf_original);
amp_filtered = abs(data_kf_filtered);

% 按频率逐列归一化
% 矩阵结构为 A(k, f)，对每一个固定频率(列)，使用最大值归一化该频率下的所有波数幅值
% 这有助于突出每个频率下的主要导波模态，忽略不同频率间的能量差异
max_original = max(amp_original, [], 1);
max_original(max_original == 0) = 1; % 防止除以零
amp_original = amp_original ./ max_original;

max_filtered = max(amp_filtered, [], 1);
max_filtered(max_filtered == 0) = 1; % 防止除以零
amp_filtered = amp_filtered ./ max_filtered;

% 8. 绘制频散曲线对比 (滤波前后)
figure('Name', '频散曲线对比', 'Position', [100, 100, 1400, 600]);

% 原始频散曲线
subplot(1, 2, 1);
surf(freq_display/1e3, -ky_vector/1e3, amp_original);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);  % 俯视图
xlabel('频率 (kHz)', 'FontSize', 12);
ylabel('波数 (rad/mm)', 'FontSize', 12);
title('原始频散曲线 (按频率归一化)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% 滤波后频散曲线
subplot(1, 2, 2);
surf(freq_display/1e3, -ky_vector/1e3, amp_filtered);
shading interp;
colormap(jet);
colorbar;
view([0, 90]);  % 俯视图
xlabel('频率 (kHz)', 'FontSize', 12);
ylabel('波数 (rad/mm)', 'FontSize', 12);
title(sprintf('滤波后频散曲线 (按频率归一化, %.0f-%.0f kHz)', lowcut*1e3, highcut*1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;

% 添加总标题
sgtitle('频散曲线滤波前后对比', ...
        'FontSize', 15, 'FontWeight', 'bold');