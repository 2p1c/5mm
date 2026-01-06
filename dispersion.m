%% 数据预处理：加载并重塑蛇形扫描数据
% 加载原始数据
load('data\5mm\100k\Scan_time.mat'); % 包含变量 x (1×2500) 和 y (961×2500)

% 计算采样率和时间向量
data_time = x; % 时间向量
fs = 1/(data_time(2)-data_time(1)); % 采样率 (Hz)

% 设置点阵参数
n_points = 31; % 31×31 点阵
spacing = 1e-3; % 物理间距 1mm = 0.001m
data_x = (0:n_points-1) * spacing; % x方向坐标 (m)

% 将蛇形扫描数据重塑为 31×31×2500 的三维数组
data_xyt = zeros(n_points, n_points, length(data_time));

for col = 1:n_points
    % 计算当前列在y数组中的起始和结束索引
    start_idx = (col-1) * n_points + 1;
    end_idx = col * n_points;
    
    % 提取当前列的数据
    col_data = y(start_idx:end_idx, :);
    
    % 根据列数决定是否翻转（偶数列从下到上）
    if mod(col, 2) == 0
        col_data = flipud(col_data); % 翻转偶数列
    end
    
    % 存储到三维数组中
    data_xyt(col, :, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d × %d\n', n_points, n_points);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  数据形状: %s\n', mat2str(size(data_xyt)));

%% 随机选择一个点并分析其时域和频域特性
% 随机选择点的坐标
rand_x = randi(n_points);
rand_y = randi(n_points);

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

% 可视化
figure('Name', '随机点信号分析', 'Position', [100, 100, 1200, 500]);

% 时域信号
subplot(1, 2, 1);
plot(data_time * 1e6, point_signal, 'LineWidth', 1);
xlabel('时间 (μs)');
ylabel('幅值');
title(sprintf('时域信号 - 位置: (%d, %d)', rand_x, rand_y));
grid on;

% 频域信号
subplot(1, 2, 2);
plot(freq_vector_pos / 1e6, amplitude_spectrum, 'LineWidth', 1);
xlabel('频率 (MHz)');
ylabel('幅值');
title('频谱');
grid on;
xlim([0, fs/2/1e6]); % 显示到奈奎斯特频率

fprintf('\n随机点分析:\n');
fprintf('  选择的点: (%d, %d)\n', rand_x, rand_y);
fprintf('  物理坐标: (%.2f mm, %.2f mm)\n', data_x(rand_x)*1e3, data_x(rand_y)*1e3);
fprintf('  信号RMS: %.4e\n', rms(point_signal));
fprintf('  信号峰值: %.4e\n', max(abs(point_signal)));
[~, max_freq_idx] = max(amplitude_spectrum);
fprintf('  主频: %.2f MHz\n', freq_vector_pos(max_freq_idx)/1e6);

%% 频散曲线计算（f-k域分析）

% 1. 提取中间一行数据 (y=16) 进行空间-时间二维分析
middle_row_index = 16;
data_xt = permute(data_xyt(:, middle_row_index, :), [1, 3, 2]);  % [空间 × 时间]

fprintf('\n频散曲线计算:\n');
fprintf('  使用第 %d 行数据\n', middle_row_index);

% 2. 设置FFT参数
% 使用零填充提高分辨率（增加一倍）
nfft_space = 2^(nextpow2(length(data_x)) + 1);      % 空间维度FFT点数
nfft_time = 2^(nextpow2(length(data_time)) + 1);    % 时间维度FFT点数

fprintf('  FFT点数: 空间=%d, 时间=%d\n', nfft_space, nfft_time);

% 3. 二维傅里叶变换：空间-时间 → 波数-频率
kf_spectrum = fftn(data_xt, [nfft_space, nfft_time]);

% 4. 对空间维度进行fftshift，使零波数居中
kf_shifted = fftshift(kf_spectrum, 1);

% 5. 生成频率和波数向量
% 频率向量 (Hz)
freq_vector_full = (0:nfft_time-1) * fs / nfft_time;

% 波数向量 (rad/m)
delta_x = data_x(2) - data_x(1);  % 空间采样间隔
kx_vector = ((-round(nfft_space/2) + 1 : round(nfft_space/2)) / nfft_space) ...
            * 2*pi / delta_x;

% 6. 选择感兴趣的频率范围 (0 到 6 MHz)
max_freq = 2e6;  % 最大显示频率 (Hz)
[~, freq_max_index] = min(abs(freq_vector_full - max_freq));

% 截取数据
data_kf = kf_shifted(:, 1:freq_max_index);
freq_display = freq_vector_full(1:freq_max_index);
kx_display = kx_vector;

% 7. 去除负频率部分（仅保留正频率）
data_kf(:, nfft_time/2+1:end) = 0;

fprintf('  显示频率范围: 0 - %.2f MHz\n', max_freq/1e6);
fprintf('  波数范围: %.2f - %.2f rad/mm\n', min(kx_display)/1e3, max(kx_display)/1e3);

% 8. 绘制频散曲线 (f-k谱图)
figure('Name', '频散曲线 (f-k谱)', 'Position', [100, 100, 800, 600]);
surf(freq_display, -kx_display, abs(data_kf));
shading interp;
colorbar;
view([0, 90]);  % 俯视图

% 坐标轴标签
xlabel('\fontname{宋体}\fontsize{20}频率\fontname{Times New Roman}\fontsize{20} / kHz');
ylabel('\fontname{宋体}\fontsize{20}波数\fontname{Times New Roman}\fontsize{20} /  rad·mm^{-1}');
title('频散曲线 (频率-波数谱)', 'FontSize', 16);

% 可选：设置坐标轴范围
% ylim([-4, 4]);
% xlim([0, 2e6]);