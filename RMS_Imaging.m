%% RMS成像程序
% 计算每个点的RMS值并进行可视化

clear; clc; close all;

%% ========== 参数配置区 ========== %%
% 所有可调参数集中在此处,方便修改

% 数据文件路径
data_file = 'data\10mm\100k\Scan_time.mat';

% 点阵参数
n_points = 45;           % 点阵大小 (31×31)
spacing = 1e-3;          % 物理间距 (m), 1mm

% 带通滤波器参数
center_freq = 100e3;     % 中心频率 (Hz)
bandwidth = 40e3;        % 带宽 (Hz)
filter_order = 4;        % 滤波器阶数

% 小波去噪参数
wavelet_name = 'db4';    % 小波基: 'db4', 'sym4', 'coif3'
wavelet_level = 5;       % 分解层数
threshold_method = 'soft'; % 阈值方法: 'soft' 或 'hard'

% 异常值处理参数
edge_margin = 2;         % 排除边缘点的层数
threshold_factor = 3;    % 异常值检测阈值 (标准差倍数)

% 插值参数
interp_factor = 4;       % 插值倍数 (用于提高显示分辨率)

%% ========== 数据预处理 ========== %%

% 加载原始数据
fprintf('正在加载数据: %s\n', data_file);
load(data_file); % 包含变量 x (时间) 和 y (扫描数据)

% 计算采样率和时间向量
data_time = x;
fs = 1/(data_time(2)-data_time(1));

% x, y方向坐标
data_x = (0:n_points-1) * spacing;
data_y = (0:n_points-1) * spacing;

% 将扫描数据重塑为 31×31×2500 的三维数组
data_xyt = zeros(n_points, n_points, length(data_time));

for col = 1:n_points
    start_idx = (col-1) * n_points + 1;
    end_idx = col * n_points;
    col_data = y(start_idx:end_idx, :);
    data_xyt(:, col, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d × %d\n', n_points, n_points);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);

%% ========== 应用带通滤波器 ========== %%

fprintf('\n========== 应用带通滤波器 ==========\n');
Filter.printInfo(center_freq, bandwidth, filter_order, fs);

data_xyt_filtered = zeros(size(data_xyt));

for i = 1:n_points
    for j = 1:n_points
        point_signal = squeeze(data_xyt(i, j, :));
        data_xyt_filtered(i, j, :) = Filter.apply(point_signal, fs, center_freq, bandwidth, filter_order);
    end
end

fprintf('滤波完成，共处理 %d 个空间点\n', n_points^2);

%% ========== 应用小波去噪 ========== %%

fprintf('\n========== 应用小波去噪 ==========\n');
Filter.printWaveletInfo(wavelet_name, wavelet_level, threshold_method);

data_xyt_wavelet = zeros(size(data_xyt));

fprintf('正在对滤波后的信号进行小波去噪...\n');
for i = 1:n_points
    for j = 1:n_points
        point_signal = squeeze(data_xyt_filtered(i, j, :));
        data_xyt_wavelet(i, j, :) = Filter.waveletDenoise(point_signal, wavelet_name, wavelet_level, threshold_method);
    end
end

fprintf('小波去噪完成，共处理 %d 个空间点\n', n_points^2);

%% ========== 计算RMS值 ========== %%

fprintf('\n正在计算RMS值...\n');

% 原始数据RMS
rms_image = zeros(n_points, n_points);
for i = 1:n_points
    for j = 1:n_points
        point_signal = squeeze(data_xyt(i, j, :));
        rms_image(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 滤波后数据RMS
rms_image_filtered = zeros(n_points, n_points);
for i = 1:n_points
    for j = 1:n_points
        point_signal = squeeze(data_xyt_filtered(i, j, :));
        rms_image_filtered(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 小波去噪后数据RMS
rms_image_wavelet = zeros(n_points, n_points);
for i = 1:n_points
    for j = 1:n_points
        point_signal = squeeze(data_xyt_wavelet(i, j, :));
        rms_image_wavelet(i, j) = sqrt(mean(point_signal.^2));
    end
end

fprintf('RMS计算完成\n');

%% ========== 异常值检测和处理 ========== %%

fprintf('\n正在检测和处理异常值...\n');

inner_range = (edge_margin+1):(n_points-edge_margin);

% 处理原始数据
rms_inner = rms_image(inner_range, inner_range);
rms_mean = mean(rms_inner(:));
rms_std = std(rms_inner(:));
lower_threshold = rms_mean - threshold_factor * rms_std;
upper_threshold = rms_mean + threshold_factor * rms_std;

outlier_mask = (rms_image < lower_threshold) | (rms_image > upper_threshold);
rms_image_processed = rms_image;
rms_image_processed(outlier_mask) = NaN;

if sum(outlier_mask(:)) > 0
    rms_image_repaired = rms_image_processed;
    for i = 1:n_points
        for j = 1:n_points
            if isnan(rms_image_processed(i, j))
                i_min = max(1, i-1);
                i_max = min(n_points, i+1);
                j_min = max(1, j-1);
                j_max = min(n_points, j+1);
                neighbors = rms_image_processed(i_min:i_max, j_min:j_max);
                valid_neighbors = neighbors(~isnan(neighbors));
                if ~isempty(valid_neighbors)
                    rms_image_repaired(i, j) = median(valid_neighbors);
                else
                    rms_image_repaired(i, j) = rms_mean;
                end
            end
        end
    end
    rms_image = rms_image_repaired;
else
    rms_image = rms_image_processed;
end

% 处理滤波后数据
rms_inner_filtered = rms_image_filtered(inner_range, inner_range);
rms_mean_filtered = mean(rms_inner_filtered(:));
rms_std_filtered = std(rms_inner_filtered(:));
lower_threshold_filtered = rms_mean_filtered - threshold_factor * rms_std_filtered;
upper_threshold_filtered = rms_mean_filtered + threshold_factor * rms_std_filtered;

outlier_mask_filtered = (rms_image_filtered < lower_threshold_filtered) | (rms_image_filtered > upper_threshold_filtered);
rms_image_filtered_processed = rms_image_filtered;
rms_image_filtered_processed(outlier_mask_filtered) = NaN;

if sum(outlier_mask_filtered(:)) > 0
    rms_image_filtered_repaired = rms_image_filtered_processed;
    for i = 1:n_points
        for j = 1:n_points
            if isnan(rms_image_filtered_processed(i, j))
                i_min = max(1, i-1);
                i_max = min(n_points, i+1);
                j_min = max(1, j-1);
                j_max = min(n_points, j+1);
                neighbors = rms_image_filtered_processed(i_min:i_max, j_min:j_max);
                valid_neighbors = neighbors(~isnan(neighbors));
                if ~isempty(valid_neighbors)
                    rms_image_filtered_repaired(i, j) = median(valid_neighbors);
                else
                    rms_image_filtered_repaired(i, j) = rms_mean_filtered;
                end
            end
        end
    end
    rms_image_filtered = rms_image_filtered_repaired;
else
    rms_image_filtered = rms_image_filtered_processed;
end

% 处理小波去噪后数据
rms_inner_wavelet = rms_image_wavelet(inner_range, inner_range);
rms_mean_wavelet = mean(rms_inner_wavelet(:));
rms_std_wavelet = std(rms_inner_wavelet(:));
lower_threshold_wavelet = rms_mean_wavelet - threshold_factor * rms_std_wavelet;
upper_threshold_wavelet = rms_mean_wavelet + threshold_factor * rms_std_wavelet;

outlier_mask_wavelet = (rms_image_wavelet < lower_threshold_wavelet) | (rms_image_wavelet > upper_threshold_wavelet);
rms_image_wavelet_processed = rms_image_wavelet;
rms_image_wavelet_processed(outlier_mask_wavelet) = NaN;

if sum(outlier_mask_wavelet(:)) > 0
    rms_image_wavelet_repaired = rms_image_wavelet_processed;
    for i = 1:n_points
        for j = 1:n_points
            if isnan(rms_image_wavelet_processed(i, j))
                i_min = max(1, i-1);
                i_max = min(n_points, i+1);
                j_min = max(1, j-1);
                j_max = min(n_points, j+1);
                neighbors = rms_image_wavelet_processed(i_min:i_max, j_min:j_max);
                valid_neighbors = neighbors(~isnan(neighbors));
                if ~isempty(valid_neighbors)
                    rms_image_wavelet_repaired(i, j) = median(valid_neighbors);
                else
                    rms_image_wavelet_repaired(i, j) = rms_mean_wavelet;
                end
            end
        end
    end
    rms_image_wavelet = rms_image_wavelet_repaired;
else
    rms_image_wavelet = rms_image_wavelet_processed;
end

fprintf('异常值处理完成\n');

%% ========== 插值提高显示分辨率 ========== %%

fprintf('\n正在进行 %d 倍插值...\n', interp_factor);

[X_orig, Y_orig] = meshgrid(data_x, data_y);

x_interp = linspace(data_x(1), data_x(end), n_points * interp_factor - (interp_factor - 1));
y_interp = linspace(data_y(1), data_y(end), n_points * interp_factor - (interp_factor - 1));
[X_interp, Y_interp] = meshgrid(x_interp, y_interp);

rms_image_interp = interp2(X_orig, Y_orig, rms_image, X_interp, Y_interp, 'linear');
rms_image_filtered_interp = interp2(X_orig, Y_orig, rms_image_filtered, X_interp, Y_interp, 'linear');
rms_image_wavelet_interp = interp2(X_orig, Y_orig, rms_image_wavelet, X_interp, Y_interp, 'linear');

fprintf('插值完成\n');

%% ========== 可视化: RMS成像对比 ========== %%

figure('Name', 'RMS成像对比 (原始 vs 滤波 vs 小波去噪)', 'Position', [50, 100, 1800, 550]);

% 左图：原始RMS成像
subplot(1, 3, 1);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title('原始RMS成像', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

% 中图：滤波后RMS成像
subplot(1, 3, 2);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_filtered_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title(sprintf('带通滤波 (%.0f-%.0f kHz)', ...
              (center_freq-bandwidth/2)/1e3, (center_freq+bandwidth/2)/1e3), ...
      'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

% 右图：小波去噪后RMS成像
subplot(1, 3, 3);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_wavelet_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title(sprintf('带通滤波 + 小波去噪 (%s)', wavelet_name), ...
      'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

%% ========== 可视化: 时域波形对比 ========== %%

fprintf('\n正在生成时域波形对比...\n');

% 随机选择一个点 (避免边缘点)
rng('shuffle');
random_i = randi([edge_margin+1, n_points-edge_margin]);
random_j = randi([edge_margin+1, n_points-edge_margin]);

fprintf('  随机选择的点: (%d, %d)\n', random_i, random_j);
fprintf('  物理位置: (%.2f mm, %.2f mm)\n', data_x(random_j)*1e3, data_y(random_i)*1e3);

% 提取该点的三种处理信号
signal_original = squeeze(data_xyt(random_i, random_j, :));
signal_filtered = squeeze(data_xyt_filtered(random_i, random_j, :));
signal_wavelet = squeeze(data_xyt_wavelet(random_i, random_j, :));

% 创建时域对比图
figure('Name', '随机点时域波形对比 (三种处理方法)', 'Position', [100, 100, 1800, 500]);

% 原始信号
subplot(1, 3, 1);
plot(data_time * 1e6, signal_original, 'b-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('原始信号 [位置: (%.1f, %.1f) mm]', data_x(random_j)*1e3, data_y(random_i)*1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 滤波后信号
subplot(1, 3, 2);
plot(data_time * 1e6, signal_filtered, 'r-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('带通滤波 (%.0f-%.0f kHz)', ...
              (center_freq-bandwidth/2)/1e3, (center_freq+bandwidth/2)/1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 小波去噪后信号
subplot(1, 3, 3);
plot(data_time * 1e6, signal_wavelet, 'g-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('带通滤波 + 小波去噪 (%s)', wavelet_name), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

fprintf('\n成像完成！\n');
