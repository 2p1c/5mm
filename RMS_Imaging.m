%% RMS成像程序
% 计算每个点的RMS值并进行可视化

clear; clc; close all;

%% ========== 参数配置区 ========== %%
% 所有可调参数集中在此处,方便修改

% 数据文件路径
data_file = "E:\数据\260127\kongou200k2\51_51.mat";

% 点阵参数
n_points = [51, 51];     % 点阵大小: 标量(正方形) 或 [n_x, n_y](矩形)
spacing = 1e-3;          % 物理间距 (m), 1mm

% 带通滤波器参数
center_freq = 200e3;     % 中心频率 (Hz)
bandwidth = 20e3;        % 带宽 (Hz)
filter_order = 2;        % 滤波器阶数

% 小波去噪参数
wavelet_name = 'coif3';    % 小波基: 'db4', 'sym4', 'coif3'
wavelet_level = 3;       % 分解层数
threshold_method = 'soft'; % 阈值方法: 'soft' 或 'hard'

% 异常值处理参数
edge_margin = 2;         % 排除边缘点的层数
threshold_factor = 3;    % 异常值检测阈值 (标准差倍数)

% 插值参数
interp_factor = 4;       % 插值倍数 (用于提高显示分辨率)

% 衰减补偿参数
enable_attenuation_compensation = true;  % 是否启用衰减补偿
attenuation_method = 'spatial';          % 补偿方法: 'spatial'(基于空间), 'temporal'(基于时间), 'combined'(混合)
attenuation_coefficient = 0.005;          % 衰减系数 (1/mm 或 1/μs, 根据方法而定)
source_position = [25, 60] * 1e-3;       % 波源位置 [x, y] (米), 默认在Y轴下方中心
min_compensation_distance = 10e-3;       % 最小补偿距离 (米), 小于此距离的区域增益为1（不补偿）

% 时间段选择参数
enable_time_window = true;               % 是否启用时间段截取
time_window_start = 100;                   % 起始时间 (μs)
time_window_end = 300;                   % 结束时间 (μs)

%% ========== 数据预处理 ========== %%

% 加载原始数据
[data_xyt_full, data_time_full, data_x, data_y, fs] = mat_loader(data_file, n_points, spacing);

% 应用时间段截取
if enable_time_window
    fprintf('\n========== 应用时间段截取 ==========\n');
    fprintf('截取范围: %.1f - %.1f μs\n', time_window_start, time_window_end);
    
    % 找到时间窗口对应的索引
    time_us = data_time_full * 1e6;  % 转换为微秒
    idx_start = find(time_us >= time_window_start, 1, 'first');
    idx_end = find(time_us <= time_window_end, 1, 'last');
    
    if isempty(idx_start) || isempty(idx_end) || idx_start >= idx_end
        error('时间窗口范围无效！请检查 time_window_start 和 time_window_end 参数');
    end
    
    % 截取数据
    data_time = data_time_full(idx_start:idx_end);
    data_xyt = data_xyt_full(:, :, idx_start:idx_end);
    
    fprintf('原始时间点数: %d\n', length(data_time_full));
    fprintf('截取后时间点数: %d\n', length(data_time));
    fprintf('实际截取范围: %.2f - %.2f μs\n', data_time(1)*1e6, data_time(end)*1e6);
else
    data_time = data_time_full;
    data_xyt = data_xyt_full;
    fprintf('\n时间段截取已禁用，使用完整时间范围\n');
end

% 提取点阵尺寸
[n_y, n_x, ~] = size(data_xyt);

%% ========== 应用带通滤波器 ========== %%

fprintf('\n========== 应用带通滤波器 ==========\n');
Filter.printInfo(center_freq, bandwidth, filter_order, fs);

data_xyt_filtered = zeros(size(data_xyt));

for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt(i, j, :));
        data_xyt_filtered(i, j, :) = Filter.apply(point_signal, fs, center_freq, bandwidth, filter_order);
    end
end

fprintf('滤波完成，共处理 %d 个空间点\n', n_x * n_y);

%% ========== 应用小波去噪 ========== %%

fprintf('\n========== 应用小波去噪 ==========\n');
Filter.printWaveletInfo(wavelet_name, wavelet_level, threshold_method);

data_xyt_wavelet = zeros(size(data_xyt));

fprintf('正在对滤波后的信号进行小波去噪...\n');
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt_filtered(i, j, :));
        data_xyt_wavelet(i, j, :) = Filter.waveletDenoise(point_signal, wavelet_name, wavelet_level, threshold_method);
    end
end

fprintf('小波去噪完成，共处理 %d 个空间点\n', n_x * n_y);

%% ========== 应用衰减补偿 ========== %%

if enable_attenuation_compensation
    fprintf('\n========== 应用衰减补偿 ==========\n');
    fprintf('补偿方法: %s\n', attenuation_method);
    fprintf('衰减系数: %.4f\n', attenuation_coefficient);
    fprintf('波源位置: (%.1f, %.1f) mm\n', source_position(1)*1e3, source_position(2)*1e3);
    
    data_xyt_compensated = zeros(size(data_xyt_wavelet));
    
    switch attenuation_method
        case 'spatial'
            % 基于空间距离的补偿: gain(i,j) = exp(alpha * max(0, distance - min_distance))
            % 为每个空间点计算到波源的距离
            fprintf('使用空间距离补偿: gain = exp(α * max(0, d - d_min))\n');
            fprintf('最小补偿距离: %.1f mm\n', min_compensation_distance * 1e3);
            
            % 创建空间增益图
            gain_map = zeros(n_y, n_x);
            for i = 1:n_y

                for j = 1:n_x
                    % 计算当前点到波源的距离 (米)
                    dx = data_x(j) - source_position(1);
                    dy = data_y(i) - source_position(2);
                    distance = sqrt(dx^2 + dy^2);
                    
                    % 计算有效补偿距离（仅补偿超过最小距离的部分）
                    effective_distance = max(0, distance - min_compensation_distance);
                    
                    % 计算补偿增益 (距离转换为mm)
                    gain_map(i, j) = exp(attenuation_coefficient * effective_distance * 1e3);
                end
            end
            
            % 计算最大距离（使用meshgrid处理矩形点阵）
            [X_grid, Y_grid] = meshgrid(data_x, data_y);
            all_distances = sqrt((X_grid - source_position(1)).^2 + (Y_grid - source_position(2)).^2);
            fprintf('距离范围: %.1f - %.1f mm\n', min(all_distances(:))*1e3, max(all_distances(:))*1e3);
            fprintf('增益范围: %.2f - %.2f\n', min(gain_map(:)), max(gain_map(:)));
            
            % 对每个空间点应用对应的增益
            fprintf('正在对小波去噪后的信号进行空间衰减补偿...\n');
            for i = 1:n_y

                for j = 1:n_x
                    point_signal = squeeze(data_xyt_wavelet(i, j, :));
                    data_xyt_compensated(i, j, :) = point_signal * gain_map(i, j);
                end
            end
            
            % 显示空间增益分布图
            figure('Name', '空间衰减补偿增益分布', 'Position', [100, 100, 800, 600]);
            imagesc(data_x * 1e3, data_y * 1e3, gain_map);
            axis equal tight;
            colormap('jet');
            colorbar;
            xlabel('X 位置 (mm)', 'FontSize', 12);
            ylabel('Y 位置 (mm)', 'FontSize', 12);
            title(sprintf('空间补偿增益分布 (α=%.4f /mm)', attenuation_coefficient), ...
                  'FontSize', 14, 'FontWeight', 'bold');
            hold on;
            plot(source_position(1)*1e3, source_position(2)*1e3, 'w*', 'MarkerSize', 15, 'LineWidth', 2);
            text(source_position(1)*1e3, source_position(2)*1e3 + 2, '波源', ...
                 'Color', 'white', 'FontSize', 12, 'HorizontalAlignment', 'center');
            grid on;
            set(gca, 'YDir', 'normal');
            
        case 'temporal'
            % 基于时间的补偿 (原有方法)
            fprintf('使用时间补偿: gain = exp(α * t_μs)\n');
            time_vector_us = data_time * 1e6;
            gain = exp(attenuation_coefficient * time_vector_us);
            gain = gain(:);
            
            fprintf('时间范围: %.1f - %.1f μs, 增益范围: %.2f - %.2f\n', ...
                    time_vector_us(1), time_vector_us(end), gain(1), gain(end));
            
            fprintf('正在对小波去噪后的信号进行时间衰减补偿...\n');
            for i = 1:n_y

                for j = 1:n_x
                    point_signal = squeeze(data_xyt_wavelet(i, j, :));
                    point_signal = point_signal(:);
                    data_xyt_compensated(i, j, :) = point_signal .* gain;
                end
            end
            
            % 显示时间增益曲线
            figure('Name', '时间衰减补偿增益曲线', 'Position', [100, 100, 800, 400]);
            plot(data_time * 1e6, gain, 'b-', 'LineWidth', 2);
            xlabel('时间 (μs)', 'FontSize', 12);
            ylabel('补偿增益', 'FontSize', 12);
            title(sprintf('时间补偿增益函数 (α=%.4f /μs)', attenuation_coefficient), ...
                  'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            xlim([data_time(1), data_time(end)] * 1e6);
            
        case 'combined'
            % 混合补偿: 空间 × 时间
            fprintf('使用混合补偿: gain = exp(α * (distance + c*t))\n');
            
            % 计算声速 (假设)
            sound_speed = 3000; % m/s, 根据材料调整
            time_vector_us = data_time * 1e6;
            
            fprintf('正在对小波去噪后的信号进行混合衰减补偿...\n');
            for i = 1:n_y

                for j = 1:n_x
                    % 空间距离
                    dx = data_x(j) - source_position(1);
                    dy = data_y(i) - source_position(2);
                    distance = sqrt(dx^2 + dy^2);
                    
                    % 计算有效补偿距离（仅补偿超过最小距离的部分）
                    effective_distance_mm = max(0, distance - min_compensation_distance) * 1e3;
                    
                    % 空间增益（标量）
                    spatial_gain = exp(attenuation_coefficient * effective_distance_mm * 0.5);
                    
                    % 时间增益（列向量）
                    time_gain = exp(attenuation_coefficient * time_vector_us * 0.5);
                    time_gain = time_gain(:);
                    
                    point_signal = squeeze(data_xyt_wavelet(i, j, :));
                    point_signal = point_signal(:);
                    data_xyt_compensated(i, j, :) = point_signal .* time_gain * spatial_gain;
                end
            end
            
        otherwise
            error('未知的衰减补偿方法: %s (可选: spatial, temporal, combined)', attenuation_method);
    end
    
    fprintf('衰减补偿完成，共处理 %d 个空间点\n', n_x * n_y);
    
else
    fprintf('\n衰减补偿已禁用\n');
    data_xyt_compensated = data_xyt_wavelet;
end

%% ========== 计算RMS值 ========== %%

fprintf('\n正在计算RMS值...\n');

% 原始数据RMS
rms_image = zeros(n_y, n_x);
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt(i, j, :));
        rms_image(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 滤波后数据RMS
rms_image_filtered = zeros(n_y, n_x);
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt_filtered(i, j, :));
        rms_image_filtered(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 小波去噪后数据RMS
rms_image_wavelet = zeros(n_y, n_x);
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt_wavelet(i, j, :));
        rms_image_wavelet(i, j) = sqrt(mean(point_signal.^2));
    end
end

% 衰减补偿后数据RMS
rms_image_compensated = zeros(n_y, n_x);
for i = 1:n_y
    for j = 1:n_x
        point_signal = squeeze(data_xyt_compensated(i, j, :));
        rms_image_compensated(i, j) = sqrt(mean(point_signal.^2));
    end
end

fprintf('RMS计算完成\n');

%% ========== 异常值检测和处理 ========== %%

fprintf('\n正在检测和处理异常值...\n');

inner_range_y = (edge_margin+1):(n_y-edge_margin);
inner_range_x = (edge_margin+1):(n_x-edge_margin);

% 处理原始数据
rms_inner = rms_image(inner_range_y, inner_range_x);
rms_mean = mean(rms_inner(:));
rms_std = std(rms_inner(:));
lower_threshold = rms_mean - threshold_factor * rms_std;
upper_threshold = rms_mean + threshold_factor * rms_std;

outlier_mask = (rms_image < lower_threshold) | (rms_image > upper_threshold);
rms_image_processed = rms_image;
rms_image_processed(outlier_mask) = NaN;

if sum(outlier_mask(:)) > 0
    rms_image_repaired = rms_image_processed;
    for i = 1:n_y

        for j = 1:n_x
            if isnan(rms_image_processed(i, j))
                i_min = max(1, i-1);
                i_max = min(n_y, i+1);
                j_min = max(1, j-1);
                j_max = min(n_x, j+1);
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
rms_inner_filtered = rms_image_filtered(inner_range_y, inner_range_x);
rms_mean_filtered = mean(rms_inner_filtered(:));
rms_std_filtered = std(rms_inner_filtered(:));
lower_threshold_filtered = rms_mean_filtered - threshold_factor * rms_std_filtered;
upper_threshold_filtered = rms_mean_filtered + threshold_factor * rms_std_filtered;

outlier_mask_filtered = (rms_image_filtered < lower_threshold_filtered) | (rms_image_filtered > upper_threshold_filtered);
rms_image_filtered_processed = rms_image_filtered;
rms_image_filtered_processed(outlier_mask_filtered) = NaN;

if sum(outlier_mask_filtered(:)) > 0
    rms_image_filtered_repaired = rms_image_filtered_processed;
    for i = 1:n_y

        for j = 1:n_x
            if isnan(rms_image_filtered_processed(i, j))
                i_min = max(1, i-1);
                i_max = min(n_y, i+1);
                j_min = max(1, j-1);
                j_max = min(n_x, j+1);
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
rms_inner_wavelet = rms_image_wavelet(inner_range_y, inner_range_x);
rms_mean_wavelet = mean(rms_inner_wavelet(:));
rms_std_wavelet = std(rms_inner_wavelet(:));
lower_threshold_wavelet = rms_mean_wavelet - threshold_factor * rms_std_wavelet;
upper_threshold_wavelet = rms_mean_wavelet + threshold_factor * rms_std_wavelet;

outlier_mask_wavelet = (rms_image_wavelet < lower_threshold_wavelet) | (rms_image_wavelet > upper_threshold_wavelet);
rms_image_wavelet_processed = rms_image_wavelet;
rms_image_wavelet_processed(outlier_mask_wavelet) = NaN;

if sum(outlier_mask_wavelet(:)) > 0
    rms_image_wavelet_repaired = rms_image_wavelet_processed;
    for i = 1:n_y

        for j = 1:n_x
            if isnan(rms_image_wavelet_processed(i, j))
                i_min = max(1, i-1);
                i_max = min(n_y, i+1);
                j_min = max(1, j-1);
                j_max = min(n_x, j+1);
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

% 处理衰减补偿后数据
rms_inner_compensated = rms_image_compensated(inner_range_y, inner_range_x);
rms_mean_compensated = mean(rms_inner_compensated(:));
rms_std_compensated = std(rms_inner_compensated(:));
lower_threshold_compensated = rms_mean_compensated - threshold_factor * rms_std_compensated;
upper_threshold_compensated = rms_mean_compensated + threshold_factor * rms_std_compensated;

outlier_mask_compensated = (rms_image_compensated < lower_threshold_compensated) | (rms_image_compensated > upper_threshold_compensated);
rms_image_compensated_processed = rms_image_compensated;
rms_image_compensated_processed(outlier_mask_compensated) = NaN;

if sum(outlier_mask_compensated(:)) > 0
    rms_image_compensated_repaired = rms_image_compensated_processed;
    for i = 1:n_y

        for j = 1:n_x
            if isnan(rms_image_compensated_processed(i, j))
                i_min = max(1, i-1);
                i_max = min(n_y, i+1);
                j_min = max(1, j-1);
                j_max = min(n_x, j+1);
                neighbors = rms_image_compensated_processed(i_min:i_max, j_min:j_max);
                valid_neighbors = neighbors(~isnan(neighbors));
                if ~isempty(valid_neighbors)
                    rms_image_compensated_repaired(i, j) = median(valid_neighbors);
                else
                    rms_image_compensated_repaired(i, j) = rms_mean_compensated;
                end
            end
        end
    end
    rms_image_compensated = rms_image_compensated_repaired;
else
    rms_image_compensated = rms_image_compensated_processed;
end

fprintf('异常值处理完成\n');

%% ========== 插值提高显示分辨率 ========== %%

fprintf('\n正在进行 %d 倍插值...\n', interp_factor);

[X_orig, Y_orig] = meshgrid(data_x, data_y);

x_interp = linspace(data_x(1), data_x(end), n_x * interp_factor - (interp_factor - 1));
y_interp = linspace(data_y(1), data_y(end), n_y * interp_factor - (interp_factor - 1));
[X_interp, Y_interp] = meshgrid(x_interp, y_interp);

rms_image_interp = interp2(X_orig, Y_orig, rms_image, X_interp, Y_interp, 'linear');
rms_image_filtered_interp = interp2(X_orig, Y_orig, rms_image_filtered, X_interp, Y_interp, 'linear');
rms_image_wavelet_interp = interp2(X_orig, Y_orig, rms_image_wavelet, X_interp, Y_interp, 'linear');
rms_image_compensated_interp = interp2(X_orig, Y_orig, rms_image_compensated, X_interp, Y_interp, 'linear');

fprintf('插值完成\n');

%% ========== 可视化: RMS成像对比 ========== %%

figure('Name', 'RMS成像对比 (原始 vs 滤波 vs 小波去噪 vs 衰减补偿)', 'Position', [50, 50, 1600, 900]);

% 左上图：原始RMS成像
subplot(2, 2, 1);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title('原始RMS成像', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

% 右上图：滤波后RMS成像
subplot(2, 2, 2);
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

% 左下图：小波去噪后RMS成像
subplot(2, 2, 3);
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

% 右下图：衰减补偿后RMS成像
subplot(2, 2, 4);
imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_compensated_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
if enable_attenuation_compensation
    title(sprintf('小波去噪 + 衰减补偿 (%s, α=%.2f)', attenuation_method, attenuation_coefficient), ...
          'FontSize', 14, 'FontWeight', 'bold');
else
    title('衰减补偿 (未启用)', 'FontSize', 14, 'FontWeight', 'bold');
end
grid on;
set(gca, 'YDir', 'normal');

%% ========== 可视化: 时域波形对比 ========== %%

fprintf('\n正在生成时域波形对比...\n');

% 随机选择一个点 (避免边缘点)
rng('shuffle');
random_i = randi([edge_margin+1, n_y-edge_margin]);
random_j = randi([edge_margin+1, n_x-edge_margin]);

fprintf('  随机选择的点: (%d, %d)\n', random_i, random_j);
fprintf('  物理位置: (%.2f mm, %.2f mm)\n', data_x(random_j)*1e3, data_y(random_i)*1e3);

% 提取该点的四种处理信号
signal_original = squeeze(data_xyt(random_i, random_j, :));
signal_filtered = squeeze(data_xyt_filtered(random_i, random_j, :));
signal_wavelet = squeeze(data_xyt_wavelet(random_i, random_j, :));
signal_compensated = squeeze(data_xyt_compensated(random_i, random_j, :));

% 创建时域对比图
figure('Name', '随机点时域波形对比 (四种处理方法)', 'Position', [100, 100, 1600, 800]);

% 原始信号
subplot(2, 2, 1);
plot(data_time * 1e6, signal_original, 'b-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('原始信号 [位置: (%.1f, %.1f) mm]', data_x(random_j)*1e3, data_y(random_i)*1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 滤波后信号
subplot(2, 2, 2);
plot(data_time * 1e6, signal_filtered, 'r-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('带通滤波 (%.0f-%.0f kHz)', ...
              (center_freq-bandwidth/2)/1e3, (center_freq+bandwidth/2)/1e3), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 小波去噪后信号
subplot(2, 2, 3);
plot(data_time * 1e6, signal_wavelet, 'g-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
title(sprintf('带通滤波 + 小波去噪 (%s)', wavelet_name), ...
      'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

% 衰减补偿后信号
subplot(2, 2, 4);
plot(data_time * 1e6, signal_compensated, 'm-', 'LineWidth', 1.0);
xlabel('时间 (μs)', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
if enable_attenuation_compensation
    title(sprintf('小波去噪 + 衰减补偿 (%s, α=%.2f)', attenuation_method, attenuation_coefficient), ...
          'FontSize', 13, 'FontWeight', 'bold');
else
    title('衰减补偿 (未启用)', 'FontSize', 13, 'FontWeight', 'bold');
end
grid on;
xlim([data_time(1), data_time(end)] * 1e6);

fprintf('\n成像完成！\n');



