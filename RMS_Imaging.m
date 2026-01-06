%% RMS成像程序
% 计算每个点的RMS值并进行可视化

clear; clc; close all;

%% 数据预处理：加载并重塑扫描数据
% 加载原始数据
load('data\5mm\100k\Scan_time.mat'); % 包含变量 x (1×2500) 和 y (961×2500)

% 计算采样率和时间向量
data_time = x; % 时间向量
fs = 1/(data_time(2)-data_time(1)); % 采样率 (Hz)

% 设置点阵参数
n_points = 31; % 31×31 点阵
spacing = 1e-3; % 物理间距 1mm = 0.001m
data_x = (0:n_points-1) * spacing; % x方向坐标 (m)
data_y = (0:n_points-1) * spacing; % y方向坐标 (m)

% 将扫描数据重塑为 31×31×2500 的三维数组（不进行蛇形校正）
data_xyt = zeros(n_points, n_points, length(data_time));

for col = 1:n_points
    % 计算当前列在y数组中的起始和结束索引
    start_idx = (col-1) * n_points + 1;
    end_idx = col * n_points;
    
    % 提取当前列的数据
    col_data = y(start_idx:end_idx, :);
    
    % 直接存储，不进行蛇形校正
    data_xyt(:, col, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d × %d\n', n_points, n_points);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  数据形状: %s\n', mat2str(size(data_xyt)));

%% 计算每个点的RMS值
fprintf('\n正在计算RMS值...\n');

% 初始化RMS图像矩阵
rms_image = zeros(n_points, n_points);

% 对每个空间点计算其时间序列的RMS值
for i = 1:n_points
    for j = 1:n_points
        % 提取该点的时间序列
        point_signal = squeeze(data_xyt(i, j, :));
        
        % 计算RMS值: RMS = sqrt(mean(signal^2))
        rms_image(i, j) = sqrt(mean(point_signal.^2));
    end
end

fprintf('RMS计算完成\n');
fprintf('  RMS最小值: %.6e\n', min(rms_image(:)));
fprintf('  RMS最大值: %.6e\n', max(rms_image(:)));
fprintf('  RMS平均值: %.6e\n', mean(rms_image(:)));

%% 异常值检测和处理
fprintf('\n正在检测和处理异常值...\n');

% 方法1: 排除边缘点（外围一圈）
edge_margin = 2;  % 排除外围2层的点
inner_range = (edge_margin+1):(n_points-edge_margin);
rms_inner = rms_image(inner_range, inner_range);

fprintf('  边缘排除: 外围 %d 层\n', edge_margin);
fprintf('  有效区域: %d × %d\n', length(inner_range), length(inner_range));

% 方法2: 基于统计的异常值检测（使用内部区域的统计）
rms_mean = mean(rms_inner(:));
rms_std = std(rms_inner(:));
threshold_factor = 3;  % 3倍标准差

% 定义正常值范围
lower_threshold = rms_mean - threshold_factor * rms_std;
upper_threshold = rms_mean + threshold_factor * rms_std;

fprintf('  统计阈值: 均值 ± %.1f倍标准差\n', threshold_factor);
fprintf('  正常范围: [%.6e, %.6e]\n', lower_threshold, upper_threshold);

% 创建处理后的RMS图像
rms_image_filtered = rms_image;

% 标记异常值为NaN
outlier_mask = (rms_image < lower_threshold) | (rms_image > upper_threshold);
rms_image_filtered(outlier_mask) = NaN;

% 统计异常值数量
num_outliers = sum(outlier_mask(:));
fprintf('  检测到异常值: %d / %d (%.1f%%)\n', num_outliers, numel(rms_image), 100*num_outliers/numel(rms_image));

% 对异常值进行插值修复（使用邻近正常值的平均）
if num_outliers > 0
    fprintf('  正在修复异常值...\n');
    
    % 使用inpaint_nans函数或简单的邻近插值
    % 这里使用简单的方法：用周围8个邻居的中值替代
    rms_image_repaired = rms_image_filtered;
    
    for i = 1:n_points
        for j = 1:n_points
            if isnan(rms_image_filtered(i, j))
                % 获取邻近点（3×3窗口）
                i_min = max(1, i-1);
                i_max = min(n_points, i+1);
                j_min = max(1, j-1);
                j_max = min(n_points, j+1);
                
                neighbors = rms_image_filtered(i_min:i_max, j_min:j_max);
                valid_neighbors = neighbors(~isnan(neighbors));
                
                if ~isempty(valid_neighbors)
                    % 用邻近有效值的中值替代
                    rms_image_repaired(i, j) = median(valid_neighbors);
                else
                    % 如果没有有效邻居，使用全局中值
                    rms_image_repaired(i, j) = rms_mean;
                end
            end
        end
    end
    
    fprintf('  异常值修复完成\n');
else
    rms_image_repaired = rms_image_filtered;
end

% 使用修复后的图像进行后续处理
rms_image = rms_image_repaired;

fprintf('处理后的RMS统计:\n');
fprintf('  最小值: %.6e\n', min(rms_image(:)));
fprintf('  最大值: %.6e\n', max(rms_image(:)));
fprintf('  平均值: %.6e\n', mean(rms_image(:)));

%% 4倍线性插值提高显示分辨率
fprintf('\n正在进行4倍插值...\n');

% 创建原始网格
[X_orig, Y_orig] = meshgrid(data_x, data_y);

% 创建插值网格 (31×31 -> 121×121)
interp_factor = 4;
x_interp = linspace(data_x(1), data_x(end), n_points * interp_factor - (interp_factor - 1));
y_interp = linspace(data_y(1), data_y(end), n_points * interp_factor - (interp_factor - 1));
[X_interp, Y_interp] = meshgrid(x_interp, y_interp);

% 进行二维线性插值
rms_image_interp = interp2(X_orig, Y_orig, rms_image, X_interp, Y_interp, 'linear');

fprintf('插值完成\n');
fprintf('  原始分辨率: %d × %d\n', n_points, n_points);
fprintf('  插值后分辨率: %d × %d\n', length(x_interp), length(y_interp));

%% 可视化RMS成像结果
figure('Name', 'RMS成像 (4倍插值)', 'Position', [100, 100, 800, 700]);

imagesc(x_interp * 1e3, y_interp * 1e3, rms_image_interp);
axis equal tight;
colormap('jet');
colorbar;
xlabel('X 位置 (mm)', 'FontSize', 12);
ylabel('Y 位置 (mm)', 'FontSize', 12);
title('RMS成像 (4倍插值)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'normal');

% 输出统计信息
fprintf('\nRMS统计信息:\n');
fprintf('  最小值: %.6e\n', min(rms_image(:)));
fprintf('  最大值: %.6e\n', max(rms_image(:)));
fprintf('  平均值: %.6e\n', mean(rms_image(:)));
fprintf('  中位数: %.6e\n', median(rms_image(:)));
fprintf('  标准差: %.6e\n', std(rms_image(:)));

fprintf('\n成像完成！\n');
