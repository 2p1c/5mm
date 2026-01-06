%% 波场可视化程序
% 显示超声信号在空间上的传播过程

clear; clc; close all;

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
    % if mod(col, 2) == 0
    %     col_data = flipud(col_data); % 蛇形扫描：翻转偶数列
    % end
    
    % 存储到三维数组中（列数据存储为y方向）
    data_xyt(:, col, :) = col_data;
end

fprintf('数据加载完成:\n');
fprintf('  点阵大小: %d × %d\n', n_points, n_points);
fprintf('  时间点数: %d\n', length(data_time));
fprintf('  采样率: %.2f MHz\n', fs/1e6);
fprintf('  数据形状: %s\n', mat2str(size(data_xyt)));

%% 波场动画参数设置
frame_delay = 0.001;  % 每帧显示时间 (秒)，可调整播放速度
frame_skip = 2;      % 跳帧显示，1表示显示所有帧，2表示每2帧显示一次

% 可选：只显示部分帧（节省时间）
% start_frame = 1;
% end_frame = 500;
start_frame = 1;
end_frame = length(data_time);

fprintf('\n波场动画设置:\n');
fprintf('  总帧数: %d\n', length(data_time));
fprintf('  显示帧数: %d (跳帧=%d)\n', ceil((end_frame-start_frame+1)/frame_skip), frame_skip);
fprintf('  每帧延迟: %.3f 秒\n', frame_delay);
fprintf('  预计播放时长: %.2f 秒\n', ceil((end_frame-start_frame+1)/frame_skip)*frame_delay);

%% 创建动画窗口
fig = figure('Name', '波场传播动画', 'Position', [100, 100, 900, 700]);

% 定义中心区域（取中间50%的区域）
center_range = round(n_points * 0.25) : round(n_points * 0.75);
[center_x, center_y] = meshgrid(center_range, center_range);
center_indices = [center_x(:), center_y(:)];

% 从中心区域随机选择10个点
num_sample_points = 10;
rng('shuffle');  % 使用当前时间作为随机种子
sample_idx = randperm(size(center_indices, 1), min(num_sample_points, size(center_indices, 1)));
sample_points = center_indices(sample_idx, :);

% 提取这10个点的所有时间序列数据
max_values = zeros(num_sample_points, 1);
min_values = zeros(num_sample_points, 1);

for i = 1:num_sample_points
    point_data = squeeze(data_xyt(sample_points(i,1), sample_points(i,2), :));
    max_values(i) = max(point_data);
    min_values(i) = min(point_data);
end

% 计算平均值作为颜色范围
color_min = mean(min_values);
color_max = mean(max_values);

% 使用对称范围
color_range = max(abs(color_min), abs(color_max));
clim_range = [-color_range, color_range];

% 输出颜色范围信息
fprintf('\n颜色范围计算（基于中心区域%d个随机点）:\n', num_sample_points);
fprintf('  采样点位置: \n');
for i = 1:num_sample_points
    fprintf('    点%d: (%d, %d) -> (%.1f mm, %.1f mm)\n', i, ...
            sample_points(i,1), sample_points(i,2), ...
            data_x(sample_points(i,1))*1e3, data_y(sample_points(i,2))*1e3);
end
fprintf('  最大值平均: %.6e\n', mean(max_values));
fprintf('  最小值平均: %.6e\n', mean(min_values));
fprintf('  使用的颜色范围: [%.6e, %.6e]\n', clim_range(1), clim_range(2));

fprintf('\n开始播放动画...\n');
fprintf('按 Ctrl+C 可以中止播放\n\n');

%% 播放动画
for frame_idx = start_frame:frame_skip:end_frame
    % 检查窗口是否已关闭
    if ~ishandle(fig)
        fprintf('\n动画窗口已关闭，停止播放。\n');
        break;
    end
    
    % 提取当前时刻的波场数据 (31×31)
    wavefield = squeeze(data_xyt(:, :, frame_idx));
    
    % 4倍线性插值，提高显示分辨率 (31×31 -> 121×121)
    [X_orig, Y_orig] = meshgrid(data_x, data_y);
    interp_factor = 4;
    x_interp = linspace(data_x(1), data_x(end), n_points * interp_factor - (interp_factor - 1));
    y_interp = linspace(data_y(1), data_y(end), n_points * interp_factor - (interp_factor - 1));
    [X_interp, Y_interp] = meshgrid(x_interp, y_interp);
    wavefield_interp = interp2(X_orig, Y_orig, wavefield, X_interp, Y_interp, 'linear');
    
    % 显示插值后的波场
    imagesc(x_interp * 1e3, y_interp * 1e3, wavefield_interp);
    axis equal tight;
    colormap('jet');
    colorbar;
    caxis(clim_range);  % 使用固定的颜色范围
    
    % 添加标签和标题
    xlabel('X 位置 (mm)', 'FontSize', 12);
    ylabel('Y 位置 (mm)', 'FontSize', 12);
    title(sprintf('波场传播 | 时间: %.2f μs | 帧: %d/%d', ...
                  data_time(frame_idx)*1e6, frame_idx, length(data_time)), ...
          'FontSize', 14, 'FontWeight', 'bold');
    
    % 添加网格
    grid on;
    set(gca, 'YDir', 'normal');  % 确保Y轴方向正确
    
    % 更新显示
    drawnow;
    
    % 延迟控制帧率
    pause(frame_delay);
end

fprintf('动画播放完成！\n');

%% 可选：保存动画为视频文件
% 取消下面注释可以保存视频
% video_writer = VideoWriter('wavefield_animation.avi');
% video_writer.FrameRate = 1/frame_delay;
% open(video_writer);
% 
% for frame_idx = start_frame:frame_skip:end_frame
%     wavefield = squeeze(data_xyt(:, :, frame_idx));
%     imagesc(data_x * 1e3, data_y * 1e3, wavefield');
%     axis equal tight;
%     colormap('jet');
%     colorbar;
%     caxis(clim_range);
%     xlabel('X 位置 (mm)', 'FontSize', 12);
%     ylabel('Y 位置 (mm)', 'FontSize', 12);
%     title(sprintf('波场传播 | 时间: %.2f μs', data_time(frame_idx)*1e6), ...
%           'FontSize', 14);
%     grid on;
%     set(gca, 'YDir', 'normal');
%     
%     frame = getframe(gcf);
%     writeVideo(video_writer, frame);
% end
% 
% close(video_writer);
% fprintf('视频已保存为 wavefield_animation.avi\n');
