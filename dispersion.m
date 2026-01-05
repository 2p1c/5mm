%% 数据预处理：加载并重塑蛇形扫描数据
% 加载原始数据
load('data\5mm\500k\Scan_time.mat'); % 包含变量 x (1×2500) 和 y (961×2500)

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

%% 取中间一行做一个二维傅里叶变换，得到f-k
data_xt = permute(data_xyt(:,16,:),[1,3,2]);
% fs = 1/(data_time(2)-data_time(1))/1e6;%,MHz
% fs = 6.25;% PZT激励 b3数据，24.06.13多普勒行扫+面扫
nx=2^(nextpow2(length(data_x))+1); 
nt=2^(nextpow2(length(data_time))+1);
kf=fftn(data_xt,[nx,nt]);
% figure(301);
% surf(abs(kf));shading interp;colorbar;view([0,90]);xlabel('f');ylabel('kx');title('kf');
kf1=fftshift(kf,1);
% figure(302);
% surf(abs(kf1));shading interp;colorbar;view([0,90]);xlabel('f');ylabel('kx');title('kf1');

[l1,m1]=size(kf1);

f=(0:nt-1)*fs/nt; %实际频率
[minf,index]= min(abs(f(1:nt)-6e6));%最后的数字是画图duoshaoMHz
data_kf=kf1(:,1:index);
kx=((-round(l1/2)+1:round(l1/2))/l1)*2*pi/(data_x(1)-data_x(2));  %波数kx
kx_pict=kx;
% data_kf = abs(data_kf.');   
data_kf(:,nt/2+1:end)=0;%画图用的
f_pict=f(1:index);%画图用的   

figure(3);
surf(f_pict,-kx_pict,abs(data_kf));colorbar;view([0,90]);
shading interp;
% ylabel('kx(rad/mm)');xlabel('f(Hz)');
% set(gca,'XTick',[0: 1e5:8e5],'FontSize',12,'FontName','Times New Roman');
% set(gca,'YTick',[-3:1:3],'FontSize',12,'FontName','Times New Roman');
xlabel('\fontname{宋体}\fontsize{20}频率\fontname{Times New Roman}\fontsize{20} / kHz');
ylabel('\fontname{宋体}\fontsize{20}波数\fontname{Times New Roman}\fontsize{20} /  rad·mm^{-1}');
% ylim([-4,4]);xlim([0 2e6]);