import React, { useState } from 'react';
import { ArrowUpRight, Cpu, Eye, Box, Info } from 'lucide-react';

const ProductMatrix = () => {
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);

  // 定义矩阵数据点
  const dataPoints = [
    {
      id: 1,
      name: "SoC 芯片",
      x: 20, // 百分比
      y: 30, // 百分比
      color: "#3B82F6", // Blue
      icon: <Cpu size={24} className="text-white" />,
      desc: "传统架构 / 依赖后端算力",
      value: "基础底座",
      detail: "技术前移度低，系统集成度低，需配合外围电路和算法开发。"
    },
    {
      id: 2,
      name: "智能视觉传感器",
      x: 55,
      y: 65,
      color: "#EF4444", // Red
      icon: <Eye size={24} className="text-white" />,
      desc: "前端感算一体 / 低延迟",
      value: "核心差异化",
      detail: "将计算能力前移至传感器端，大幅降低传输带宽，提升响应速度。"
    },
    {
      id: 3,
      name: "模组/系统集成",
      x: 85,
      y: 90,
      color: "#10B981", // Green
      icon: <Box size={24} className="text-white" />,
      desc: "即插即用 / 快速落地",
      value: "商业价值最大化",
      detail: "高度封装，解决客户最终问题，系统效率最高，交付门槛最低。"
    }
  ];

  return (
    <div className="w-full h-full min-h-[600px] bg-slate-50 p-8 flex flex-col items-center justify-center font-sans">
      
      {/* 标题区域 */}
      <div className="mb-8 text-center">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">智能视觉技术路线-产品价值矩阵</h2>
        <p className="text-slate-500 text-sm">投委会决策参考图：技术前移 vs 系统效率</p>
      </div>

      {/* 图表主体容器 */}
      <div className="relative w-full max-w-4xl aspect-[16/10] bg-white rounded-xl shadow-xl border border-slate-200 overflow-hidden">
        
        {/* 背景网格 (装饰) */}
        <div className="absolute inset-0 grid grid-cols-6 grid-rows-6 pointer-events-none">
          {[...Array(36)].map((_, i) => (
            <div key={i} className="border-r border-b border-slate-50 last:border-0" />
          ))}
        </div>

        {/* 核心 SVG 绘图区 */}
        <svg className="absolute inset-0 w-full h-full p-12" viewBox="0 0 800 500">
          <defs>
            {/* 定义趋势箭头的渐变色 */}
            <linearGradient id="trendGradient" x1="0%" y1="100%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#E2E8F0" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.1" />
            </linearGradient>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#64748B" />
            </marker>
          </defs>

          {/* 巨大的趋势背景箭头 (暗示价值跃迁) */}
          <path 
            d="M 100 450 Q 400 450 700 100" 
            fill="none" 
            stroke="url(#trendGradient)" 
            strokeWidth="80" 
            strokeLinecap="round"
          />
          <path 
            d="M 100 450 Q 400 450 700 100" 
            fill="none" 
            stroke="#CBD5E1" 
            strokeWidth="2" 
            strokeDasharray="8 8"
            className="opacity-50"
          />
          <text x="450" y="320" fill="#94A3B8" fontSize="14" transform="rotate(-25 450 320)" textAnchor="middle">
            价值跃迁趋势：从器件到方案
          </text>

          {/* 坐标轴绘制 */}
          {/* X 轴 */}
          <line x1="50" y1="450" x2="750" y2="450" stroke="#64748B" strokeWidth="3" markerEnd="url(#arrowhead)" />
          <text x="750" y="485" textAnchor="end" fill="#334155" fontWeight="bold" fontSize="14">
            技术前移程度 (Technology Shift) →
          </text>
          <text x="50" y="480" textAnchor="start" fill="#94A3B8" fontSize="12">传统集中计算</text>
          <text x="730" y="460" textAnchor="end" fill="#94A3B8" fontSize="12">前端感存算一体</text>

          {/* Y 轴 */}
          <line x1="50" y1="450" x2="50" y2="50" stroke="#64748B" strokeWidth="3" markerEnd="url(#arrowhead)" />
          <text x="60" y="50" textAnchor="start" fill="#334155" fontWeight="bold" fontSize="14">
            ↑ 系统效率提升 (System Efficiency)
          </text>
          
          {/* 数据点绘制 */}
          {dataPoints.map((point) => {
            // 将百分比坐标转换为 SVG 坐标
            // X轴范围: 50 -> 750 (宽度 700)
            // Y轴范围: 450 -> 50 (高度 400)
            const cx = 50 + (point.x / 100) * 700;
            const cy = 450 - (point.y / 100) * 400;
            const isHovered = hoveredPoint === point.id;

            return (
              <g 
                key={point.id} 
                className="cursor-pointer transition-all duration-300"
                onMouseEnter={() => setHoveredPoint(point.id)}
                onMouseLeave={() => setHoveredPoint(null)}
                style={{ transformOrigin: `${cx}px ${cy}px` }}
              >
                {/* 连接线 (Hover时显示) */}
                <line 
                  x1={cx} y1={cy} x2={cx} y2={450} 
                  stroke={point.color} 
                  strokeWidth="1" 
                  strokeDasharray="4 4"
                  opacity={isHovered ? 0.6 : 0}
                  className="transition-opacity duration-300"
                />
                <line 
                  x1={cx} y1={cy} x2={50} y2={cy} 
                  stroke={point.color} 
                  strokeWidth="1" 
                  strokeDasharray="4 4"
                  opacity={isHovered ? 0.6 : 0}
                  className="transition-opacity duration-300"
                />

                {/* 外部光晕 (Hover时扩大) */}
                <circle 
                  cx={cx} cy={cy} r={isHovered ? 40 : 0} 
                  fill={point.color} 
                  opacity="0.1"
                  className="transition-all duration-500"
                />

                {/* 核心圆点 */}
                <circle 
                  cx={cx} cy={cy} r={28} 
                  fill={point.color} 
                  className="shadow-lg drop-shadow-md"
                />
                
                {/* 内部 Icon (SVG foreignObject 无法很好兼容所有导出，这里用纯SVG圆替代或叠加) */}
                {/* 模拟 Icon 位置 */}
                <circle cx={cx} cy={cy} r="25" fill="none" stroke="white" strokeWidth="2" strokeOpacity="0.5" />
                
                {/* 标签文本 */}
                <text 
                  x={cx} y={cy + 45} 
                  textAnchor="middle" 
                  fill={isHovered ? point.color : "#334155"} 
                  fontWeight="bold"
                  fontSize="14"
                  className="transition-colors duration-300"
                >
                  {point.name}
                </text>
                 <text 
                  x={cx} y={cy + 62} 
                  textAnchor="middle" 
                  fill="#64748B" 
                  fontSize="10"
                >
                  {point.desc}
                </text>
              </g>
            );
          })}
        </svg>

        {/* 悬浮卡片 (HTML层) - 用于展示详细信息 */}
        {hoveredPoint !== null && (
           <div 
             className="absolute bg-white/95 backdrop-blur-sm p-4 rounded-lg shadow-xl border-l-4 border-slate-200 z-10 w-64 animate-in fade-in zoom-in duration-200"
             style={{
                // 简单的定位逻辑，实际项目可优化
                left: 50 + (dataPoints[hoveredPoint-1].x / 100) * 70 + '%',
                top: 80 - (dataPoints[hoveredPoint-1].y / 100) * 80 + '%',
                borderColor: dataPoints[hoveredPoint-1].color
             }}
           >
             <div className="flex items-center gap-2 mb-2">
               <div className="p-1 rounded-full" style={{backgroundColor: dataPoints[hoveredPoint-1].color}}>
                  {dataPoints[hoveredPoint-1].icon}
               </div>
               <span className="font-bold text-slate-800">{dataPoints[hoveredPoint-1].name}</span>
             </div>
             <p className="text-xs font-semibold text-slate-600 mb-1">
               核心价值: {dataPoints[hoveredPoint-1].value}
             </p>
             <p className="text-xs text-slate-500 leading-relaxed">
               {dataPoints[hoveredPoint-1].detail}
             </p>
           </div>
        )}

        {/* 右上角图例/说明 */}
        <div className="absolute top-4 right-4 bg-white/80 p-3 rounded-lg border border-slate-100 shadow-sm text-xs text-slate-500">
           <div className="flex items-center gap-2 mb-1">
             <ArrowUpRight size={14} className="text-blue-500" />
             <span>发展方向：更高集成度，更高效率</span>
           </div>
           <div className="flex items-center gap-2">
             <Info size={14} className="text-slate-400" />
             <span>鼠标悬停节点查看详情</span>
           </div>
        </div>

      </div>
    </div>
  );
};

export default ProductMatrix;