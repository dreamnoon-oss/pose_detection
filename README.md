# Pose Detection — 端头门司机行为分析

基于 YOLO 姿态估计的列车司机标准动作实时识别与合规判断系统。

## 架构

```
run_xxx.py → VideoPlayer → ParallelDetector → detection.py → geometry.py
                ↕                ↕
          annotation.py    SequenceAnalyzer (事后分析)
          visualization.py
```

**核心思路**：所有动作规则**并行独立检测**，各自记录触发时间戳。视频播放结束后，按事件发生顺序映射到预期动作序列，判定合规性。

## 项目结构

```
pose_detection/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── config.py          # 全局配置（关键点、骨架、阈值）
│   ├── geometry.py        # 几何计算（角度、线段相交）
│   ├── detection.py       # 4 种检测算法（平行线/穿区域/指向/组合）
│   ├── detector.py        # 并行检测器（多规则同时运行）
│   ├── analyzer.py        # 时序分析器（事件→动作映射+顺序判定）
│   ├── state_machine.py   # [已弃用] 旧版串行状态机
│   ├── visualization.py   # 可视化（骨架、面板、中文渲染、调试射线）
│   ├── annotation.py      # 标注工具（区域框选、参考线）
│   └── player.py          # 交互式视频播放器
├── scripts/
│   ├── run_shangtichang.py
│   └── run_baoshan.py
├── data/                   # 标注数据 (JSON)
├── models/                 # 模型文件
└── output/                 # 输出视频
```

## 检测策略

| 类型 | 函数 | 说明 |
|------|------|------|
| `parallel_line` | `check_arm_parallel_to_line()` | 肩→腕向量与参考线夹角 < 阈值。可选肘部回退、躯干夹角下限 |
| `pass_region` | `check_arm_passes_region()` | 肩→腕射线（可延长）穿过/落在矩形区域内 |
| `pointing` | `check_pointing()` | 手臂方向与区域角点夹角 < 阈值 |
| `pointing_with_line` | `check_pointing_with_line()` | 手臂平行于线 且 朝向区域 |

### 关键参数

| 参数 | 值 | 说明 |
|------|----|------|
| 关键点 | 5-12 | 肩/肘/腕/髋 |
| 平行角度阈值 | 40° | 手臂与参考线最大夹角 |
| 躯干夹角下限 | 45° | 手臂 vs 肩→髋夹角需 > 45°（防止未抬臂误触发） |
| 延长倍数 | 6× | pass_region 时腕部沿手臂方向延长倍数 |
| 持续帧数 | 15 | 连续命中帧数确认事件 |
| 帧衰减 | -2/帧 | 容忍短暂丢帧 |
| 冷却期 | 45 帧 | 事件触发后同规则暂停检测 |
| 最小手臂长度 | 30px | 过滤无效检测 |

## 动作序列（上体场）

| 动作 | 规则 | 检测类型 | 目标 |
|------|------|------|------|
| 动作1 | rule_A (第1次) | parallel_line | line_1 |
| 动作2 | rule_B (第1次) | parallel_line | line_2（肘部回退+躯干夹角） |
| 动作3 | rule_A (第2次) | parallel_line | line_1 |
| 动作4 | rule_C (第1次) | pass_region | region_1（延长射线） |

## 实时指标面板

播放/暂停时左上角显示并行检测面板，下方额外显示每个动作的实时指标：

- **parallel_line 规则**：显示肩→腕（或肩→肘）与参考线的当前夹角
- **pointing 规则**：显示手臂方向与区域的最小夹角
- **pass_region 规则**：显示"穿过"或"未穿过"

暂停后面板自动切换到右上角，避免与"PAUSED"文字重叠。每条规则独立计算，不受检测阈值限制，始终可见。

### 可视化增强

- **手臂线段**：肩→肘→腕以加粗青色（左臂）/ 洋红色（右臂）绘制，置信度阈值降至 0.3
- **延长射线**：绿色 = 命中区域，红色 = 未命中
- 暂停和拖拽进度条时完整渲染所有面板

## 快速开始

### 环境要求

- Python 3.10+
- CUDA（推荐）

### 安装

```bash
pip install -r requirements.txt
```

将 `yolo26x-pose.pt` 放入 `models/` 目录。

### 运行

```bash
python scripts/run_shangtichang.py   # 上体场（主版本）
python scripts/run_baoshan.py        # 宝山（角度法）
```

## 操作说明

| 按键 | 功能 |
|------|------|
| `空格` | 暂停 / 继续 |
| `Q` | 退出 |
| `Z` | 重置检测器 |
| 拖拽进度条 | 跳转 |
| **暂停时** | |
| `R` | 框选矩形区域 |
| `L` | 鼠标画参考线 |
| `D` | 删除最后区域 |
| `S` | 保存标注到 JSON |
