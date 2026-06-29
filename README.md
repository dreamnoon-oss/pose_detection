# Pose Detection — 端头门司机行为分析

基于 YOLO 姿态估计的列车司机标准动作实时识别与合规判断系统。

## 项目结构

```
pose_detection/
├── README.md
├── pyproject.toml                # 项目配置
├── requirements.txt
├── .gitignore
├── src/                          # Python 包
│   ├── __init__.py
│   ├── config.py                 # 全局配置常量
│   ├── geometry.py               # 几何计算（角度、线段相交）
│   ├── detection.py              # 动作检测算法（3种策略）
│   ├── state_machine.py          # 动作状态机
│   ├── visualization.py          # 可视化绘制（骨架、中文、面板）
│   ├── annotation.py             # 标注工具（区域框选、参考线）
│   └── player.py                 # 交互式视频播放器
├── scripts/                      # 运行入口
│   ├── run_shangtichang2.py      # 上体场2视频（主要版本）
│   └── run_baoshan1.py           # 宝山1视频
├── data/                         # 标注数据
├── models/                       # 模型文件（需自行下载）
└── output/                       # 输出视频
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA（推荐，用于 GPU 推理）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载模型

将 `yolo26x-pose.pt` 放入 `models/` 目录。

### PyCharm 设置

首次打开项目后，右键 `pose_detection` 根目录 → **Mark Directory as** → **Sources Root**，消除 `src` 包的红色波浪线。

### 运行

```bash
# 上体场2视频（主要版本，平行线 + 穿区域检测）
python scripts/run_shangtichang2.py

# 宝山1视频（角度法检测）
python scripts/run_baoshan1.py
```

## 操作说明

| 按键 | 功能 |
|------|------|
| `空格` | 暂停 / 继续 |
| `Q` | 退出 / 关闭窗口 |
| 拖拽进度条 | 跳转到指定帧 |
| **暂停时可用** | |
| `R` | 鼠标框选矩形区域 |
| `L` | 鼠标点击两点画参考线 |
| `D` | 删除最后一个区域 |
| `S` | 保存标注到 JSON 文件 |
| **随时可用** | |
| `Z` | 重置动作状态机 |

## 检测策略

| 类型 | 函数 | 说明 |
|------|------|------|
| `parallel_line` | `check_arm_parallel_to_line()` | 肩→腕向量与参考线夹角 < 40°，可回退用肘部 |
| `pass_region` | `check_arm_passes_region()` | 肩→腕线段穿过/落在矩形区域内 |
| `pointing` | `check_pointing()` | 手臂方向与区域最小夹角 < 30° |
| `pointing_with_line` | `check_pointing_with_line()` | 手臂平行于线 且 朝向区域 |

### 参数

| 参数 | 值 | 说明 |
|------|----|------|
| 关键点 | 5-12 | 肩/肘/腕/髋 |
| 平行角度阈值 | 40° | 手臂与参考线最大夹角 |
| 持续帧数 | 15 | 连续命中帧数确认动作 |
| 帧衰减 | -2/帧 | 容忍短暂丢帧 |
| 最小手臂长度 | 30px | 过滤无效检测 |
| 置信度阈值 | 0.5 | 关键点置信度 |

## 动作序列（上体场2）

| 动作 | 类型 | 目标 | 备注 |
|------|------|------|------|
| 动作1 | `parallel_line` | line_1 | 手指呼唤 |
| 动作2 | `parallel_line` | line_2 | 手动关门，允许肘部回退 |
| 动作3 | `parallel_line` | line_1 | 确认夹缝 |
| 动作4 | `pass_region` | region_1 | 确认站台指示灯 |

## 动作规范

基于申通技术中心文档：

1. 开门后手指呼唤 — 手臂举起与肩平行，维持 2 秒以上
2. 手动关门 — 手臂抬起约 60°，维持 2 秒以上
3. 关门后确认夹缝 — 手臂举起与肩平行指向夹缝，维持 2 秒以上
4. 开车前确认站台指示灯 — 手臂举起 ≥ 肩部指向指示灯，维持 1 秒以上
5. 开车前确认道岔（可选）— 手臂举起与肩平行指向道岔，维持 1 秒以上
