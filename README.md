# Pose Detection — 端头门司机行为分析

基于 YOLO 姿态估计的列车司机标准动作实时识别与合规判断系统。

## 项目结构

```
pose_detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                          # 核心源码包
│   ├── __init__.py
│   ├── config.py                 # 全局配置常量
│   ├── geometry.py               # 几何计算（角度、线段相交）
│   ├── detection.py              # 动作检测算法（3种策略）
│   ├── state_machine.py          # 动作状态机
│   ├── visualization.py          # 可视化绘制（骨架、中文、面板）
│   ├── annotation.py             # 标注工具（区域框选、参考线）
│   └── player.py                 # 交互式视频播放器
├── scripts/                      # 运行入口
│   ├── run_shangtichang2.py      # 上体场2视频（平行线+穿区域）
│   └── run_baoshan1.py           # 宝山1视频（角度指向）
├── data/                         # 标注数据
│   ├── regions_shangtichang2.json
│   └── regions_baoshan1.json
├── models/                       # 模型文件（需自行下载）
│   └── .gitkeep
├── output/                       # 输出视频
│   └── .gitkeep
└── docs/
    └── PROJECT_STATUS.md          # 项目进度文档
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA (推荐，用于 GPU 推理)

### 安装

```bash
pip install -r requirements.txt
```

### 下载模型

将 `yolo26x-pose.pt` 放入 `models/` 目录。

### 运行

```bash
# 上体场2视频（主要版本）
python scripts/run_shangtichang2.py

# 宝山1视频
python scripts/run_baoshan1.py
```

## 操作说明

| 按键 | 功能 |
|------|------|
| `空格` | 暂停 / 继续 |
| `Q` | 退出 |
| 拖拽进度条 | 跳转到指定帧 |
| **暂停时可用** | |
| `R` | 鼠标框选矩形区域 |
| `L` | 鼠标点击两点画参考线 |
| `D` | 删除最后一个区域 |
| `S` | 保存标注到 JSON 文件 |
| **随时可用** | |
| `Z` | 重置动作状态机 |

## 检测策略

| 类型 | 说明 | 使用场景 |
|------|------|---------|
| `parallel_line` | 手臂方向与参考线平行 | 手指呼唤、手动关门 |
| `pass_region` | 手臂穿过目标区域 | 确认指示灯 |
| `pointing` | 手臂角度朝向区域 | 备选方案 |
| `pointing_with_line` | 平行于线 + 朝向区域 | 双重判断 |

## 动作规范

基于申通技术中心文档，覆盖 5 类标准动作：

1. 开门后手指呼唤
2. 手动关门
3. 关门后确认夹缝
4. 开车前确认站台指示灯
5. 开车前确认道岔（可选）
