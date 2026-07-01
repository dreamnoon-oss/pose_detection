# 检测逻辑文档

## 版本概述

| 版本 | 架构 | 文件 | 状态 |
|------|------|------|------|
| v1 | 串行状态机 | `src/state_machine.py` | 已弃用，保留供参考 |
| v2 | 并行检测 + 事后分析 | `src/detector.py` + `src/analyzer.py` | 当前使用 |

---

## v2（当前版本）：并行检测 + 事后分析

### 架构

```
所有规则同时独立运行 → 各自记录时间戳事件 → 视频结束后按发生顺序映射到动作 → 判定合规
```

### 检测规则 vs 动作映射

**核心概念**：检测规则（唯一检测条件）与动作（期望行为）分离。同一条规则可能触发多次，按发生顺序映射到不同动作。

### 上体场2 — 检测规则

```python
DETECTION_RULES = [
    {"name": "rule_A", "type": "parallel_line", "ref_line": "line_1"},
    {"name": "rule_B", "type": "parallel_line", "ref_line": "line_2", "allow_elbow": True},
    {"name": "rule_C", "type": "pass_region", "target_region": "region_1"},
]
```

### 上体场2 — 动作映射

```python
ACTION_MAPPING = [
    {"action": "动作1", "rule": "rule_A", "occurrence": 1},  # 手指呼唤
    {"action": "动作2", "rule": "rule_B", "occurrence": 1},  # 手动关门
    {"action": "动作3", "rule": "rule_A", "occurrence": 2},  # 确认夹缝（同规则，第2次出现）
    {"action": "动作4", "rule": "rule_C", "occurrence": 1},  # 确认站台指示灯
]
```

`occurrence` 字段表示"该规则第 N 次触发时映射到此动作"。

### 并行检测器 (ParallelDetector)

每个规则独立维护：

```
每帧:
  1. 检查冷却期（事件触发后 45 帧内跳过）
  2. 调用对应检测函数
  3. 命中: hold_counter += 1
     未命中: hold_counter = max(0, hold_counter - 2)
  4. hold_counter ≥ 15 → 记录事件(规则名, 帧号, 手臂侧, 角度, 坐标)
                       → hold_counter 归零, 进入 45 帧冷却
```

### 时序分析 (SequenceAnalyzer)

视频结束后执行：

```
1. 按规则名分组事件
2. 对每个动作映射: 取对应规则的第 N 次事件 → 记录帧号
3. 检查所有动作帧号是否严格递增 → 判断顺序正确/异常
4. 输出报告
```

---

## v1（旧版）：串行状态机

### 架构

```
动作1 完成 → 动作2 开始 → 动作3 开始 → 动作4 开始 → 全部完成
```

严格按顺序检测，前一个动作未完成则不会检测后面的动作。

### 状态机 (ActionStateMachine)

```python
# 初始化
current_idx = 0          # 指向当前期望的动作
hold_counter = 0         # 当前动作的连续命中计数

# 每帧
def update(keypoints_obj):
    target = action_sequence[current_idx]          # 只检测当前动作
    hit = run_detection(target, keypoints_obj)      # 调用对应检测函数
    if hit:
        hold_counter += 1
        if hold_counter >= 15:
            current_idx += 1  # 前进到下一个动作
            hold_counter = 0
    else:
        hold_counter = max(0, hold_counter - 2)
```

### 旧版动作序列（上体场2）

```python
ACTION_SEQUENCE = [
    {"name": "动作1", "ref_line": "line_1", "type": "parallel_line"},
    {"name": "动作2", "ref_line": "line_2", "type": "parallel_line", "allow_elbow": True},
    {"name": "动作3", "ref_line": "line_1", "type": "parallel_line"},
    {"name": "动作4", "target_region": "region_1", "type": "pass_region"},
]
```

### 两版关键差异

| | v1 串行状态机 | v2 并行检测 |
|------|------|------|
| 动作1和动作3冲突 | 不会（顺序执行） | 靠 occurrence 区分（第1次 vs 第2次触发） |
| 动作顺序容错 | 无（必须严格按序） | 有（检测完再判定，可报告"顺序异常"） |
| 可选动作 | 需硬编码跳过 | 直接配置即可 |
| 重复动作 | 不支持 | 可检测到多次并记录 |
| 冷却机制 | 不需要（自动推进） | 需要（防止同一手势分裂为多个事件） |

---

## 检测算法详解（v1 和 v2 共用）

### 1. parallel_line — 手臂平行于参考线

**文件**: `src/detection.py` → `check_arm_parallel_to_line()`

**逻辑**:
```
对每一帧:
  遍历所有检测到的人 × 左右臂:
    1. 肩部置信度 > 0.5（必须）
    2. 确定远端关键点:
       - 优先用腕部（置信度 > 0.5）
       - allow_elbow=True 时回退用肘部
       - 两者都不够 → 跳过
    3. 肩→远端 向量长度 > 30px
    4. 计算 肩→远端 与 参考线 的夹角
    5. 夹角 < 40° → 可选检查: 肩→腕 与 肩→髋 夹角 > 45°（v2 新增）
    6. 全部满足 → 返回命中
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `angle_threshold` | 40° | 手臂与参考线最大夹角 |
| `min_arm_len` | 30px | 最小臂长 |
| `allow_elbow` | False | 是否允许肘部回退 |
| `min_arm_torso_angle` | 0 (不检查) | 手臂与躯干最小夹角 (v2 新增) |

**适用动作**:
- 动作1 (手指呼唤): `parallel_line + line_1`，手臂水平举起与肩平行
- 动作2 (手动关门): `parallel_line + line_2`，手臂抬起约60°，允许肘部回退
- 动作3 (确认夹缝): `parallel_line + line_1`，同动作1但发生在动作2之后

---

### 2. pass_region — 手臂穿过/指向矩形区域

**文件**: `src/detection.py` → `check_arm_passes_region()`

**逻辑**:
```
对每一帧:
  遍历所有检测到的人 × 左右臂:
    1. 肩部和腕部置信度 > 0.5（两者都必须，无肘部回退）
    2. 肩→腕 向量长度 > 30px
    3. 构建检测射线:
       - v1: 肩 → 腕（有限线段）
       - v2: 肩 → 腕延长6倍臂长（射线）
    4. 三项检查（命中任一即通过）:
       a. 肩部端点落在矩形内
       b. 远端端点落在矩形内
       c. 射线与矩形任一边相交（叉积法，含边界）
    5. 命中 → 返回 True
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_arm_len` | 30px | 最小臂长 |
| `extend_ray` | True (v2) | 是否延长手臂方向形成射线 |

**适用动作**:
- 动作4 (确认站台指示灯): `pass_region + region_1`

**v1 vs v2 差异**: v1 只检测肩→腕的有限线段，必须手真正到达区域；v2 默认延长6倍形成射线，手臂**指向**区域即可命中。

---

### 3. pointing — 手臂指向区域（角度法）

**文件**: `src/detection.py` → `check_pointing()`

**逻辑**:
```
对每一帧:
  遍历所有检测到的人 × 左右臂:
    1. 肩部和腕部置信度 > 0.5
    2. 肩→腕 向量长度 > 30px
    3. 计算手臂方向与矩形5个点（4角+中心）的最小夹角
    4. 夹角 < 30° → 返回命中
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `angle_threshold` | 30° | 手臂方向与区域的最小夹角 |
| `min_arm_len` | 30px | 最小臂长 |

**适用场景**: 宝山1视频（角度法检测）

---

### 4. pointing_with_line — 组合检测：平行于线且指向区域

**文件**: `src/detection.py` → `check_pointing_with_line()`

**逻辑**:
```
对每一帧:
  遍历所有检测到的人 × 左右臂:
    1. 肩部和腕部置信度 > 0.5
    2. 肩→腕 向量长度 > 30px
    3. 检查手臂与参考线夹角 < 40°（平行条件）
    4. 检查手臂方向与矩形夹角 < 55°（指向条件，比纯 pointing 宽松）
    5. 两个条件同时满足 → 返回命中
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `line_angle_threshold` | 40° | 手臂与参考线最大夹角 |
| `loose_angle_threshold` | 55° | 手臂方向与区域最大夹角（宽松） |
| `min_arm_len` | 30px | 最小臂长 |

**适用场景**: 宝山1视频（组合检测，比纯 pointing 更严格的方向约束）

---

## 几何计算 (geometry.py)

所有检测函数共用：

| 函数 | 算法 | 用途 |
|------|------|------|
| `angle_between(v1, v2)` | arccos(dot/(n1*n2)) | 两向量夹角 |
| `min_angle_to_rect(wrist, arm_dir, rect)` | 手臂方向与矩形4角+中心的最小夹角 | pointing 检测 |
| `segments_intersect(p1,p2,p3,p4)` | 叉积法判断两线段相交 | pass_region 检测 |

---

## 关键点索引 (COCO 17-keypoint)

| 索引 | 名称 | 用途 |
|------|------|------|
| 5 | 左肩 | 手臂起点 |
| 6 | 右肩 | 手臂起点 |
| 7 | 左肘 | 肘部回退（仅 parallel_line） |
| 8 | 右肘 | 肘部回退（仅 parallel_line） |
| 9 | 左腕 | 手臂远端 |
| 10 | 右腕 | 手臂远端 |
| 11 | 左髋 | 躯干参考（仅 min_arm_torso_angle） |
| 12 | 右髋 | 躯干参考（仅 min_arm_torso_angle） |

同侧映射:
- 左臂: 肩(5) → 肘(7) → 腕(9), 髋(11)
- 右臂: 肩(6) → 肘(8) → 腕(10), 髋(12)

---

## 通用参数汇总

| 参数 | 值 | 适用检测类型 | 说明 |
|------|-----|-------------|------|
| 关键点置信度阈值 | 0.5 | 全部 | 关键点低于此值视为无效 |
| 平行角度阈值 | 40° | parallel_line, pointing_with_line | 臂与参考线夹角上限 |
| 躯干夹角下限 | 45° | parallel_line (v2) | 臂与躯干夹角下限，防未抬臂 |
| 指向角度阈值 | 30° | pointing | 臂方向与区域夹角上限 |
| 宽松指向阈值 | 55° | pointing_with_line | 组合检测的指向条件 |
| 延长倍数 | 6× | pass_region (v2) | 射线延长倍率 |
| 最小臂长 | 30px | 全部 | 过滤无效短臂 |
| 持续帧数 | 15 | 全部 | 连续命中确认事件 |
| 帧衰减 | -2/帧 | 全部 | 容忍短暂丢帧 |
| 冷却期 | 45帧 | 全部 (v2) | 事件触发后同规则暂停 |
| 检测框置信度 | 0.5 | YOLO | 人物检测阈值 |
