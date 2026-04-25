# PINN-DIC

一个面向教学的 **PINN-DIC（Physics-Informed Neural Network Digital Image Correlation）** 简化示例项目。

本仓库用于《**AI+Mech**》教材配套实验，目标是帮助读者快速理解并跑通 PINN-DIC 的核心流程：

- 读取参考图像 / 变形图像 / ROI；
- 构建坐标网格并提取 ROI 点；
- 用全连接神经网络回归位移场 \((u,v)\)；
- 通过可微插值将参考图像映射到变形图像并构造灰度误差损失；
- 训练后输出位移场与可视化结果。

> 说明：这是一个“可读、可改、可运行”的最小教学版本，重点在于 PINN-DIC 主流程与实验可复现性，而非完整工程化或极致性能。

---

## 1. 代码结构

```text
PINN-DIC/
├── main.py                    # 主程序：数据准备、训练、预测、结果可视化
├── train.py                   # 两阶段训练包装（Adam / LBFGS）
├── predict.py                 # 位移场预测接口
├── plot.py                    # 云图与等值线绘图
├── src/
│   ├── FCNN.py                # 全连接神经网络定义
│   └── PINN.py                # PINN-DIC 核心：损失、插值、优化、预测
├── utils/
│   ├── utils.py               # 图像加载、ROI处理、网格构建、矩阵工具函数
│   └── select_roi.py          # ROI 选择辅助脚本
└── speckle_image/
    └── real/
        ├── circle/            # 圆环示例数据（默认 main.py 使用）
        └── strip/             # 条带示例数据
```

---

## 2. 环境与依赖

建议 Python 3.9+，核心依赖：

- `torch`
- `numpy`
- `scipy`
- `Pillow`
- `matplotlib`

可参考如下命令安装（按需调整 CUDA 版本）：

```bash
pip install torch numpy scipy pillow matplotlib
```

---

## 3. 运行示例

### 3.1 直接运行默认示例

`main.py` 默认使用 `speckle_image/real/circle/` 下的参考图、变形图和 ROI：

```bash
python main.py
```

程序流程：

1. 读取图像并构建 \([-1,1]\times[-1,1]\) 坐标网格；
2. 初始化 `PhysicsInformedNN`；
3. 执行预训练 + 正式训练；
4. 预测全场位移；
5. 绘制位移云图与等值线图。

### 3.2 切换数据示例

如需切换到其他数据（例如 `strip`），可在 `main.py` 中修改如下路径：

- `ref_image_path`
- `def_image_path`
- `roi_path`

然后再次运行：

```bash
python main.py
```

---

## 4. 这是一个《AI+Mech》教材示例

本项目定位为《**AI+Mech**》教材中的简明示例，强调以下教学目标：

- 让读者看懂 PINN-DIC 的“输入—网络—物理/成像约束—优化—输出”链路；
- 给出可运行的最小代码骨架，便于课程实验与二次开发；
- 在不过度封装的前提下保留核心可微优化机制。

---

## 5. 拔高拓展建议

在本示例基础上，可开展以下“拔高”方向：

1. **B 样条任意阶插值函数**  
   通过 B 样条曲线/曲面构造任意阶插值核，实现更灵活、更平滑的灰度重采样模型。

2. **自编写快速插值程序（替代通用插值算子）**  
   在 PINN-DIC 反向传播中，插值算子的效率与梯度质量非常关键。实践中，直接使用 PyTorch 的通用插值/采样接口在某些场景下会带来位移场分辨率压缩效应，进而影响优化梯度的有效性。可尝试：
   - 针对 DIC 位移场形态设计专用插值核；
   - 优化前向与反向计算路径，减少不必要的分辨率损失；
   - 在保持可微性的同时提升收敛稳定性与速度。

---

## 6. 参考文献

- Li, B., Zhou, S., Ma, Q. et al. **Physics-Informed Neural Network Based Digital Image Correlation Method**. *Experimental Mechanics* **65**, 221–240 (2025).  
  https://doi.org/10.1007/s11340-024-01139-w

---

## 7. 致读者

如果你正在将本示例用于课程作业或科研起步，建议先从以下顺序入手：

1. 先跑通 `main.py`；
2. 阅读 `src/PINN.py` 的 `loss_fn1/loss_fn2/interp_IPD`；
3. 再尝试替换网络结构、损失形式或插值实现；
4. 最后做多组数据对比与误差分析。

欢迎在此基础上扩展为更完整的 PINN-DIC 研究框架。
