# 三核 CNN：无卷积偏置、电子全连接偏置、三组残差材料权重

这是独立的模块化实验代码，不依赖仓库里互相导入的历史实验脚本，不修改 `CURRENT_SCHEME.json` 或正式检查点。

## 选定结构

```text
14×14 灰度输入
→ 三个 5×5 卷积核，stride=1，padding=0，卷积无偏置
→ 3×10×10，电子 ReLU
→ 自适应平均池化至 3×3
→ 展平 27
→ 全连接 27→32 + 电子偏置 + ReLU
→ 全连接 32→16 + 电子偏置 + ReLU
→ 全连接 16→10 + 电子偏置
→ 10 个类别分数，argmax 分类
```

每个卷积和全连接权重用三组残差表示：`W≈Q1+αQ2+α²Q3`，`α≈0.033045674863435`。每组都是正负两列，每列十个二态位点，理想相对透过率为 1 或 0.9。层号从 1 起的固定写入顺序为 `5,6,4,7,3,8,2,9,1,10`。

**卷积没有偏置；全连接电子偏置保留，共 58 个。没有常数 1 光输入，没有把偏置写进材料。** 三个残差银行是权重编码，不是神经网络的残差连接。

## 训练和推理的界限

先在电子端离线训练浮点参考网络，每次优化后把权重约束在 [-1,1]；再将四个权重矩阵编译为三组材料写入层数。编译后的推理逐组计算正负透射强度、相减、按 α 合成，最后加电子全连接偏置。隐藏层输入按 `s=max(1,max(h))` 编码至 [0,1]，读出恢复 s 后才加偏置。

`compiled.npz` 中只有整数层数、三个残差系数、电子偏置和配置，没有原始浮点权重矩阵。所有 `CompiledCNN` 权重都从层数恢复，运行时没有可训练参数。

这是**理想强度通道仿真**，不包含散粒噪声、电子读噪、实际写入误差、衍射、串扰或 ADC；代码不是激光控制器，也不是完整光路设计。电子差分/加权重构、激活和亮度编码不能称为全光计算。三个残差组的时序和分光损耗尚未硬件验证。

## 资源与历史数字

- 有效有符号权重：1,611 个。
- 三组残差正负列：9,666 列；二态位点：**96,660 个**。
- 网络权重层：卷积 + 三层全连接，共 4 层；每层 3 个残差矩阵，共 12 个独立材料矩阵银行，不是深度方向的 120 层。
- 原来的 95.54% / 94.97% 和 33,440 个位点来自不同的有偏置、单组材料方案，不能作为这版代码的结果。
- 18.55 bit 来自另一个 16 输入、8 输出的理想随机乘加基准，不是网络分类精度、材料状态数或硬件 ENOB。命令 `benchmark` 仅复现该基准。

## 模块

| 文件 | 职责 |
|---|---|
| `config.py` | 选定网络、材料和资源数 |
| `data.py` | 本地 MNIST 校验、固定划分、14×14 预处理 |
| `encoding.py` | 111 档码本、最近级选择、三组残差、十层掩模 |
| `readout.py` | 各银行正负读出、差分和电子重构 |
| `network.py` | 无卷积偏置参考网络、编译网络 |
| `training.py` | 离线训练、验证集选检查点、分类评估 |
| `artifacts.py` | 独立编译模型保存/读取、写入层数和示意坐标导出 |
| `metrics.py` | 明确区分权重误差、乘加位数、分类准确率 |
| `pipeline.py` | 完整实验、先锁定全部种子再测试、源文件哈希 |
| `__main__.py` | 简洁 CLI 入口 |
| `tests/` | 编码、物理重构、数据、序列化和训练集成测试 |

## 使用

在仓库根目录运行，使用 Python 3.10 或更高版本。验证环境为 PyTorch 2.7.0+cu118（实际使用 CPU）、NumPy 2.2.6；只需要 NumPy、PyTorch，测试另需 pytest，不需要 sklearn，不自动联网下载数据。

```powershell
python -m pip install -r residual_material_cnn/requirements.txt
python -B -m residual_material_cnn describe
python -B -m pytest residual_material_cnn/tests -q
python -B -m residual_material_cnn benchmark
python -B -m residual_material_cnn train --output outputs/residual3_new_run --seeds 0 1 2 --epochs 20
```

数据默认读项目根目录 `data/MNIST/raw` 的四个 gzip 文件，固定划分已独立打包在 `assets/split_indices.npz`，与历史划分逐字节一致。可通过 `--data` 和 `--split-indices` 指定本地副本；校验失败直接报错，不重新生成划分。训练始终使用 50k/10k/10k 原划分；本项目已使用过官方测试集，不是新的盲测队列。输出目录必须是新的，避免覆盖旧实验。

三个种子的编译模型和准确率记录已包含在 `examples/`：理想三组残差测试准确率为 **94.95% ± 0.05 pp**，与各自浮点参考的测试预测一致。原始数据和训练生成的大型掩模表不随仓库发布。README 中的训练路径只是示例；目录已经存在时请改用一个新名称。

仅从已编译模型推理，不加载浮点参考：

```powershell
python -B -m residual_material_cnn evaluate --model residual_material_cnn/examples/seed0_compiled.npz --split test --data data/MNIST/raw
python -B -m residual_material_cnn export --model residual_material_cnn/examples/seed0_compiled.npz --output outputs/residual3_masks
```

每个种子导出 `write_masks/write_masks.npz`，轴顺序为 `[残差组,深度层,正负支路,输出,输入]`；1 表示写入低透过率态。`column_counts.csv` 给每列写入数及示意坐标；`electronic_biases.json` 独立保存电子偏置。坐标只说明几何排布，不保证实际层选择性、光学隔离或曝光效果。

准确率、预测差异、权重误差与逐层线性乘加误差写入新实验的 `summary.json`、`REPORT_CN.md`。逐层 bit 使用固定跨度 2、点积除以输入项数；不包含偏置、激活和硬件误差，不应与 ADC 位数或未经统一口径的其他论文指标直接比较。
