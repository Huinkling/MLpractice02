# 玉米南方锈病多任务分类系统

本项目实现了一个基于深度学习的玉米南方锈病遥感识别系统，使用无人机获取的.tif多光谱图像和.json标注文件，实现两个任务的同时分类：

1. **感染部位分类**: 识别病害发生在植株的上部/中部/下部 (3分类)
2. **感染等级分类**: 识别病害的严重程度，分为无/轻度/中度/重度/极重度 (5分类)

## 主要改进

### 1. 数据处理改进

- **类别不平衡处理**: 实现了类别权重计算，针对稀有类别进行加权，提高模型对少数类的识别能力
- **增强的数据增强**: 实现了多种数据增强方法（翻转、旋转、亮度/对比度调整、随机裁剪等），提高模型的泛化能力
- **可配置的增强概率**: 支持设置数据增强应用的概率参数，灵活控制增强的强度

### 2. 模型架构改进

- **多任务学习架构**: 采用共享特征提取器 + 双任务分类头的架构设计
- **注意力机制增强**: 在ResNet模型基础上增加了通道注意力和空间注意力模块，提高特征提取能力
- **任务特定注意力**: 为两个分类任务设计了专用的注意力模块，增强任务相关特征的捕获

### 3. 训练策略改进

- **多种损失函数**: 支持CrossEntropy和FocalLoss，后者对难分样本给予更高权重
- **灵活的任务权重调整**: 可配置位置分类和等级分类任务的相对权重
- **丰富的学习率调度**: 支持多种学习率调度策略（ReduceLROnPlateau, CosineAnnealing, StepLR）
- **检查点管理**: 完善的模型保存和恢复机制，支持训练中断后继续训练

### 4. 评估指标改进

- **全面的评估指标**: 计算准确率、F1分数、精确率、召回率、混淆矩阵等
- **单独评估两个任务**: 分别评估位置分类和等级分类的性能，更清晰地了解模型在各任务上的表现
- **各类别F1分数**: 提供每个类别单独的F1分数，方便定位模型弱点

## 使用方法

### 1. 安装依赖

```bash
pip install torch torchvision rasterio numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. 数据准备

将无人机获取的多光谱图像(.tif)和标注文件(.json)按照以下结构组织：

```
bigearthnet/Mini Data/
├── mini_14l/         # 下部感染样本的.tif图像
├── mini_14m/         # 中部感染样本的.tif图像
├── mini_14t/         # 上部感染样本的.tif图像
├── mini_14l_json/    # 下部感染样本的.json标注
├── mini_14m_json/    # 中部感染样本的.json标注
└── mini_14t_json/    # 上部感染样本的.json标注
```

### 3. 训练模型

使用默认参数训练模型：

```bash
python bigearthnet/train.py
```

使用高级参数训练模型：

```bash
python bigearthnet/train.py \
    --data_root bigearthnet/Mini Data \
    --model_type resnet_plus \
    --img_size 256 \
    --in_channels 3 \
    --batch_size 32 \
    --epochs 50 \
    --loss_type focal \
    --weighted_loss \
    --task_weights 0.5,0.5 \
    --optimizer adam \
    --lr 0.001 \
    --lr_scheduler cosine \
    --aug_prob 0.7 \
    --output_dir bigearthnet/output
```

### 4. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | 数据根目录 | bigearthnet/Mini Data |
| `--json_root` | JSON标注根目录 | None (自动推断) |
| `--img_size` | 图像大小 | 128 |
| `--in_channels` | 输入通道数 | 3 |
| `--train_ratio` | 训练集比例 | 0.8 |
| `--aug_prob` | 数据增强应用概率 | 0.5 |
| `--model_type` | 模型类型(simple/resnet/resnet_plus) | simple |
| `--loss_type` | 损失函数类型(ce/focal) | ce |
| `--focal_gamma` | Focal Loss的gamma参数 | 2.0 |
| `--weighted_loss` | 使用加权损失函数 | False |
| `--task_weights` | 任务权重，用逗号分隔 | 0.5,0.5 |
| `--optimizer` | 优化器类型(adam/sgd) | adam |
| `--lr` | 学习率 | 0.001 |
| `--min_lr` | 最小学习率 | 1e-6 |
| `--weight_decay` | 权重衰减 | 1e-4 |
| `--lr_scheduler` | 学习率调度器类型 | plateau |
| `--batch_size` | 批次大小 | 32 |
| `--epochs` | 训练轮数 | 30 |
| `--patience` | 早停耐心值 | 10 |
| `--resume` | 恢复训练的检查点路径 | None |
| `--output_dir` | 输出目录 | bigearthnet/output |

## 模型结构

本项目提供三种模型结构：

1. **Simple CNN (DiseaseClassifier)**: 简单的三层CNN架构，适合快速实验
2. **ResNet (DiseaseResNet)**: 基于ResNet的架构，提供更强的特征提取能力
3. **ResNet Plus (DiseaseResNetPlus)**: 在ResNet基础上增加注意力机制，是性能最佳的模型

## 结果可视化

训练完成后，在输出目录中可以找到：

- **training_metrics.png**: 训练过程中的损失和准确率曲线
- **position_confusion_matrix.png**: 位置分类的混淆矩阵
- **grade_confusion_matrix.png**: 等级分类的混淆矩阵
- **best_model.pth**: 验证集表现最佳的模型权重

## 注意事项

- 本项目使用TIF格式的多光谱图像，如果您的数据集为其他格式，需要相应修改dataset.py
- 类别标签是从文件名和目录结构中提取的，需要确保您的数据组织符合期望格式
- 对于大型数据集，建议使用`--model_type resnet_plus`获得更好的性能