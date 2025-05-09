# 玉米南方锈病多任务智能识别系统

本项目基于深度学习技术，实现了玉米南方锈病的遥感智能识别，使用无人机获取的多光谱图像(.tif)和人工标注文件(.json)，通过多任务CNN模型实现两大目标：

1. **感染部位分类**：上部/中部/下部 (3分类)
2. **感染等级判断**：无/轻度/中度/重度/极重度 (5分类，对应病害等级：0/3/5/7/9)

## 数据结构

项目使用的数据包括两部分：
- `.tif` 多光谱图像文件
- `.json` 标注文件

数据目录结构：
```
Mini Data/
    ├── mini_14l/        # 下部感染样本的.tif文件
    ├── mini_14l_json/   # 下部感染样本的.json标注
    ├── mini_14m/        # 中部感染样本的.tif文件
    ├── mini_14m_json/   # 中部感染样本的.json标注
    ├── mini_14t/        # 上部感染样本的.tif文件
    └── mini_14t_json/   # 上部感染样本的.json标注
```

## 环境要求

项目基于PyTorch深度学习框架开发，主要依赖项：
- Python 3.6+
- PyTorch 1.7+
- rasterio (用于读取.tif文件)
- scikit-learn
- matplotlib
- seaborn
- tqdm

可通过以下命令安装依赖：
```bash
pip install torch torchvision rasterio scikit-learn matplotlib seaborn tqdm
```

## 训练模型

使用`train.py`脚本训练模型，主要参数：

```bash
python bigearthnet/train.py \
    --data_root bigearthnet/Mini\ Data \
    --img_size 128 \
    --model_type simple \
    --batch_size 32 \
    --epochs 30 \
    --lr 0.001 \
    --output_dir bigearthnet/output
```

参数说明：
- `--data_root`：数据根目录，包含mini_14l、mini_14m、mini_14t子目录
- `--json_root`：JSON标注目录，默认为None(自动推断)
- `--img_size`：图像大小，默认128x128
- `--in_channels`：输入通道数，默认3
- `--train_ratio`：训练集比例，默认0.8
- `--model_type`：模型类型，可选`simple`(简单CNN)或`resnet`(ResNet结构)
- `--batch_size`：批次大小
- `--epochs`：训练轮数
- `--lr`：学习率
- `--output_dir`：输出目录，存放模型和可视化结果

训练过程会输出位置分类和等级分类的准确率，并保存最佳模型。

## 测试模型

使用`test.py`脚本测试训练好的模型：

```bash
python bigearthnet/test.py \
    --data_root bigearthnet/Mini\ Data \
    --model_path bigearthnet/output/best_model.pth \
    --model_type simple \
    --output_dir bigearthnet/test_results
```

参数说明：
- `--data_root`：测试数据根目录
- `--model_path`：训练好的模型权重文件
- `--model_type`：模型类型，需与训练时一致
- `--num_viz_samples`：可视化样本数量，默认10
- `--output_dir`：测试结果输出目录

测试脚本会计算并输出测试集上的准确率、F1分数等指标，并生成混淆矩阵和预测可视化结果。

## 模型结构

项目提供两种模型结构：

1. **简单CNN模型 (DiseaseClassifier)**：
   - 3个卷积块(Conv2d-BatchNorm-ReLU-MaxPool)
   - 两个分类头：位置分类(3类)和等级分类(5类)

2. **ResNet结构模型 (DiseaseResNet)**：
   - 基于残差连接的更深层网络
   - 同样具有两个分类头

## 结果评估

模型性能评估指标包括：
- 位置分类和等级分类的准确率(Accuracy)
- 位置分类和等级分类的F1分数
- 混淆矩阵
- 预测可视化样本

## 常见问题

**Q: 如何处理不同波段数的.tif文件?**  
A: 数据集类会自动处理不同波段数的.tif文件，如果通道数小于3，会复制现有通道；如果大于3，会保留原始通道数。

**Q: 无法读取.tif文件怎么办?**  
A: 确保安装了rasterio库并有适当的GDAL支持。如果遇到读取问题，检查.tif文件的格式和完整性。

**Q: 如何调整两个任务的损失权重?**  
A: 在`train.py`的`train_one_epoch`函数中，可以修改损失计算部分：
```python
loss = alpha * loss_position + beta * loss_grade  # 添加权重系数
```

**Q: 如何使用自己的数据集?**  
A: 准备好.tif图像和对应的.json标注文件，按照项目的目录结构组织，然后按需调整`CornRustDataset`类中的标签解析逻辑。

## 结果示例

训练完成后，可以在输出目录中查看：
- 训练指标曲线图
- 混淆矩阵
- 测试预测可视化结果

## 许可证

本项目仅供学习和研究使用。 