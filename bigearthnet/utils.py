# 工具函数文件：提供各种辅助功能，包括自定义损失函数(FocalLoss)、模型保存与加载、性能指标计算、数据可视化以及数据集下载与生成工具
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import os
import gdown
import tarfile
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss实现，针对类别不平衡问题
    
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    
    其中:
    - pt: 模型对真实类别的预测概率
    - gamma: 聚焦参数，增大时更关注困难样本（难以正确分类的样本）
    - alpha: 类别权重参数，形状为[num_classes]，为少数类赋予更高权重
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        初始化Focal Loss
        
        参数:
            alpha: 类别权重，形状为[num_classes]，None表示等权重
            gamma: 聚焦参数，大于0，默认为2，值越大对难分类样本关注越多
            reduction: 损失计算方式，'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重，形状为[num_classes]
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction  # 损失计算方式
        
    def forward(self, inputs, targets):
        """
        前向传播计算损失
        
        参数:
            inputs: 模型输出的logits，形状为 [N, C]
            targets: 真实标签，形状为 [N]
            
        返回:
            loss: 计算的损失值
        """
        # 计算BCE损失 - 二元交叉熵
        bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.alpha)(inputs, targets)
        
        # sigmoid激活获取预测概率
        probs = torch.sigmoid(inputs)
        
        # 根据真实标签选择对应的概率或其补值
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 应用focal衰减因子 - 对高置信度预测降低权重，对低置信度预测提高权重
        focal_weight = (1 - p_t) ** self.gamma
        
        # 计算最终的focal loss
        loss = focal_weight * bce_loss
        
        # 根据reduction方式处理损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    保存模型检查点，用于恢复训练或后续使用
    
    参数:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前训练轮次
        save_path: 保存路径，建议使用.pth后缀
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型和优化器状态
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    
    print(f"模型已保存到 {save_path}")

def load_checkpoint(model, optimizer, load_path):
    """
    加载模型检查点，恢复训练状态
    
    参数:
        model: 模型实例
        optimizer: 优化器实例
        load_path: 加载路径
        
    返回:
        int: 已训练的轮次
    """
    # 检查检查点是否存在
    if not os.path.exists(load_path):
        print(f"检查点 {load_path} 不存在")
        return 0
    
    # 加载检查点
    checkpoint = torch.load(load_path)
    
    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 获取已训练的轮次
    epoch = checkpoint['epoch']
    print(f"从轮次 {epoch} 加载模型")
    
    return epoch

def find_best_threshold(y_true, y_pred_probs, thresholds=None):
    """
    搜索最佳分类阈值，使F1分数最大化
    用于二分类或多标签分类场景
    
    参数:
        y_true: 真实标签 [N, C]
        y_pred_probs: sigmoid后的预测概率 [N, C]
        thresholds: 待评估的阈值列表，默认在0.1-0.9之间搜索
        
    返回:
        best_threshold: 最佳阈值
        best_f1: 最佳F1值
        threshold_results: 不同阈值的评估结果字典列表
    """
    # 设置默认阈值范围
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.1)
    
    best_f1 = 0
    best_threshold = 0.5  # 默认阈值
    threshold_results = []
    
    # 转换为numpy数组进行计算
    y_true_np = y_true.cpu().numpy()
    y_pred_probs_np = y_pred_probs.cpu().numpy()
    
    # 尝试不同阈值
    for threshold in thresholds:
        # 应用阈值获得二分类结果
        y_pred_binary = (y_pred_probs_np > threshold).astype(np.float32)
        
        # 计算评估指标
        f1 = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
        precision = precision_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
        recall = recall_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
        
        # 保存当前阈值的结果
        threshold_results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        # 更新最佳阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, threshold_results

def calculate_class_weights(y_true):
    """
    根据标签频率计算类别权重，用于处理类别不平衡
    
    参数:
        y_true: 真实标签 [N, C] 或 [N]
        
    返回:
        class_weights: 类别权重，形状为 [C]，反比于类别频率
    """
    # 计算每个类别的正样本数量
    positive_counts = y_true.sum(axis=0)
    total_samples = len(y_true)
    
    # 避免除零错误
    positive_counts = np.maximum(positive_counts, 1)
    
    # 计算类别权重：反比于频率
    # 对于稀有类别，给予更高的权重
    class_weights = total_samples / (positive_counts * len(positive_counts))
    
    return class_weights

def calculate_metrics(y_true, y_pred, threshold=0.3, search_threshold=True):
    """
    计算多标签分类指标，全面评估模型性能
    
    参数:
        y_true: 真实标签 [N, C]
        y_pred: 预测分数（logits） [N, C]
        threshold: 二分类阈值，降低至0.3以更容易预测正样本
        search_threshold: 是否搜索最佳阈值
        
    返回:
        dict: 包含各种性能指标的字典
    """
    # 对预测分数进行sigmoid激活，确保范围在[0,1]之间
    y_pred_probs = torch.sigmoid(y_pred).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    # 计算类别权重，用于后续训练优化
    class_weights = calculate_class_weights(y_true_np)
    
    if search_threshold:
        # 搜索最佳阈值
        best_threshold, _, threshold_results = find_best_threshold(
            y_true, torch.sigmoid(y_pred),
            thresholds=np.arange(0.1, 0.91, 0.1)
        )
        # 使用最佳阈值进行预测
        y_pred_binary = (y_pred_probs > best_threshold).astype(np.float32)
        used_threshold = best_threshold
    else:
        # 使用固定阈值
        y_pred_binary = (y_pred_probs > threshold).astype(np.float32)
        used_threshold = threshold
    
    # 计算各种指标
    f1_macro = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    precision_macro = precision_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    
    # 样本级别的F1
    f1_samples = f1_score(y_true_np, y_pred_binary, average='samples', zero_division=0)
    
    # 统计标签分布信息
    positive_counts = y_true_np.sum(axis=0)
    total_samples = len(y_true_np)
    label_frequencies = positive_counts / total_samples
    
    # 随机打印标签分布
    if np.random.random() < 0.1:  # 只在10%的评估中打印，避免日志过多
        print("\n标签分布情况:")
        for i, freq in enumerate(label_frequencies):
            print(f"标签 {i}: {freq:.4f} ({int(positive_counts[i])}/{total_samples})")
    
    return {
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_samples': f1_samples,
        'threshold': used_threshold,
        'label_frequencies': label_frequencies.tolist(),
        'class_weights': class_weights.tolist()
    }

def plot_metrics(train_metrics, val_metrics, save_path=None):
    """
    绘制训练过程中的指标变化曲线，可视化训练进展
    
    参数:
        train_metrics: 训练指标历史记录字典
        val_metrics: 验证指标历史记录字典
        save_path: 图像保存路径
    """
    metrics = ['loss', 'f1_macro', 'precision_macro', 'recall_macro']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 遍历每个指标，绘制对应曲线
    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric in train_metrics:
            ax.plot(train_metrics[metric], label=f'训练 {metric}')
        if metric in val_metrics:
            ax.plot(val_metrics[metric], label=f'验证 {metric}')
        ax.set_xlabel('轮次')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 变化曲线')
        ax.legend()
    
    # 调整布局并保存
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def download_bigearthnet_mini(output_dir='bigearthnet'):
    """
    下载并解压BigEarthNet-Mini数据集
    
    参数:
        output_dir: 输出目录
    
    返回:
        str: 数据集路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # BigEarthNet-Mini的Google Drive ID
    file_id = '1A9wN21biA9IbRH5s_oCRdnFrWO-Mc4_Y'
    
    # 下载路径
    output_path = os.path.join(output_dir, 'bigearthnet-mini.tar.gz')
    
    # 下载文件
    if not os.path.exists(output_path):
        print(f"正在下载BigEarthNet-Mini数据集...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
    
    # 解压文件
    extract_path = os.path.join(output_dir, 'Mini Data')
    if not os.path.exists(extract_path):
        print(f"正在解压数据集...")
        with tarfile.open(output_path) as tar:
            tar.extractall(path=output_dir)
    
    print(f"数据集已准备完成: {extract_path}")
    return extract_path

def generate_mock_data(directory, num_samples=100):
    """
    生成模拟数据，用于测试和开发
    
    参数:
        directory: 目标目录
        num_samples: 生成的样本数量
    """
    # 创建目录
    os.makedirs(directory, exist_ok=True)
    
    # 位置标签
    positions = ['l', 'm', 't']
    
    # 为每个位置创建目录
    for pos in positions:
        pos_dir = os.path.join(directory, f'mini_14{pos}')
        json_dir = os.path.join(directory, f'mini_14{pos}_json')
        
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        
        # 每个位置生成样本
        for i in range(num_samples // 3):
            # 创建随机TIF文件
            tif_path = os.path.join(pos_dir, f'{pos}_sample_{i}.tif')
            # 生成随机图像数据
            img_data = np.random.rand(3, 128, 128).astype(np.float32)
            
            # 保存为numpy文件，模拟TIF
            np.save(tif_path.replace('.tif', '.npy'), img_data)
            
            # 创建随机JSON标注
            json_path = os.path.join(json_dir, f'{pos}_sample_{i}.json')
            
            # 随机决定是否为健康样本
            is_healthy = np.random.random() > 0.7
            
            # 创建JSON内容
            if is_healthy:
                label = 'health'
                grade = 0
            else:
                # 随机疾病等级 (0, 3, 5, 7, 9)
                grade = np.random.choice([3, 5, 7, 9])
                label = f'disease_{grade}'
            
            # 构建JSON结构
            json_data = {
                'imagePath': f'{pos}_sample_{i}.tif',
                'imageHeight': 128,
                'imageWidth': 128,
                'shapes': [
                    {
                        'label': label,
                        'points': [[20, 20], [100, 100]],
                        'shape_type': 'rectangle'
                    }
                ]
            }
            
            # 保存JSON文件
            with open(json_path, 'w') as f:
                import json
                json.dump(json_data, f)
    
    print(f"已生成 {num_samples} 个模拟样本到目录: {directory}")

def visualize_attention(image, attention_map, save_path=None):
    """
    可视化注意力图，帮助理解模型关注区域
    
    参数:
        image: 输入图像，形状为[3, H, W]
        attention_map: 注意力图，形状为[H, W]
        save_path: 保存路径
    """
    # 转换图像为numpy并调整为正确的顺序 [H, W, C]
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:  # [C, H, W] -> [H, W, C]
        image = np.transpose(image, (1, 2, 0))
    
    # 确保值在[0,1]范围内
    image = np.clip(image, 0, 1)
    
    # 转换注意力图为numpy
    if torch.is_tensor(attention_map):
        attention_map = attention_map.cpu().numpy()
    
    # 压缩为2D
    if len(attention_map.shape) > 2:
        attention_map = attention_map.squeeze()
    
    # 绘制原始图像和注意力热图
    plt.figure(figsize=(10, 5))
    
    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 注意力热图
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(attention_map, alpha=0.5, cmap='jet')
    plt.title('注意力热图')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def count_parameters(model):
    """
    统计模型参数量
    
    参数:
        model: PyTorch模型
        
    返回:
        int: 参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)