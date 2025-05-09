# 训练脚本文件：实现模型训练、验证和测试的主要流程，包括数据加载、损失函数定义、优化器配置、训练循环、模型评估和结果保存等功能
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 导入自定义模块
from dataset import CornRustDataset, get_dataloaders
from model import get_model
from utils import save_checkpoint, load_checkpoint, calculate_metrics, plot_metrics, download_bigearthnet_mini, FocalLoss, calculate_class_weights

# 定义数据增强变换
def get_data_transforms(train=True):
    """
    获取数据增强变换
    
    参数:
        train: 是否为训练模式，训练时应用数据增强，验证/测试时不应用
    
    返回:
        transforms: 数据增强变换组合
    """
    if train:
        # 训练时使用多种数据增强方法提高模型泛化能力
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转，增加样本多样性
            transforms.RandomVerticalFlip(),    # 随机垂直翻转
            transforms.RandomRotation(15),      # 随机旋转，角度范围为±15度
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)  # 随机颜色变化
        ])
    else:
        # 验证/测试时不进行数据增强，保持原始图像
        return None

def train_one_epoch(model, train_loader, optimizer, position_criterion, grade_criterion, device, task_weights=[0.7, 0.3]):
    """
    训练模型一个epoch
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器实例
        position_criterion: 位置分类的损失函数(CrossEntropy)
        grade_criterion: 等级预测的损失函数(MSE回归损失)
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重，默认[0.7, 0.3]表示位置任务占70%，等级任务占30%
        
    返回:
        dict: 包含训练指标的字典，包括总损失、位置损失、等级损失、位置准确率和等级MAE
    """
    model.train()  # 设置模型为训练模式，启用Dropout和BatchNorm
    total_loss = 0.0  # 累计总损失
    position_loss_sum = 0.0  # 累计位置分类损失
    grade_loss_sum = 0.0  # 累计等级回归损失
    position_correct = 0  # 位置分类正确样本数
    total_samples = 0  # 总样本数
    grade_mae_sum = 0.0  # 等级预测平均绝对误差累计
    
    # 使用tqdm显示进度条，增强用户体验
    progress_bar = tqdm(train_loader, desc="训练中")
    
    # 遍历训练数据批次
    for images, position_labels, grade_labels in progress_bar:
        # 将数据移动到指定设备
        images = images.to(device)  # 输入图像
        position_labels = position_labels.to(device)  # 位置标签（0,1,2）
        
        # 将等级标签转换为float类型并添加维度，用于回归任务
        # 从形状[batch_size]变为[batch_size, 1]
        grade_labels = grade_labels.float().unsqueeze(1).to(device)
        
        # 前向传播
        position_logits, grade_values = model(images)
        
        # 计算位置分类损失 - 使用CrossEntropy
        loss_position = position_criterion(position_logits, position_labels)
        
        # 计算等级回归损失 - 使用MSE
        loss_grade = grade_criterion(grade_values, grade_labels)
        
        # 使用任务权重组合损失 - 位置分类权重0.7，等级回归权重0.3
        loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
        
        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size  # 累加总损失
        position_loss_sum += loss_position.item() * batch_size  # 累加位置损失
        grade_loss_sum += loss_grade.item() * batch_size  # 累加等级损失
        
        # 计算位置分类准确率
        _, position_preds = torch.max(position_logits, 1)  # 获取预测类别
        position_correct += (position_preds == position_labels).sum().item()  # 统计正确预测数
        
        # 计算等级预测MAE(平均绝对误差)
        grade_mae = torch.abs(grade_values - grade_labels).mean().item()  # 当前批次的MAE
        grade_mae_sum += grade_mae * batch_size  # 累加MAE
        
        total_samples += batch_size  # 累加样本数
        
        # 更新进度条显示当前性能指标
        progress_bar.set_postfix({
            'loss': loss.item(),  # 当前批次损失
            'pos_acc': position_correct / total_samples,  # 当前位置准确率
            'grade_mae': grade_mae_sum / total_samples  # 当前平均等级MAE
        })
    
    # 计算整个epoch的平均指标
    avg_loss = total_loss / total_samples  # 平均总损失
    avg_position_loss = position_loss_sum / total_samples  # 平均位置损失
    avg_grade_loss = grade_loss_sum / total_samples  # 平均等级损失
    position_accuracy = position_correct / total_samples  # 位置分类准确率
    grade_mae = grade_mae_sum / total_samples  # 等级预测平均MAE
    
    # 返回包含所有训练指标的字典
    return {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'grade_mae': grade_mae
    }

def evaluate(model, val_loader, position_criterion, grade_criterion, device, task_weights=[0.7, 0.3]):
    """
    评估模型在验证集上的性能
    
    参数:
        model: 模型实例
        val_loader: 验证数据加载器
        position_criterion: 位置分类的损失函数
        grade_criterion: 等级预测的损失函数（回归损失）
        device: 计算设备(CPU/GPU)
        task_weights: 任务权重，默认[0.7, 0.3]表示位置任务占70%，等级任务占30%
        
    返回:
        dict: 包含详细评估指标的字典，包括多种性能指标
    """
    model.eval()  # 设置模型为评估模式，禁用Dropout
    total_loss = 0.0  # 累计总损失
    position_loss_sum = 0.0  # 累计位置分类损失
    grade_loss_sum = 0.0  # 累计等级回归损失
    
    # 收集所有预测和真实标签，用于计算整体指标
    position_preds_all = []  # 所有位置预测
    position_labels_all = []  # 所有位置真实标签
    grade_values_all = []  # 所有等级预测
    grade_labels_all = []  # 所有等级真实标签
    
    with torch.no_grad():  # 关闭梯度计算，减少内存占用
        for images, position_labels, grade_labels in val_loader:
            # 将数据移动到指定设备
            images = images.to(device)
            position_labels = position_labels.to(device)
            # 将等级标签转换为float类型并添加维度，用于回归
            grade_labels = grade_labels.float().unsqueeze(1).to(device)
            
            # 前向传播
            position_logits, grade_values = model(images)
            
            # 计算损失
            loss_position = position_criterion(position_logits, position_labels)
            loss_grade = grade_criterion(grade_values, grade_labels)
            
            # 使用任务权重组合损失
            loss = task_weights[0] * loss_position + task_weights[1] * loss_grade
            
            # 统计指标
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            position_loss_sum += loss_position.item() * batch_size
            grade_loss_sum += loss_grade.item() * batch_size
            
            # 获取位置预测类别
            _, position_preds = torch.max(position_logits, 1)
            
            # 收集预测和标签用于计算整体指标
            position_preds_all.extend(position_preds.cpu().numpy())
            position_labels_all.extend(position_labels.cpu().numpy())
            grade_values_all.extend(grade_values.cpu().numpy())
            grade_labels_all.extend(grade_labels.cpu().numpy())
    
    # 计算平均指标
    total_samples = len(val_loader.dataset)
    avg_loss = total_loss / total_samples
    avg_position_loss = position_loss_sum / total_samples
    avg_grade_loss = grade_loss_sum / total_samples
    
    # 计算位置分类详细指标
    position_accuracy = accuracy_score(position_labels_all, position_preds_all)  # 准确率
    position_f1 = f1_score(position_labels_all, position_preds_all, average='macro')  # 宏平均F1
    position_f1_per_class = f1_score(position_labels_all, position_preds_all, average=None)  # 每类F1
    position_cm = confusion_matrix(position_labels_all, position_preds_all)  # 混淆矩阵
    position_precision = precision_score(position_labels_all, position_preds_all, average='macro')  # 宏平均精确率
    position_recall = recall_score(position_labels_all, position_preds_all, average='macro')  # 宏平均召回率
    
    # 计算等级回归指标
    grade_values_all = np.array(grade_values_all)
    grade_labels_all = np.array(grade_labels_all)
    grade_mae = np.mean(np.abs(grade_values_all - grade_labels_all))  # 平均绝对误差
    
    # 计算±2误差容忍率 - 在实际应用中，等级预测误差在±2范围内可接受
    tolerance = 2.0
    grade_tolerance_accuracy = np.mean(np.abs(grade_values_all - grade_labels_all) <= tolerance)
    
    # 返回包含所有评估指标的字典
    return {
        'loss': avg_loss,
        'position_loss': avg_position_loss,
        'grade_loss': avg_grade_loss,
        'position_accuracy': position_accuracy,
        'position_f1': position_f1,
        'position_f1_per_class': position_f1_per_class,
        'position_precision': position_precision,
        'position_recall': position_recall,
        'position_cm': position_cm,
        'grade_mae': grade_mae,
        'grade_tolerance_accuracy': grade_tolerance_accuracy
    }

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """
    绘制混淆矩阵可视化图
    
    参数:
        cm: 混淆矩阵
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径，如果提供则保存图像
    """
    plt.figure(figsize=(10, 8))  # 设置图像大小
    
    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # 添加标签和标题
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(title)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    plt.close()  # 关闭图形，释放内存

def plot_metrics(metrics_history, save_dir):
    """
    绘制训练过程中的指标变化曲线
    
    参数:
        metrics_history: 包含各指标历史记录的字典
        save_dir: 图像保存目录
    """
    # 创建一个2x2的图表布局，显示4种主要指标
    plt.figure(figsize=(16, 12))
    
    # 绘制损失曲线 - 右上角
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='训练损失')
    plt.plot(metrics_history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴只显示整数
    
    # 绘制位置准确率和F1曲线 - 右上角
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_position_accuracy'], label='训练位置准确率')
    plt.plot(metrics_history['val_position_accuracy'], label='验证位置准确率')
    plt.plot(metrics_history['val_position_f1'], label='验证位置F1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / F1')
    plt.title('位置分类性能')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制等级MAE曲线 - 左下角
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['train_grade_mae'], label='训练等级MAE')
    plt.plot(metrics_history['val_grade_mae'], label='验证等级MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('等级预测平均绝对误差')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制等级容忍率曲线 - 右下角
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['val_grade_tolerance'], label='验证等级容忍率(±2)')
    plt.xlabel('Epoch')
    plt.ylabel('Tolerance Rate')
    plt.title('等级预测容忍率')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 调整子图布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()  # 关闭图形，释放内存

def main(args):
    """
    主训练函数，处理整个训练流程
    
    参数:
        args: 命令行参数解析后的对象，包含所有训练超参数
    """
    # 设置随机种子，确保实验可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置计算设备 (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录，用于保存模型和结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取训练和验证数据加载器
    train_loader, val_loader = get_dataloaders(
        args.data_root,
        json_root=args.json_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        train_ratio=args.train_ratio,
        aug_prob=args.aug_prob  # 数据增强应用概率
    )
    print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型实例
    model = get_model(model_type=args.model_type, in_channels=args.in_channels, img_size=args.img_size)
    model = model.to(device)  # 将模型移动到指定设备
    print(f"使用模型: {args.model_type}")
    
    # 计算样本权重以处理类别不平衡问题
    if args.weighted_loss:
        print("使用加权损失函数处理类别不平衡...")
        # 计算位置类别权重
        position_weights = None
        if hasattr(train_loader.dataset, 'get_class_weights'):
            position_weights, _ = train_loader.dataset.get_class_weights()
            position_weights = torch.tensor(position_weights, dtype=torch.float32).to(device)
            print(f"位置类别权重: {position_weights}")
    else:
        position_weights = None
    
    # 定义损失函数
    # 位置分类损失函数 - 使用CrossEntropyLoss或FocalLoss
    if args.loss_type == 'ce':
        # 交叉熵损失，可选使用类别权重
        position_criterion = nn.CrossEntropyLoss(weight=position_weights)
    elif args.loss_type == 'focal':
        # 使用Focal Loss处理类别不平衡问题
        from utils import FocalLoss
        position_criterion = FocalLoss(alpha=position_weights, gamma=args.focal_gamma)
    else:
        raise ValueError(f"不支持的损失函数类型: {args.loss_type}")
    
    # 等级回归损失函数 - 使用MSE均方误差
    grade_criterion = nn.MSELoss()
    
    # 定义优化器
    if args.optimizer == 'adam':
        # Adam优化器：自适应学习率，适合大多数问题
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        # SGD优化器：经典随机梯度下降，需要合适的学习率调度
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer}")
    
    # 学习率调度器 - 根据训练进展动态调整学习率
    if args.lr_scheduler == 'plateau':
        # 当指标不再改善时降低学习率
        # 使用位置F1分数作为监控指标，提高对少数类的关注
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.scheduler_patience, verbose=True
        )
    elif args.lr_scheduler == 'cosine':
        # 余弦退火学习率，从初始值平滑降低到最小值
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == 'step':
        # 阶梯式学习率，每固定步数降低一次
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=0.5
        )
    else:
        scheduler = None
    
    # 加载检查点（如果有），支持从中断处继续训练
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"成功从轮次 {start_epoch - 1} 恢复训练")
    
    # 跟踪指标历史记录，用于后续绘图和分析
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'train_position_accuracy': [],
        'val_position_accuracy': [],
        'train_grade_mae': [],  # 修改为MAE
        'val_grade_mae': [],    # 修改为MAE
        'val_position_f1': [],
        'val_grade_tolerance': [],  # 添加容忍率
        'learning_rates': []
    }
    
    # 训练循环
    best_val_loss = float('inf')  # 记录最佳验证损失
    best_position_f1 = 0.0  # 记录最佳位置F1分数
    best_grade_mae = float('inf')  # 记录最佳等级MAE
    early_stop_counter = 0  # 早停计数器
    
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, 
            position_criterion, grade_criterion,
            device, args.task_weights
        )
        
        # 在验证集上评估模型
        val_metrics = evaluate(
            model, val_loader, 
            position_criterion, grade_criterion,
            device, args.task_weights
        )
        
        # 更新学习率 - 使用位置F1分数作为指标
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_metrics['position_f1'])  # 监控F1分数而非损失
            else:
                scheduler.step()
        
        # 保存当前学习率
        metrics_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 保存本轮指标到历史记录
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['train_position_accuracy'].append(train_metrics['position_accuracy'])
        metrics_history['val_position_accuracy'].append(val_metrics['position_accuracy'])
        metrics_history['train_grade_mae'].append(train_metrics['grade_mae'])
        metrics_history['val_grade_mae'].append(val_metrics['grade_mae'])
        metrics_history['val_position_f1'].append(val_metrics['position_f1'])
        metrics_history['val_grade_tolerance'].append(val_metrics['grade_tolerance_accuracy'])
        
        # 打印当前性能
        print(f"训练损失: {train_metrics['loss']:.4f}, 位置准确率: {train_metrics['position_accuracy']:.4f}, 等级MAE: {train_metrics['grade_mae']:.4f}")
        print(f"验证损失: {val_metrics['loss']:.4f}, 位置准确率: {val_metrics['position_accuracy']:.4f} (F1: {val_metrics['position_f1']:.4f}), 等级MAE: {val_metrics['grade_mae']:.4f} (容忍率: {val_metrics['grade_tolerance_accuracy']:.4f})")
        
        # 定期保存检查点
        if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'metrics': metrics_history
            }, checkpoint_path)
        
        # 保存最好的模型 - 基于位置F1分数
        if val_metrics['position_f1'] > best_position_f1:
            best_position_f1 = val_metrics['position_f1']  # 更新最佳F1
            best_val_loss = val_metrics['loss']  # 更新最佳损失
            best_grade_mae = val_metrics['grade_mae']  # 更新最佳MAE
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': {
                    'position_accuracy': val_metrics['position_accuracy'],
                    'position_f1': val_metrics['position_f1'],
                    'grade_mae': val_metrics['grade_mae'],
                    'grade_tolerance': val_metrics['grade_tolerance_accuracy']
                }
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print("保存最佳模型 (基于位置F1分数).")
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1  # 增加早停计数器
        
        # 在最后一个epoch或达到早停条件时，绘制混淆矩阵
        if epoch == args.epochs - 1 or early_stop_counter >= args.patience:
            # 绘制位置分类混淆矩阵，可视化分类性能
            position_class_names = ['下部', '中部', '上部']
            plot_confusion_matrix(
                val_metrics['position_cm'],  # 混淆矩阵
                position_class_names,  # 类别名称
                f'位置分类混淆矩阵 (F1: {val_metrics["position_f1"]:.4f})',  # 标题
                os.path.join(args.output_dir, 'position_confusion_matrix.png')  # 保存路径
            )
        
        # 早停机制 - 当位置F1分数在连续多个epoch没有提升时提前结束训练
        if early_stop_counter >= args.patience:
            print(f"位置F1分数在{args.patience}个epoch内没有改善，停止训练。")
            break
    
    # 训练结束后绘制所有指标曲线
    plot_metrics(metrics_history, args.output_dir)
    
    # 打印最终训练结果
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳位置分类F1: {best_position_f1:.4f}")
    print(f"对应等级预测MAE: {best_grade_mae:.4f}")

if __name__ == '__main__':
    # 命令行参数解析，定义所有可配置的训练参数
    parser = argparse.ArgumentParser(description='玉米南方锈病多任务分类训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='bigearthnet/Mini Data', 
                        help='数据根目录，包含mini_14l、mini_14m、mini_14t子目录')
    parser.add_argument('--json_root', type=str, default=None, 
                        help='JSON标注根目录，如果为None，则使用data_root + "_json"')
    parser.add_argument('--img_size', type=int, default=128, 
                        help='图像处理尺寸，所有图像将调整为该大小')
    parser.add_argument('--in_channels', type=int, default=3, 
                        help='输入图像通道数，可以为3(RGB)或多通道(多光谱)')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='训练集比例，剩余部分用作验证集')
    parser.add_argument('--aug_prob', type=float, default=0.5, 
                        help='数据增强应用概率，控制每种增强方法的应用频率')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='resnet_plus', 
                        choices=['simple', 'resnet', 'resnet_plus'], 
                        help='模型类型：simple=简单CNN，resnet=标准ResNet，resnet_plus=带注意力的增强ResNet')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='ce', 
                        choices=['ce', 'focal'], 
                        help='位置分类的损失函数类型: ce=CrossEntropy, focal=FocalLoss(处理类别不平衡)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, 
                        help='Focal Loss的gamma参数，控制难样本关注度，大于1时增加对难分类样本的关注')
    parser.add_argument('--weighted_loss', action='store_true', 
                        help='是否使用加权损失函数处理类别不平衡，为少数类赋予更高权重')
    parser.add_argument('--task_weights', type=str, default='0.7,0.3', 
                        help='任务权重，用逗号分隔，如"0.7,0.3"表示位置任务占70%，等级任务占30%')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd'], 
                        help='优化器类型：adam=自适应学习率，sgd=随机梯度下降')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        help='最小学习率，学习率衰减的下限')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='权重衰减系数，用于L2正则化防止过拟合')
    
    # 学习率调度器参数
    parser.add_argument('--lr_scheduler', type=str, default='plateau', 
                        choices=['plateau', 'cosine', 'step', 'none'], 
                        help='学习率调度器类型：plateau=根据指标调整，cosine=余弦退火，step=阶梯式衰减')
    parser.add_argument('--scheduler_patience', type=int, default=3, 
                        help='ReduceLROnPlateau调度器的耐心值，连续几个epoch无改善后降低学习率')
    parser.add_argument('--scheduler_step_size', type=int, default=10, 
                        help='StepLR调度器的步长，每多少个epoch降低一次学习率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='批次大小，根据GPU内存调整')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='训练总轮数')
    parser.add_argument('--patience', type=int, default=10, 
                        help='早停耐心值，监控位置F1分数，连续多少个epoch无改善后停止训练')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子，确保实验可重复性')
    parser.add_argument('--no_cuda', action='store_true', 
                        help='禁用CUDA，强制使用CPU训练')
    parser.add_argument('--resume', type=str, default=None, 
                        help='恢复训练的检查点路径，用于从中断处继续训练')
    parser.add_argument('--save_interval', type=int, default=5, 
                        help='保存检查点的间隔epoch数，控制保存频率')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='bigearthnet/output', 
                        help='输出目录，用于保存模型、日志和可视化结果')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载线程数，一般设置为CPU核心数')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 处理任务权重 - 将字符串转换为浮点数列表
    try:
        args.task_weights = [float(w) for w in args.task_weights.split(',')]
        assert len(args.task_weights) == 2, "任务权重必须有两个值，用于位置和等级任务"
        # 归一化权重确保总和为1
        total = sum(args.task_weights)
        args.task_weights = [w / total for w in args.task_weights]
    except:
        print("任务权重格式错误，使用默认权重[0.7, 0.3]")
        args.task_weights = [0.7, 0.3]
    
    # 执行主函数
    main(args)