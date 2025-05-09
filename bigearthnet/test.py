# 加载模型和数据：从指定路径加载训练好的模型和测试数据集。
# 模型评估：使用测试数据集评估模型的性能，计算位置和等级的分类准确率和F1分数。
# 混淆矩阵：绘制并保存位置和等级分类的混淆矩阵，帮助分析模型的分类效果。
# 可视化预测：随机选择一些测试样本，展示模型的预测结果与真实标签的对比，并保存可视化图像。
# 保存结果：将测试结果和指标保存到指定的输出目录中。

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import get_model
from dataset import CornRustDataset

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"已保存混淆矩阵图像到: {save_path}")
    plt.close()

def visualize_predictions(model, test_loader, device, save_dir, num_samples=5):
    """可视化模型预测结果"""
    model.eval()
    
    # 位置和等级的类别名称
    position_names = ['下部', '中部', '上部']
    grade_names = ['无(0)', '轻度(3)', '中度(5)', '重度(7)', '极重度(9)']
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集一些样本
    samples = []
    with torch.no_grad():
        for images, position_labels, grade_labels in test_loader:
            for i in range(min(len(images), num_samples - len(samples))):
                img = images[i].cpu().numpy()
                pos_label = position_labels[i].item()
                grade_label = grade_labels[i].item()
                
                # 获取模型预测
                image_tensor = images[i:i+1].to(device)
                pos_logits, grade_logits = model(image_tensor)
                pos_pred = torch.argmax(pos_logits, dim=1).item()
                grade_pred = torch.argmax(grade_logits, dim=1).item()
                
                samples.append((img, pos_label, grade_label, pos_pred, grade_pred))
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
    
    # 可视化样本
    for i, (img, pos_label, grade_label, pos_pred, grade_pred) in enumerate(samples):
        # 转换为RGB格式显示
        if img.shape[0] > 3:
            # 如果通道数大于3，只显示前3个通道
            display_img = img[:3]
        else:
            display_img = img
            
        # 转置为 (H, W, C) 格式
        display_img = np.transpose(display_img, (1, 2, 0))
        
        # 归一化到 [0, 1] 范围
        if display_img.max() > 1.0:
            display_img = display_img / 255.0
            
        plt.figure(figsize=(8, 8))
        plt.imshow(display_img)
        
        # 添加预测结果标题
        true_pos = position_names[pos_label]
        pred_pos = position_names[pos_pred]
        true_grade = grade_names[grade_label]
        pred_grade = grade_names[grade_pred]
        
        correct_pos = '✓' if pos_pred == pos_label else '✗'
        correct_grade = '✓' if grade_pred == grade_label else '✗'
        
        title = f"样本 {i+1}\n位置: 真实={true_pos} 预测={pred_pos} {correct_pos}\n等级: 真实={true_grade} 预测={pred_grade} {correct_grade}"
        plt.title(title)
        plt.axis('off')
        
        # 保存图像
        save_path = os.path.join(save_dir, f"sample_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        
    print(f"已保存 {len(samples)} 个预测可视化结果到 {save_dir}")

def test_model(model, test_loader, device, output_dir):
    """测试模型并计算指标"""
    model.eval()
    
    # 类别名称
    position_names = ['下部', '中部', '上部']
    grade_names = ['无(0)', '轻度(3)', '中度(5)', '重度(7)', '极重度(9)']
    
    # 收集预测和真实标签
    position_preds = []
    position_labels = []
    grade_preds = []
    grade_labels = []
    
    print("评估模型中...")
    with torch.no_grad():
        for images, pos_label, grade_label in tqdm(test_loader):
            images = images.to(device)
            
            # 获取模型预测
            pos_logits, grade_logits = model(images)
            _, pos_pred = torch.max(pos_logits, 1)
            _, grade_pred = torch.max(grade_logits, 1)
            
            # 收集结果
            position_preds.extend(pos_pred.cpu().numpy())
            position_labels.extend(pos_label.cpu().numpy())
            grade_preds.extend(grade_pred.cpu().numpy())
            grade_labels.extend(grade_label.cpu().numpy())
    
    # 计算混淆矩阵
    position_cm = confusion_matrix(position_labels, position_preds)
    grade_cm = confusion_matrix(grade_labels, grade_preds)
    
    # 计算分类报告
    position_report = classification_report(position_labels, position_preds, 
                                           target_names=position_names, 
                                           digits=4, output_dict=True)
    grade_report = classification_report(grade_labels, grade_preds, 
                                        target_names=grade_names, 
                                        digits=4, output_dict=True)
    
    # 打印结果
    print("\n位置分类报告:")
    print(classification_report(position_labels, position_preds, 
                               target_names=position_names, digits=4))
    
    print("\n等级分类报告:")
    print(classification_report(grade_labels, grade_preds, 
                               target_names=grade_names, digits=4))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        position_cm, position_names, 
        f"位置分类混淆矩阵 (准确率: {position_report['accuracy']:.4f})",
        os.path.join(output_dir, "position_cm_test.png")
    )
    
    plot_confusion_matrix(
        grade_cm, grade_names, 
        f"等级分类混淆矩阵 (准确率: {grade_report['accuracy']:.4f})",
        os.path.join(output_dir, "grade_cm_test.png")
    )
    
    # 返回结果
    return {
        'position_accuracy': position_report['accuracy'],
        'position_f1': position_report['macro avg']['f1-score'],
        'grade_accuracy': grade_report['accuracy'],
        'grade_f1': grade_report['macro avg']['f1-score']
    }

def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建测试数据集
    test_datasets = []
    for pos in ['14l', '14m', '14t']:
        data_dir = os.path.join(args.data_root, f'mini_{pos}')
        json_dir = None if args.json_root is None else os.path.join(args.json_root, f'mini_{pos}_json')
        
        if os.path.exists(data_dir):
            dataset = CornRustDataset(data_dir, json_dir, img_size=args.img_size)
            test_datasets.append(dataset)
        else:
            print(f"警告: 数据目录不存在: {data_dir}")
    
    # 合并所有数据集
    if not test_datasets:
        raise ValueError(f"没有找到有效的数据集目录在 {args.data_root}")
    
    # 如果只有一个数据集，则直接使用它
    if len(test_datasets) == 1:
        test_dataset = test_datasets[0]
    else:
        # 合并多个数据集
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 加载模型
    model = get_model(model_type=args.model_type, in_channels=args.in_channels, img_size=args.img_size)
    model = model.to(device)
    
    # 加载模型参数
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"已加载模型: {args.model_path}")
    else:
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")
    
    # 测试模型
    metrics = test_model(model, test_loader, device, args.output_dir)
    
    # 可视化一些预测结果
    visualize_predictions(model, test_loader, device, os.path.join(args.output_dir, 'visualizations'), args.num_viz_samples)
    
    # 保存指标
    with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"位置分类准确率: {metrics['position_accuracy']:.4f}\n")
        f.write(f"位置分类F1分数: {metrics['position_f1']:.4f}\n")
        f.write(f"等级分类准确率: {metrics['grade_accuracy']:.4f}\n")
        f.write(f"等级分类F1分数: {metrics['grade_f1']:.4f}\n")
    
    print(f"\n测试完成! 结果已保存到 {args.output_dir}")
    print(f"位置分类准确率: {metrics['position_accuracy']:.4f}")
    print(f"位置分类F1分数: {metrics['position_f1']:.4f}")
    print(f"等级分类准确率: {metrics['grade_accuracy']:.4f}")
    print(f"等级分类F1分数: {metrics['grade_f1']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='玉米南方锈病多任务分类模型测试')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='bigearthnet/Mini Data', help='测试数据根目录')
    parser.add_argument('--json_root', type=str, default=None, help='JSON标注根目录，如果为None，则使用data_root + "_json"')
    parser.add_argument('--img_size', type=int, default=128, help='图像大小')
    parser.add_argument('--in_channels', type=int, default=3, help='输入通道数')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'resnet'], help='模型类型')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--num_viz_samples', type=int, default=10, help='可视化样本数量')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='bigearthnet/test_results', help='输出目录')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    args = parser.parse_args()
    
    main(args) 