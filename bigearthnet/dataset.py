# 数据集处理文件：负责读取和处理玉米南方锈病的多光谱图像数据，包括TIF图像加载、JSON标注解析、数据增强以及多任务学习的标签预处理（感染部位分类和感染等级回归）
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import json
from torchvision.transforms import functional as transforms_functional
import random
from collections import Counter

class CornRustDataset(Dataset):
    """
    玉米南方锈病数据集加载类
    处理.tif多光谱图像和.json标注文件
    支持多任务学习:
    1. 感染部位: 上部/中部/下部 -> 0/1/2 (3分类)
    2. 感染等级: 无/轻度/中度/重度/极重度 -> 0/3/5/7/9 -> 0-9 (回归任务)
    """
    def __init__(self, data_dir, json_dir=None, transform=None, img_size=128):
        """
        初始化玉米南方锈病数据集
        
        参数:
            data_dir (str): .tif文件的数据集目录，包含多光谱图像
            json_dir (str, optional): .json标注文件目录，如果为None，则使用data_dir + '_json'
            transform (callable, optional): 数据增强和转换函数
            img_size (int, optional): 图像统一调整大小，默认128x128
        """
        self.data_dir = data_dir
        self.json_dir = json_dir if json_dir else data_dir + '_json'
        self.transform = transform
        self.img_size = img_size
        
        # 映射字典 - 将文本标签映射为数值标签
        # 位置标签：l(下部)=0, m(中部)=1, t(上部)=2
        self.position_map = {"l": 0, "m": 1, "t": 2}  # 下部/中部/上部
        
        # 等级标签：以前是将0/3/5/7/9映射为0/1/2/3/4，现在直接使用原始值进行回归
        # 保留此映射用于向后兼容和统计
        self.grade_map = {0: 0, 3: 1, 5: 2, 7: 3, 9: 4}  # 无/轻度/中度/重度/极重度
        
        # 获取所有样本文件路径对
        self.samples = self._get_samples()
        
        # 缓存标签分布以计算类别权重 - 用于处理数据不平衡
        self.position_labels = []
        self.grade_labels = []
        self._cache_labels()
        
    def _get_samples(self):
        """
        获取所有样本路径和对应json文件路径
        
        返回:
            list: 包含(tif_path, json_path)元组的列表，每个元组对应一个样本
        """
        samples = []
        
        # 检查目录是否存在
        if not os.path.exists(self.data_dir) or not os.path.exists(self.json_dir):
            print(f"数据目录不存在: {self.data_dir} 或 {self.json_dir}")
            return samples
        
        # 查找所有.tif文件
        tif_files = [f for f in os.listdir(self.data_dir) if f.endswith('.tif')]
        
        # 遍历tif文件，找到对应的json文件
        for tif_file in tif_files:
            tif_path = os.path.join(self.data_dir, tif_file)
            # 找到对应的json文件
            json_file = tif_file.replace('.tif', '.json')
            json_path = os.path.join(self.json_dir, json_file)
            
            # 检查json文件是否存在
            if os.path.exists(json_path):
                samples.append((tif_path, json_path))
            else:
                print(f"警告: 找不到对应的json文件: {json_path}")
                
        return samples
    
    def _cache_labels(self):
        """
        缓存所有样本的标签，用于计算类别权重和统计分布
        在初始化时调用一次，避免重复解析标签
        """
        self.position_labels = []
        self.grade_labels = []
        
        # 遍历所有样本解析标签
        for _, json_path in self.samples:
            position, grade = self._parse_json_label(json_path)
            self.position_labels.append(position)
            self.grade_labels.append(grade)
    
    def get_class_weights(self):
        """
        计算位置和等级分类的类别权重，用于处理类别不平衡问题
        反比于频率的权重，稀有类得到更高权重
        
        返回:
            tuple: (position_weights, grade_weights)
                - position_weights: 位置类别权重，形状为 [3]
                - grade_weights: 等级类别权重，形状为 [5] (用于向后兼容)
        """
        # 计算位置标签分布 - 使用Counter统计每个类别的样本数
        position_counter = Counter(self.position_labels)
        total_position = len(self.position_labels)
        position_weights = []
        
        # 为每个位置类别计算权重 (3个类别)
        for i in range(3):  # 下部/中部/上部 (0/1/2)
            count = position_counter.get(i, 0)
            # 避免除零错误
            if count == 0:
                position_weights.append(1.0)  # 如果没有样本，设置默认权重
            else:
                # 反比于频率的权重 - 频率越低权重越高
                # 乘以类别数，使权重平均值接近1
                position_weights.append(total_position / (count * 3))
        
        # 计算等级标签分布 (5个类别，用于向后兼容)
        grade_counter = Counter(self.grade_labels)
        total_grade = len(self.grade_labels)
        grade_weights = []
        
        for i in range(5):  # 无/轻度/中度/重度/极重度 (0/1/2/3/4)
            count = grade_counter.get(i, 0)
            # 避免除零错误
            if count == 0:
                grade_weights.append(1.0)
            else:
                # 反比于频率的权重
                grade_weights.append(total_grade / (count * 5))
        
        return position_weights, grade_weights
    
    def __len__(self):
        """
        返回数据集中样本数量
        
        返回:
            int: 样本数量
        """
        return len(self.samples)
    
    def _parse_json_label(self, json_path):
        """
        解析JSON标注文件，提取感染部位和感染等级信息
        
        参数:
            json_path (str): JSON标注文件路径
            
        返回:
            tuple: (position_label, grade_label) 
                - position_label: 感染部位的数值标签 (0-2)
                - grade_label: 感染等级的数值标签 (0-4为分类标签，但实际使用0-9的原始值进行回归)
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 从文件名或数据目录提取位置信息(l/m/t)
            # 示例: 如果数据位于目录 "14l" 或文件名包含 "l"，表示下部感染
            file_name = os.path.basename(json_path)
            
            # 通过文件名或路径判断位置类别
            if 'l_' in file_name or '14l' in json_path:
                position = 'l'  # 下部
            elif 'm_' in file_name or '14m' in json_path:
                position = 'm'  # 中部
            elif 't_' in file_name or '14t' in json_path:
                position = 't'  # 上部
            else:
                # 默认为中部
                position = 'm'
            
            # 从JSON标注中提取疾病等级，默认为无疾病(0)
            grade = 0  
            
            # 解析JSON数据，查找是否存在健康标签
            is_healthy = True
            if 'shapes' in data:
                for shape in data['shapes']:
                    if 'label' in shape and shape['label'] != 'health':
                        # 发现非健康标签，说明存在疾病
                        is_healthy = False
                        # 尝试从标签中提取等级信息
                        # 假设标签格式可能包含等级信息，如 "disease_5"
                        label = shape['label']
                        if '_' in label:
                            try:
                                # 尝试提取数字部分作为等级
                                grade_str = label.split('_')[-1]
                                if grade_str.isdigit():
                                    grade = int(grade_str)
                                    # 检查是否为有效等级
                                    if grade in self.grade_map:
                                        break
                                    else:
                                        grade = 5  # 默认为中度
                            except:
                                grade = 5  # 解析失败，设置默认中度
                        else:
                            grade = 5  # 如果没有具体等级，默认为中度
            
            # 如果是健康样本，设置为0级
            if is_healthy:
                grade = 0
            elif grade not in self.grade_map and grade != 0:
                grade = 5  # 默认为中度
            
            # 转换为模型需要的标签
            position_label = self.position_map[position]  # 将文本位置转为数值
            
            # 现在直接返回原始等级值 (0-9) 用于回归任务
            # 但也保留分类标签，用于向后兼容和统计
            grade_label = self.grade_map.get(grade, 2)  # 默认为中度(2)
            
            return position_label, grade_label
            
        except Exception as e:
            print(f"解析JSON标签时出错: {e}")
            # 默认为中部(1)和无疾病(0)
            return 1, 0
    
    def __getitem__(self, idx):
        """
        获取单个样本的数据和标签
        PyTorch数据集的核心方法
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (image, position_label, grade_label)
                - image: 图像张量 [C, H, W]
                - position_label: 感染部位标签 (0-2)
                - grade_label: 感染等级标签 (0-9，用于回归)
        """
        # 获取路径对
        tif_path, json_path = self.samples[idx]
        
        # 解析标签
        position_label, grade_label = self._parse_json_label(json_path)
        
        # 读取.tif多光谱图像
        try:
            with rasterio.open(tif_path) as src:
                # 读取所有波段 - 多光谱/高光谱图像可能有多个波段
                img = src.read()
                
                # 获取通道数
                num_channels = img.shape[0]
                
                # 如果通道数过多（如高光谱图像），选择3个代表性波段
                if num_channels > 10:  # 多于10个通道视为高光谱
                    # 选择具有代表性的3个波段 - 三个不同的光谱区域:
                    # 对于500波段，选择波段索引: 50（可见光）、200（近红外）、400（短波红外）
                    # 这些索引需要根据实际的光谱范围调整
                    selected_bands = [min(50, num_channels-1), 
                                     min(200, num_channels-1), 
                                     min(400, num_channels-1)]
                    
                    # 创建新的3通道图像
                    selected_img = np.zeros((3, img.shape[1], img.shape[2]), dtype=np.float32)
                    for i, band_idx in enumerate(selected_bands):
                        selected_img[i] = img[band_idx]
                    img = selected_img
                    num_channels = 3
                
                # 转换为浮点类型用于标准化
                img = img.astype(np.float32)
                
                # 对每个通道单独标准化到[0,1]范围 - 重要的预处理步骤
                for i in range(num_channels):
                    band = img[i]
                    min_val = np.min(band)
                    max_val = np.max(band)
                    
                    # 避免除零错误 - 如果最大值等于最小值
                    if max_val > min_val:
                        img[i] = (band - min_val) / (max_val - min_val)
                    else:
                        img[i] = np.zeros_like(band)
                
                # 如果通道数小于3，复制现有通道填充到3通道
                # 大多数模型设计用于处理3通道图像
                if num_channels < 3:
                    # 复制现有通道到3个通道
                    temp_img = np.zeros((3, img.shape[1], img.shape[2]), dtype=np.float32)
                    for i in range(3):
                        temp_img[i] = img[min(i, num_channels-1)]
                    img = temp_img
                
                # 调整图像大小到指定尺寸 - 统一尺寸便于批处理
                if img.shape[1] != self.img_size or img.shape[2] != self.img_size:
                    # 使用简单的调整方法
                    resized_img = np.zeros((img.shape[0], self.img_size, self.img_size), dtype=np.float32)
                    for i in range(img.shape[0]):
                        # 使用线性插值，选择等间隔的像素点
                        h_indices = np.linspace(0, img.shape[1]-1, self.img_size).astype(int)
                        w_indices = np.linspace(0, img.shape[2]-1, self.img_size).astype(int)
                        resized_img[i] = img[i][h_indices[:, None], w_indices]
                    img = resized_img
        
        except Exception as e:
            print(f"读取TIF文件时出错: {e}")
            # 创建随机图像作为替代，避免中断训练
            img = np.random.rand(3, self.img_size, self.img_size).astype(np.float32)
        
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img)
        
        # 应用数据增强变换
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # 转换标签为张量
        position_label = torch.tensor(position_label, dtype=torch.long)
        
        # 将等级转换为回归目标 - 从数值类别转为原始等级值(0-9)
        # 这里为了保持向后兼容性，我们将分类标签0-4映射回原始等级值
        if grade_label == 0:  # 无病害
            regression_grade = 0.0
        elif grade_label == 1:  # 轻度
            regression_grade = 3.0
        elif grade_label == 2:  # 中度
            regression_grade = 5.0
        elif grade_label == 3:  # 重度
            regression_grade = 7.0
        elif grade_label == 4:  # 极重度
            regression_grade = 9.0
        else:
            regression_grade = 5.0  # 默认中度
        
        # 转换为浮点张量(回归目标)
        grade_label = torch.tensor(regression_grade, dtype=torch.float)
        
        return img_tensor, position_label, grade_label

class DataAugmentation:
    """
    数据增强类，提供多种图像增强方法
    用于增加训练数据的多样性，提高模型泛化能力
    """
    def __init__(self, aug_prob=0.5):
        """
        初始化数据增强
        
        参数:
            aug_prob (float): 应用每种增强方法的概率，默认0.5
                              较小的值会减少增强程度，较大的值会增加增强强度
        """
        self.aug_prob = aug_prob
    
    def __call__(self, img):
        """
        应用随机数据增强
        
        参数:
            img (Tensor): 输入图像张量，形状为 [C, H, W]
            
        返回:
            Tensor: 增强后的图像，形状不变
        """
        # 随机水平翻转 - 模拟不同视角
        if random.random() < self.aug_prob:
            img = transforms_functional.hflip(img)
            
        # 随机垂直翻转 - 模拟不同视角
        if random.random() < self.aug_prob:
            img = transforms_functional.vflip(img)
            
        # 随机旋转 - 模拟不同角度
        if random.random() < self.aug_prob:
            # 随机选择90度的倍数进行旋转
            angle = random.choice([90, 180, 270])
            img = transforms_functional.rotate(img, angle)
            
        # 随机亮度调整 - 模拟光照变化
        if random.random() < self.aug_prob:
            brightness_factor = random.uniform(0.8, 1.2)  # 亮度变化范围
            img = transforms_functional.adjust_brightness(img, brightness_factor)
            
        # 随机对比度调整 - 模拟成像条件变化
        if random.random() < self.aug_prob:
            contrast_factor = random.uniform(0.8, 1.2)  # 对比度变化范围
            img = transforms_functional.adjust_contrast(img, contrast_factor)
            
        # 随机裁剪然后缩放回原始大小 - 模拟局部观察
        if random.random() < self.aug_prob:
            # 获取图像尺寸
            _, h, w = img.shape
            # 随机裁剪范围(80%-100%)
            crop_ratio = random.uniform(0.8, 1.0)
            new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
            # 计算裁剪区域的左上角坐标
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            # 裁剪图像
            img = transforms_functional.crop(img, top, left, new_h, new_w)
            # 调整回原始大小
            img = transforms_functional.resize(img, (h, w))
            
        return img

def get_dataloaders(data_root, json_root=None, batch_size=32, num_workers=4, img_size=128, train_ratio=0.8, aug_prob=0.5):
    """
    创建训练和验证数据加载器
    
    参数:
        data_root (str): 数据根目录，应包含14l、14m、14t子目录
        json_root (str, optional): JSON标注根目录，如果为None，则使用data_root + '_json'
        batch_size (int): 批次大小，影响内存使用和更新频率
        num_workers (int): 数据加载线程数，加速数据加载
        img_size (int): 图像统一调整大小
        train_ratio (float): 训练集比例，范围0-1
        aug_prob (float): 数据增强应用概率
        
    返回:
        tuple: (train_loader, val_loader) - 训练和验证数据加载器
    """
    datasets = []
    
    # 加载每个位置类别的数据集
    for pos in ['14l', '14m', '14t']:  # 下部/中部/上部
        # 构建路径
        data_dir = os.path.join(data_root, f'mini_{pos}')
        json_dir = None if json_root is None else os.path.join(json_root, f'mini_{pos}_json')
        
        # 检查数据目录是否存在
        if os.path.exists(data_dir):
            # 创建各个位置的数据集
            dataset = CornRustDataset(data_dir, json_dir, img_size=img_size)
            datasets.append(dataset)
        else:
            print(f"警告: 数据目录不存在: {data_dir}")
    
    # 合并所有数据集
    if not datasets:
        raise ValueError("未找到有效的数据集，请检查数据路径")
    
    # 处理单个或多个数据集的情况
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        # 合并多个位置的数据集
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(datasets)
    
    # 分割训练集和验证集
    total_size = len(combined_dataset)
    train_size = int(total_size * train_ratio)  # 训练集大小
    val_size = total_size - train_size  # 验证集大小
    
    # 创建数据增强变换实例
    data_augmentation = DataAugmentation(aug_prob=aug_prob)
    
    # 使用随机分割创建训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子以确保可重复性
    )
    
    # 为训练集添加数据增强
    # 注意：需要访问原始数据集的transform属性
    train_dataset.dataset.transform = data_augmentation
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 打乱训练数据
        num_workers=num_workers,  # 多线程加载
        pin_memory=True  # 提高GPU数据传输速度
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader