# 模型定义文件：包含玉米南方锈病识别的各种神经网络模型定义，包括简单CNN、ResNet和带注意力机制的增强ResNet，实现多任务学习（同时预测感染部位和感染等级）
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiseaseClassifier(nn.Module):
    """
    双头CNN模型，用于玉米南方锈病多任务分类：
    1. 感染部位: 下部/中部/上部 (3分类)
    2. 感染等级: 无/轻度/中度/重度/极重度 (5分类)
    """
    def __init__(self, in_channels=3, img_size=128):
        """
        初始化双头CNN分类器
        
        参数:
            in_channels: 输入图像的通道数，默认为3（RGB）
            img_size: 输入图像的尺寸，默认为128x128
        """
        super(DiseaseClassifier, self).__init__()
        self.in_channels = in_channels
        
        # 共享特征提取层 - 使用三层卷积网络提取图像特征
        self.features = nn.Sequential(
            # 第一个卷积块 - 输入通道 -> 32通道，使用3x3卷积核和2x2最大池化
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 保持空间维度不变
            nn.BatchNorm2d(32),  # 批归一化加速训练并提高稳定性
            nn.ReLU(inplace=True),  # 使用ReLU激活函数引入非线性
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样，减小特征图尺寸
            
            # 第二个卷积块 - 32通道 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块 - 64通道 -> 128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算卷积后的特征图大小 - 经过三次2x2最大池化，尺寸变为原来的1/8
        conv_output_size = img_size // 8  # 三次下采样 (2^3=8)
        self.fc_input_size = 128 * conv_output_size * conv_output_size
        
        # 位置分类头 (3分类) - 将特征向量映射到3个类别
        self.position_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(self.fc_input_size, 256),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 第二个Dropout层进一步防止过拟合
            nn.Linear(256, 3)  # 输出3个类别的logits: 下部/中部/上部
        )
        
        # 等级分类头 (5分类) - 将特征向量映射到5个类别
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为一维向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(self.fc_input_size, 256),  # 全连接层降维
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # 输出5个类别的logits: 无/轻度/中度/重度/极重度
        )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入图像张量, 形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (position_logits, grade_logits)
                - position_logits: 位置分类的logits，形状为 [batch_size, 3]
                - grade_logits: 等级分类的logits，形状为 [batch_size, 5]
        """
        # 共享特征提取 - 对输入图像提取特征
        features = self.features(x)
        
        # 位置分类 - 通过位置分类头预测感染部位
        position_logits = self.position_classifier(features)
        
        # 等级分类 - 通过等级分类头预测感染等级
        grade_logits = self.grade_classifier(features)
        
        return position_logits, grade_logits

class ResidualBlock(nn.Module):
    """
    ResNet的基本残差块
    残差连接允许梯度直接流过网络，缓解深度网络的梯度消失问题
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 第一个卷积层的步长，用于下采样，默认为1
        """
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，可能用于下采样（当stride > 1时）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层，保持空间维度不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，需要调整残差连接 - 使用1x1卷积进行通道调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, in_channels, height, width]
            
        返回:
            out: 残差块输出，形状为 [batch_size, out_channels, height/stride, width/stride]
        """
        residual = x  # 保存输入作为残差连接
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加残差连接，实现跳跃连接
        out += self.shortcut(residual)
        out = F.relu(out)  # 最后的非线性激活
        
        return out

class DiseaseResNet(nn.Module):
    """
    基于ResNet结构的双头模型，用于玉米南方锈病多任务分类
    使用残差连接和更深的网络结构增强特征提取能力
    """
    def __init__(self, in_channels=3, img_size=128):
        """
        初始化疾病ResNet模型
        
        参数:
            in_channels: 输入图像的通道数，默认为3
            img_size: 输入图像的尺寸，默认为128x128
        """
        super(DiseaseResNet, self).__init__()
        self.in_channels = in_channels
        
        # 初始卷积层 - 7x7大卷积核提取低级特征
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样
        
        # ResNet块 - 构建三层残差网络
        self.layer1 = self._make_layer(64, 64, 2)  # 第一层：64->64通道，2个残差块
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 第二层：64->128通道，2个残差块，下采样
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 第三层：128->256通道，2个残差块，下采样
        
        # 计算卷积后的特征图大小
        # 图像尺寸经过初始卷积和MaxPool后变为 img_size/4
        # 再经过三个ResNet layer (其中两层有stride=2)，变为 img_size/16
        conv_output_size = img_size // 16
        self.fc_input_size = 256 * conv_output_size * conv_output_size
        
        # 全局平均池化 - 降低参数量并保留空间特征
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出1x1特征图
        
        # 位置分类头 - 预测感染部位
        self.position_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平
            nn.Dropout(0.5),  # 减少过拟合
            nn.Linear(256, 3)  # 全连接层输出3个类别
        )
        
        # 等级分类头 - 预测感染等级
        self.grade_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)  # 全连接层输出5个类别
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        创建残差层，包含多个残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 第一个残差块的步长，用于下采样
            
        返回:
            nn.Sequential: 包含多个残差块的顺序容器
        """
        layers = []
        # 第一个block可能需要调整维度
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # 剩余blocks保持维度不变
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入图像张量, 形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (position_logits, grade_logits)
                - position_logits: 位置分类的logits，形状为 [batch_size, 3]
                - grade_logits: 等级分类的logits，形状为 [batch_size, 5]
        """
        # 特征提取
        x = self.conv1(x)  # 初始卷积
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 最大池化
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.avgpool(x)  # 转为1x1特征图
        
        # 位置分类
        position_logits = self.position_classifier(x)
        
        # 等级分类
        grade_logits = self.grade_classifier(x)
        
        return position_logits, grade_logits

# 新增注意力机制模块
class ChannelAttention(nn.Module):
    """
    通道注意力机制
    捕捉通道之间的依赖关系，对重要的通道赋予更高的权重
    结合平均池化和最大池化的信息，提高特征表示能力
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化通道注意力模块
        
        参数:
            in_channels: 输入特征图的通道数
            reduction_ratio: 降维比例，用于减少参数量
        """
        super(ChannelAttention, self).__init__()
        # 全局平均池化 - 捕获通道的全局分布
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出1x1特征图
        # 全局最大池化 - 捕获通道的显著特征
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通过两个1x1卷积实现全连接层，减少参数量
        self.fc = nn.Sequential(
            # 第一个1x1卷积，降维
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            # 第二个1x1卷积，恢复维度
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        # sigmoid激活函数，将注意力权重归一化到[0,1]范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, in_channels, height, width]
            
        返回:
            attention: 通道注意力权重，形状为 [batch_size, in_channels, 1, 1]
        """
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x))
        # 融合两个分支的信息
        out = avg_out + max_out
        # 应用sigmoid归一化
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    空间注意力机制
    关注图像的空间位置重要性，对重要区域赋予更高权重
    结合通道平均值和最大值的信息，增强模型对空间区域的感知能力
    """
    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块
        
        参数:
            kernel_size: 卷积核大小，默认为7，用于捕获更大的感受野
        """
        super(SpatialAttention, self).__init__()
        # 使用单层卷积学习空间注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()  # 注意力权重归一化

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, channels, height, width]
            
        返回:
            attention: 空间注意力权重，形状为 [batch_size, 1, height, width]
        """
        # 沿通道维度计算平均值 - 捕获全局通道信息
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维度计算最大值 - 捕获显著特征
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接通道平均值和最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 形状为 [batch_size, 2, height, width]
        # 通过卷积生成空间注意力图
        x = self.conv(x)  # 输出单通道特征图
        # 应用sigmoid归一化
        return self.sigmoid(x)

# 新增带注意力机制的残差块
class AttentionResidualBlock(nn.Module):
    """
    带注意力机制的残差块
    在基本残差块基础上增加了通道注意力和空间注意力机制
    结合CBAM(Convolutional Block Attention Module)思想，串联使用通道和空间注意力
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化带注意力机制的残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 第一个卷积层的步长，默认为1
        """
        super(AttentionResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 注意力模块 - 分别实现通道和空间注意力
        self.ca = ChannelAttention(out_channels)  # 通道注意力
        self.sa = SpatialAttention()  # 空间注意力
        
        # 如果输入输出通道数不同，需要调整残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图，形状为 [batch_size, in_channels, height, width]
            
        返回:
            out: 残差块输出，形状为 [batch_size, out_channels, height/stride, width/stride]
        """
        residual = x  # 保存输入作为残差连接
        
        # 常规残差块前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用通道注意力 - 增强重要通道特征
        out = self.ca(out) * out
        # 应用空间注意力 - 突出重要空间区域
        out = self.sa(out) * out
        
        # 添加残差连接并激活
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

# 新增改进的ResNet模型
class DiseaseResNetPlus(nn.Module):
    """
    改进版ResNet模型，带有注意力机制和更深的网络结构
    使用多任务学习架构，同时预测感染部位和感染等级
    修改为感染位置分类 + 等级回归的双头网络
    """
    def __init__(self, in_channels=3, img_size=128):
        """
        初始化改进的ResNet模型
        
        参数:
            in_channels: 输入图像的通道数，默认为3
            img_size: 输入图像的尺寸，默认为128x128
        """
        super(DiseaseResNetPlus, self).__init__()
        self.in_channels = in_channels
        
        # 初始卷积层 - 大尺寸卷积核提取更多低级特征
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 带注意力机制的ResNet块 - 比标准ResNet更深
        self.layer1 = self._make_layer(64, 64, 2)  # 第一层：64->64通道，2个残差块
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 第二层：64->128通道，2个残差块，下采样
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 第三层：128->256通道，2个残差块，下采样
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 第四层：256->512通道，2个残差块，下采样
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 任务特定的通道注意力层 - 为不同任务提供专注的特征
        self.position_attention = ChannelAttention(512, reduction_ratio=16)  # 位置任务的注意力
        self.grade_attention = ChannelAttention(512, reduction_ratio=16)  # 等级任务的注意力
        
        # 位置分类头 - 保持分类头设计，加入Dropout和BatchNorm增强泛化能力
        self.position_classifier = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(512, 256),  # 全连接层降维
            nn.BatchNorm1d(256),  # 添加BatchNorm提高训练稳定性
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Dropout(0.3),  # 再次应用Dropout
            nn.Linear(256, 3)  # 最后的全连接层，输出3个类别的logits：下部/中部/上部
        )
        
        # 等级回归头 - 将分类头改为回归头，预测0-9的连续等级值
        self.grade_regressor = nn.Sequential(
            nn.Flatten(),  # 将特征图展平为向量
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(512, 256),  # 全连接层降维
            nn.BatchNorm1d(256),  # 添加BatchNorm提高训练稳定性
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Dropout(0.3),  # 再次应用Dropout
            nn.Linear(256, 1)  # 最后的全连接层，输出单个连续值作为等级预测
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        创建包含多个注意力残差块的层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 第一个残差块的步长，用于下采样
            
        返回:
            nn.Sequential: 包含多个注意力残差块的顺序容器
        """
        layers = []
        # 第一个block可能需要调整维度，使用带步长的卷积进行下采样
        layers.append(AttentionResidualBlock(in_channels, out_channels, stride))
        # 剩余blocks保持维度不变
        for _ in range(1, blocks):
            layers.append(AttentionResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入图像张量, 形状为 [batch_size, in_channels, height, width]
            
        返回:
            tuple: (position_logits, grade_value)
                - position_logits: 位置分类的logits，形状为 [batch_size, 3]
                - grade_value: 等级回归值，形状为 [batch_size, 1]，范围0-9
        """
        # 共享特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 通过残差层提取层次化特征
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化 - 将特征图压缩为1x1
        x = self.avgpool(x)
        
        # 使用任务特定的注意力机制 - 增强每个任务相关的特征
        position_features = self.position_attention(x) * x  # 位置分类的特征增强
        grade_features = self.grade_attention(x) * x  # 等级预测的特征增强
        
        # 位置分类 - 输出logits供softmax分类
        position_logits = self.position_classifier(position_features)
        
        # 等级回归 - 输出单值回归结果，表示0-9之间的等级
        grade_value = self.grade_regressor(grade_features)
        
        return position_logits, grade_value

def get_model(model_type='simple', in_channels=3, img_size=128):
    """
    获取模型实例
    
    参数:
        model_type: 模型类型，选项为:
            - 'simple': 简单的CNN双头模型
            - 'resnet': 基于ResNet的双头模型
            - 'resnet_plus': 带注意力机制的改进ResNet模型(多任务学习)
        in_channels: 输入通道数
        img_size: 输入图像尺寸
        
    返回:
        model: 模型实例
        
    异常:
        ValueError: 当model_type不是支持的类型时抛出
    """
    if model_type == 'simple':
        return DiseaseClassifier(in_channels=in_channels, img_size=img_size)
    elif model_type == 'resnet':
        return DiseaseResNet(in_channels=in_channels, img_size=img_size)
    elif model_type == 'resnet_plus':
        return DiseaseResNetPlus(in_channels=in_channels, img_size=img_size)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")