"""
抗恶劣环境行人重识别系统 - 骨干网络模块
支持多种专用ReID网络：OSNet、AGW、ResNet-IBN等

使用方法：
    from backbone import build_backbone, SUPPORTED_BACKBONES
    model = build_backbone('osnet', num_classes=751)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from torchvision.models import resnet50, resnet34, ResNet50_Weights, ResNet34_Weights


def _load_local_resnet50_weights(checkpoint_path='checkpoints/market1501_pretrained.pth', device='cpu'):
    """
    从本地训练好的 checkpoint 中提取 ResNet50 backbone 权重 (conv1~layer4)。
    当 ImageNet 预训练权重下载失败时作为高质量替代方案。
    """
    if checkpoint_path is None:
        return None
    p = Path(checkpoint_path)
    if not p.is_absolute():
        # backbone.py 在 src/ 目录下，项目根目录是其父目录
        p = Path(__file__).parent.parent / p
    if not p.exists():
        return None
    try:
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        resnet_dict = {}
        for k, v in state_dict.items():
            # 只提取 ResNet50 backbone 部分：conv1, bn1, layer1~layer4
            if k.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')):
                resnet_dict[k] = v
        if len(resnet_dict) >= 10:
            print(f"  [NNLRNet] 从本地 {p} 提取 ResNet50 backbone 权重 ({len(resnet_dict)} 层)")
            return resnet_dict
    except Exception as e:
        print(f"  [NNLRNet] 读取本地 checkpoint 失败: {e}")
    return None


def _load_resnet50(pretrained=True, local_checkpoint='checkpoints/market1501_pretrained.pth'):
    """安全加载 ResNet50，优先使用本地预训练权重，下载失败时自动回退"""
    if not pretrained:
        return resnet50(weights=None)
    # 优先从本地已训练的 Market-1501 checkpoint 加载 backbone，避免网络下载失败
    local_weights = _load_local_resnet50_weights(local_checkpoint)
    if local_weights is not None:
        try:
            model = resnet50(weights=None)
            model.load_state_dict(local_weights, strict=False)
            print("  [NNLRNet] 成功加载本地预训练 ResNet50 backbone")
            return model
        except Exception as e:
            print(f"  [NNLRNet] 本地权重加载失败: {e}")
    try:
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"  [警告] ResNet50 预训练权重下载失败: {e}")
        print(f"  [警告] 回退到无预训练权重（需更多训练轮数补偿）")
        return resnet50(weights=None)



# ============== OSNet (Omni-Scale Network) ==============
class OSNet(nn.Module):
    """OSNet: Omni-Scale Network for Person Re-ID
    专门为ReID设计的轻量级网络，具有多尺度特征提取能力
    轻量级实现版本
    """
    
    def __init__(self, num_classes=751, loss='softmax', pretrained=True):
        super(OSNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512
        
        # OSNet 轻量级架构
        # conv1: 3 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # conv2: 64 -> 256
        self.conv2 = self._make_layer(64, 256, blocks=2, stride=2)
        
        # conv3: 256 -> 512
        self.conv3 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # conv4: 512 -> 512
        self.conv4 = self._make_layer(512, 512, blocks=2, stride=1)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
        
        self._init_params()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """构建OSNet风格的层"""
        layers = []
        
        # 第一个块可能有下采样
        layers.append(OSNetBlock(in_channels, out_channels, stride=stride))
        
        # 剩余的块
        for _ in range(1, blocks):
            layers.append(OSNetBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        global_feat = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        logits = self.classifier(global_feat)
        
        return logits, global_feat


class OSNetBlock(nn.Module):
    """OSNet 基本块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(OSNetBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第三个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


# ============== AGW (Aggregated Gaussian Residual) ==============
class AGWBackbone(nn.Module):
    """AGW: Deep Learning for Person Re-Identification with Sensitive Image Degradation
    
    基于ResNet50的改进版本，添加了：
    1. Non-local注意力机制
    2. 改进的残差连接
    3. 梯度增强
    """
    
    def __init__(self, num_classes=751, pretrained=True):
        super(AGWBackbone, self).__init__()
        
        # 基础 ResNet50
        base = _load_resnet50(pretrained)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        # Non-local 注意力模块
        self.non_local = NonLocalBlock(2048)
        
        # 特征维度
        self.feature_dim = 2048
        
        # 池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool_geo = nn.AdaptiveMaxPool2d(1)
        
        # 分类器 (avg + max = 2048 * 2 = 4096)
        self.classifier = nn.Linear(4096, num_classes)
        
        # BNNeck
        self.bnneck = nn.BatchNorm1d(4096)
        
        self._init_params()
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Non-local attention
        x = self.non_local(x)
        
        # 双池化
        avg_feat = self.avgpool(x).squeeze(-1).squeeze(-1)
        max_feat = self.maxpool_geo(x).squeeze(-1).squeeze(-1)
        
        # 特征拼接
        feat = torch.cat([avg_feat, max_feat], dim=1)
        
        # BNNeck
        feat_bn = self.bnneck(feat)
        
        logits = self.classifier(feat_bn)
        
        return logits, feat


class NonLocalBlock(nn.Module):
    """Non-local 注意力模块"""
    def __init__(self, in_channels, reduction=2):
        super(NonLocalBlock, self).__init__()
        self.inter_channels = in_channels // reduction
        
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)
        
        w_y = self.W(y)
        return x + w_y


# ============== ResNet-IBN ==============
class IBN(nn.Module):
    """Instance Batch Normalization"""
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat([out1, out2], dim=1)


class ResNetIBN(nn.Module):
    """ResNet with IBN layers for better appearance modeling"""
    
    def __init__(self, num_classes=751, pretrained=True):
        super(ResNetIBN, self).__init__()
        
        # 使用基础ResNet结构，自己添加IBN
        base = _load_resnet50(pretrained)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        # Layer1: 使用 IBN
        self.layer1 = self._make_layer_ibn(64, 256, blocks=3, stride=1)
        # Layer2-4: 保持原样
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.feature_dim = 2048
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)
        self.bottleneck = nn.BatchNorm1d(2048)
        
        self._init_params()
    
    def _make_layer_ibn(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        
        in_channels = out_channels
        for i in range(blocks):
            layers.append(self._make_block_ibn(in_channels, out_channels, stride=1 if i == 0 else 1))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _make_block_ibn(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = [
            IBN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        
        return nn.Sequential(*layers)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        feat = self.bottleneck(x)
        logits = self.classifier(feat)
        
        return logits, feat


# ============== PCB (Part-based Convolutional Baseline) ==============
class PCB(nn.Module):
    """PCB: Part-based Convolutional Baseline for Re-ID
    
    将特征图水平分成多个部分，分别进行分类
    正确的实现：在空间维度上划分，而不是通道维度
    """
    
    def __init__(self, num_classes=751, num_parts=6, pretrained=True):
        super(PCB, self).__init__()
        self.num_parts = num_parts
        
        base = _load_resnet50(pretrained)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.feature_dim = 2048
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 全局分类器
        self.global_classifier = nn.Linear(2048, num_classes)
        
        # 每个part的分类器
        self.part_classifiers = nn.ModuleList([
            nn.Linear(2048, num_classes) for _ in range(num_parts)
        ])
    
    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, H, W]
        
        # 全局特征
        global_feat = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, 2048]
        
        # 水平分成多个部分 (在H维度上划分)
        B, C, H, W = x.shape
        h_part = H // self.num_parts  # 每个part的高度
        
        part_logits = []
        part_features = []
        for i in range(self.num_parts):
            # 提取第i个水平条带的特征
            start_h = i * h_part
            end_h = start_h + h_part if i < self.num_parts - 1 else H
            part = x[:, :, start_h:end_h, :]  # [B, 2048, h_part, W]
            
            # 对每个part进行平均池化
            part_feat = nn.functional.adaptive_avg_pool2d(part, 1).squeeze(-1).squeeze(-1)  # [B, 2048]
            part_features.append(part_feat)
            part_logits.append(self.part_classifiers[i](part_feat))
        
        # 合并part特征作为最终特征
        concat_feat = torch.cat(part_features, dim=1)  # [B, 2048 * num_parts]
        
        # 使用全局分类器
        logits = self.global_classifier(global_feat)
        
        # 训练时返回多个part的logits用于多损失，推理时返回全局logits
        if self.training:
            return part_logits, concat_feat
        else:
            return logits, concat_feat[:, :2048]  # 推理时只返回全局特征


# ============== NNLR-Net 三分支架构 (借鉴车辆ReID技术) ==============
class SLRM(nn.Module):
    """空间局部细化模块 (Spatial Local Refinement Module)
    利用像素最近邻相关性细化局部特征
    来源: 车辆重识别技术方案中的NNLR-Net
    """
    def __init__(self, in_channels, top_k=8):
        super(SLRM, self).__init__()
        self.in_channels = in_channels
        self.top_k = top_k
        # 用于计算特征相似度的1x1卷积
        self.query_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.shape
        N = H * W  # 空间位置数
        
        # 降维到低维空间计算相似度
        query = self.query_conv(x).view(batch, -1, N).permute(0, 2, 1)  # [B, N, C//4]
        key = self.key_conv(x).view(batch, -1, N)  # [B, C//4, N]
        
        # 计算所有像素对之间的相似度
        affinity = torch.bmm(query, key)  # [B, N, N]
        affinity = F.softmax(affinity, dim=-1)
        
        # 取每个像素的top-k最近邻
        topk_vals, topk_idx = torch.topk(affinity, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        # 聚合最近邻特征
        value = self.value_conv(x).view(batch, C, -1)  # [B, C, N]
        # 用scatter实现top-k聚合
        refined = torch.zeros_like(value)
        for b in range(batch):
            for i in range(N):
                neighbors = topk_idx[b, i]  # [top_k]
                weights = topk_weights[b, i]  # [top_k]
                for k in range(self.top_k):
                    refined[b, :, i] += weights[k] * value[b, :, neighbors[k]]
        
        refined = refined.view(batch, C, H, W)
        out = x + self.gamma * refined
        return out


class CLRM(nn.Module):
    """通道局部细化模块 (Channel Local Refinement Module)
    利用通道最近邻相关性细化特征
    来源: 车辆重识别技术方案中的NNLR-Net
    """
    def __init__(self, in_channels, top_k=8):
        super(CLRM, self).__init__()
        self.in_channels = in_channels
        self.top_k = top_k
        # 通道降维
        self.reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(in_channels // 4, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.shape
        
        # 获取通道描述符
        desc = self.reduction(x).view(batch, -1)  # [B, C//4]
        channel_weight = torch.sigmoid(self.fc(desc))  # [B, C]
        
        # 计算通道间相似度
        x_flat = x.view(batch, C, -1)  # [B, C, H*W]
        x_norm = F.normalize(x_flat, p=2, dim=2)
        channel_sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, C, C]
        
        # 取每个通道的top-k最近邻
        topk_vals, topk_idx = torch.topk(channel_sim, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        # 聚合最近邻通道
        refined = torch.zeros_like(x_flat)
        for b in range(batch):
            for c in range(C):
                neighbors = topk_idx[b, c]
                weights = topk_weights[b, c]
                for k in range(self.top_k):
                    refined[b, c, :] += weights[k] * x_flat[b, neighbors[k], :]
        
        refined = refined.view(batch, C, H, W)
        out = x + self.gamma * refined * channel_weight.view(batch, C, 1, 1)
        return out


class NNLRBranch(nn.Module):
    """NNLR-Net 中的单个分支（全局/局部）
    每个分支：res_conv5 → [可选SLRM/CLRM] → GAP → 1×1conv → BN → GELU → 256D
    """
    def __init__(self, backbone_layer, refine_module=None):
        super(NNLRBranch, self).__init__()
        self.backbone_layer = backbone_layer  # res_conv5
        self.refine = refine_module
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.backbone_layer(x)
        if self.refine is not None:
            x = self.refine(x)
        feat = self.gap(x).squeeze(-1).squeeze(-1)
        return feat


class NNLRNet(nn.Module):
    """NNLR-Net: 三分支局部细化网络
    来源于车辆重识别技术方案，适配行人ReID
    共享ResNet-50 backbone到conv4_x，之后分为三个分支：
      - Global分支: 标准全局特征
      - Local_1分支: 空间局部细化 (SLRM)
      - Local_2分支: 通道局部细化 (CLRM)
    每个分支输出256D，测试时拼接为768D
    """
    def __init__(self, num_classes=751, pretrained=True):
        super(NNLRNet, self).__init__()
        
        # 加载预训练ResNet-50（优先本地已训练权重，失败则回退ImageNet/随机）
        if pretrained:
            base = _load_resnet50(pretrained=True, local_checkpoint='checkpoints/market1501_pretrained.pth')
        else:
            base = resnet50(weights=None)

        
        # 共享backbone: 到conv4_x为止
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # 共享的最后一层
        
        in_features = 2048  # ResNet-50 conv5输出通道
        
        # BNNeck: 768D (3个分支各256D)
        self.bnneck = nn.BatchNorm1d(768)
        self.bnneck.bias.requires_grad_(False)
        
        # 三个分支的降维层（原来在forward中懒初始化，改为__init__中直接创建）
        self.branch_g = nn.Linear(2048, 256)   # global
        self.branch_l1 = nn.Linear(2048, 256)  # local1
        self.branch_l2 = nn.Linear(2048, 256)  # local2
        
        # 每个分支的分类器
        self.classifier_global = nn.Linear(256, num_classes)
        self.classifier_local1 = nn.Linear(256, num_classes)
        self.classifier_local2 = nn.Linear(256, num_classes)
        
        # Dropout 防过拟合（小底库加大丢率）
        self.dropout = nn.Dropout(0.5)
        
        # 暴露特征维度供外部使用（CircleLoss等需要）
        self.feature_dim = 768
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 共享 backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 三个分支独立处理
        # （简化实现：使用不同的1x1卷积模拟分支差异）
        # 实际应用中应在此处插入SLRM/CLRM模块
        
        # 全局平均池化 → [B, 2048, 1, 1]
        x_pooled = F.adaptive_avg_pool2d(x, 1)
        
        # 展平: [B, 2048]
        x_vec = x_pooled.squeeze(-1).squeeze(-1)
        
        # 三个独立分支: 2048 → 256D
        feat_g = self.branch_g(x_vec)   # global
        feat_l1 = self.branch_l1(x_vec)  # local1 (简化实现，实际应插入SLRM)
        feat_l2 = self.branch_l2(x_vec)  # local2 (简化实现，实际应插入CLRM)
        
        # 合并 → [B, 768]
        feat_concat = torch.cat([feat_g, feat_l1, feat_l2], dim=1)
        
        # BNNeck
        feat_bn = self.bnneck(feat_concat)
        
        # Dropout（仅训练时生效）
        feat_g = self.dropout(feat_g)
        feat_l1 = self.dropout(feat_l1)
        feat_l2 = self.dropout(feat_l2)
        
        # 三个分支的logits
        logits = self.classifier_global(feat_g)
        logits_l1 = self.classifier_local1(feat_l1)
        logits_l2 = self.classifier_local2(feat_l2)
        
        if self.training:
            return logits, logits_l1, logits_l2, feat_concat
        else:
            return logits, feat_concat

# ============== 模型工厂 ==============
SUPPORTED_BACKBONES = {
    'resnet50': 'ResNet50 (ImageNet预训练)',
    'resnet34': 'ResNet34 (ImageNet预训练)',
    'osnet': 'OSNet (专用轻量级ReID网络)',
    'agw': 'AGW (带Non-local注意力的ResNet)',
    'resnet_ibn': 'ResNet-IBN (带Instance Normalization)',
    'pcb': 'PCB (Part-based卷积基线)',
    'nnlrnet': '三分支NNLR-Net (借鉴车辆ReID局部细化技术)',
}


def build_backbone(name, num_classes=751, loss='softmax', pretrained=True):
    """构建骨干网络
    
    Args:
        name: 网络名称
        num_classes: 分类类别数
        loss: 损失类型
        pretrained: 是否使用预训练权重
    
    Returns:
        model: 骨干网络模型
    """
    name = name.lower()
    
    if name == 'resnet50':
        model = ResNet50Backbone(num_classes, pretrained)
    elif name == 'resnet34':
        model = ResNet34Backbone(num_classes, pretrained)
    elif name == 'osnet':
        model = OSNet(num_classes, loss, pretrained)
    elif name == 'agw':
        model = AGWBackbone(num_classes, pretrained)
    elif name == 'resnet_ibn':
        model = ResNetIBN(num_classes, pretrained)
    elif name == 'pcb':
        model = PCB(num_classes, pretrained=pretrained)
    elif name == 'nnlrnet':
        model = NNLRNet(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的骨干网络: {name}. 可选: {list(SUPPORTED_BACKBONES.keys())}")
    
    return model


class ResNet50Backbone(nn.Module):
    """标准 ResNet50 骨干网络"""
    
    def __init__(self, num_classes=751, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        
        base = _load_resnet50(pretrained)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.feature_dim = 2048
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)
        self.bottleneck = nn.BatchNorm1d(2048)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        feat = self.bottleneck(x)
        # 推理时对特征做 L2 归一化，使余弦相似度更有区分度
        if not self.training:
            feat = nn.functional.normalize(feat, p=2, dim=1)
        logits = self.classifier(feat)
        
        return logits, feat


class ResNet34Backbone(nn.Module):
    """轻量级 ResNet34 骨干网络"""
    
    def __init__(self, num_classes=751, pretrained=True):
        super(ResNet34Backbone, self).__init__()
        
        base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 从 resnet34 正确提取各层
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1  # 64 -> 64
        self.layer2 = base.layer2  # 64 -> 128
        self.layer3 = base.layer3  # 128 -> 256
        self.layer4 = base.layer4  # 256 -> 512
        
        self.feature_dim = 512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self.bottleneck = nn.BatchNorm1d(512)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        feat = self.bottleneck(x)
        # 推理时对特征做 L2 归一化，使余弦相似度更有区分度
        if not self.training:
            feat = nn.functional.normalize(feat, p=2, dim=1)
        logits = self.classifier(feat)
        
        return logits, feat


if __name__ == '__main__':
    # 测试各种骨干网络
    print("支持的骨干网络:")
    for name, desc in SUPPORTED_BACKBONES.items():
        print(f"  {name}: {desc}")
    
    print("\n测试 OSNet:")
    model = build_backbone('osnet', num_classes=751)
    x = torch.randn(2, 3, 256, 128)
    logits, feat = model(x)
    print(f"  Logits shape: {logits.shape if hasattr(logits, 'shape') else 'list of tensors'}")
    print(f"  Feature shape: {feat.shape}")
    
    print("\n测试 ResNet-IBN:")
    model = build_backbone('resnet_ibn', num_classes=751)
    logits, feat = model(x)
    print(f"  Feature shape: {feat.shape}")
    
    print("\n所有骨干网络测试完成!")
