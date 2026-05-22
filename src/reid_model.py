"""
抗恶劣环境行人重识别模型
支持可见光-红外双模态融合
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from backbone import build_backbone, SUPPORTED_BACKBONES

class ReIDModel(nn.Module):
    """行人重识别模型 - 支持双模态输入"""
    
    def __init__(self, num_classes=1000, feat_dim=512, modality='both'):
        """
        初始化ReID模型
        Args:
            num_classes: 训练时的身份数量（推理时可忽略）
            feat_dim: 特征维度
            modality: 'visible', 'infrared', 'both'
        """
        super(ReIDModel, self).__init__()
        self.modality = modality
        
        # 可见光分支
        if modality in ['visible', 'both']:
            self.visible_backbone = models.resnet50(pretrained=False)
            self.visible_backbone = nn.Sequential(*list(self.visible_backbone.children())[:-2])
            self.visible_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 红外分支
        if modality in ['infrared', 'both']:
            self.infra_backbone = models.resnet50(pretrained=False)
            self.infra_backbone = nn.Sequential(*list(self.infra_backbone.children())[:-2])
            self.infra_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征融合/投影层
        self.feat_fc = nn.Linear(2048, feat_dim)  # 单模态投影
        if modality == 'both':
            self.fusion_fc = nn.Linear(4096, feat_dim)  # 双模态融合 2048*2=4096
        
        # 特征归一化
        self.bn = nn.BatchNorm1d(feat_dim)
        
        # 分类头（训练时使用）
        self.classifier = nn.Linear(feat_dim, num_classes)
        
        self.feat_dim = feat_dim
        
    def forward(self, x_visible=None, x_infrared=None, return_feat=True):
        """
        前向传播
        Args:
            x_visible: 可见光图像 [B, 3, H, W]
            x_infrared: 红外图像 [B, 3, H, W] (单通道会复制为3通道)
            return_feat: 是否返回特征（推理时True，训练时False）
        """
        features = []
        
        # 提取可见光特征
        if x_visible is not None:
            feat_v = self.visible_backbone(x_visible)
            feat_v = self.visible_pool(feat_v)
            feat_v = feat_v.view(feat_v.size(0), -1)
            features.append(feat_v)
        
        # 提取红外特征
        if x_infrared is not None:
            feat_i = self.infra_backbone(x_infrared)
            feat_i = self.infra_pool(feat_i)
            feat_i = feat_i.view(feat_i.size(0), -1)
            features.append(feat_i)
        
        # 特征融合
        if len(features) == 2:
            fused = torch.cat(features, dim=1)
            feat = self.fusion_fc(fused)
        elif len(features) == 1:
            feat = self.feat_fc(features[0])
        else:
            raise ValueError("必须提供至少一个模态的输入")
        
        # 归一化
        feat = self.bn(feat)
        feat = F.normalize(feat, p=2, dim=1)
        
        if return_feat:
            return feat
        else:
            logits = self.classifier(feat)
            return logits, feat


class ImagePreprocessor:
    """图像预处理器 - 适配恶劣环境"""
    
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size
        
        # 标准预处理 - 对齐 ReidInferencer 的 transform (256x128 resize + 224x112 center crop)
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.CenterCrop((224, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 恶劣环境增强预处理
        self.harsh_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.CenterCrop((224, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image, is_infrared=False, enhance_harsh=False):
        """
        预处理图像
        Args:
            image: PIL Image 或 numpy array
            is_infrared: 是否为红外图像
            enhance_harsh: 是否使用恶劣环境增强
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # 灰度图
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # 红外图像处理：将单通道转为3通道
        if is_infrared and image.mode == 'L':
            image = Image.merge('RGB', (image, image, image))
        
        transform = self.harsh_transform if enhance_harsh else self.transform
        return transform(image).unsqueeze(0)


class ReIDSystem:
    """完整的ReID系统（基于build_backbone，正确加载训练好的模型权重）"""
    
    def __init__(self, model_path=None, device='cpu', backbone='resnet50', num_classes=751):
        """
        Args:
            model_path: 模型权重路径（checkpoints/best_model.pth）
            device: 计算设备
            backbone: 骨干网络类型
            num_classes: 训练时的类别数
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.preprocessor = ImagePreprocessor()
        
        # 使用 build_backbone 构建模型（与训练时的架构一致）
        print(f"[ReIDSystem] 加载模型 backbone={backbone}, device={self.device}")
        self.model = build_backbone(backbone, num_classes=num_classes, loss='softmax', pretrained=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载训练好的权重
        if model_path and Path(model_path).exists():
            self._load_weights(model_path)
        elif model_path:
            print(f"[ReIDSystem] WARNING: 模型文件不存在: {model_path}")
        
        # 特征库
        self.feature_gallery = []
        self.image_paths = []
        self.person_ids = []
    
    def _load_weights(self, model_path):
        """加载 checkpoint 权重"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 尝试多种可能的 key 格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 验证key匹配
        model_keys = set(self.model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        common = model_keys & ckpt_keys
        print(f"[ReIDSystem] checkpoint keys: {len(ckpt_keys)}, model keys: {len(model_keys)}, matching: {len(common)}")
        
        if len(common) < len(model_keys) * 0.5:
            print(f"[ReIDSystem] WARNING: 只有 {len(common)}/{len(model_keys)} 个key匹配！")
            print(f"[ReIDSystem] 尝试自动重映射key...")
            # 尝试去掉可见光分支前缀的映射
            new_sd = {}
            for k in model_keys:
                # 检查 checkpoint 中是否有去掉前缀后匹配的 key
                # checkpoint 格式: conv1.weight, bn1.weight, layer1.0.conv1.weight, ...
                # model 格式 (build_backbone): conv1.weight, bn1.weight, layer1.0.conv1.weight, ...
                if k in state_dict:
                    new_sd[k] = state_dict[k]
            
            if len(new_sd) > 0:
                print(f"[ReIDSystem] 重映射后匹配: {len(new_sd)}/{len(model_keys)}")
                state_dict = new_sd
        
        # 加载权重
        try:
            self.model.load_state_dict(state_dict, strict=False)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                print(f"[ReIDSystem] 未找到的key ({len(missing)}): {missing[:5]}...")
            if len(unexpected) > 0:
                print(f"[ReIDSystem] 多余的key ({len(unexpected)}): {unexpected[:5]}...")
            print(f"[ReIDSystem] 模型权重加载成功: {model_path}")
            
            # 显示训练指标
            if 'rank1' in checkpoint:
                print(f"    Rank-1: {checkpoint.get('rank1', 'N/A'):.2f}%")
            if 'rank5' in checkpoint:
                print(f"    Rank-5: {checkpoint.get('rank5', 'N/A'):.2f}%")
            if 'mAP' in checkpoint:
                print(f"    mAP: {checkpoint.get('mAP', 'N/A'):.2f}%")
        except Exception as e:
            print(f"[ReIDSystem] 权重加载失败: {e}")
            print(f"[ReIDSystem] 将继续使用随机初始化的模型")
    
    def extract_features(self, image, is_infrared=False):
        """提取图像特征"""
        with torch.no_grad():
            tensor = self.preprocessor.preprocess(image, is_infrared).to(self.device)
            outputs = self.model(tensor)
            # build_backbone 模型返回 (logits, feat)
            feat = outputs[1] if isinstance(outputs, tuple) else outputs
            return feat.cpu().numpy().squeeze()
    
    def extract_features_fusion(self, visible_img, infrared_img):
        """双模态特征融合（使用可见光模型处理两个输入后平均）"""
        feat_v = self.extract_features(visible_img, is_infrared=False)
        feat_i = self.extract_features(infrared_img, is_infrared=True)
        # 平均融合
        return (feat_v + feat_i) / 2.0
    
    def add_to_gallery(self, image, person_id, image_path=None, is_infrared=False):
        """将图像添加到特征库"""
        feat = self.extract_features(image, is_infrared)
        self.feature_gallery.append(feat)
        self.person_ids.append(person_id)
        self.image_paths.append(image_path)
    
    def search(self, query_image, is_infrared=False, top_k=5):
        """在特征库中搜索相似人员"""
        if len(self.feature_gallery) == 0:
            return []
        
        query_feat = self.extract_features(query_image, is_infrared)
        gallery_feats = np.array(self.feature_gallery)
        
        # L2 归一化
        query_norm = np.linalg.norm(query_feat)
        if query_norm > 0:
            query_feat = query_feat / query_norm
        gallery_norms = np.linalg.norm(gallery_feats, axis=1, keepdims=True)
        gallery_norms = np.where(gallery_norms > 0, gallery_norms, 1.0)
        gallery_feats_normalized = gallery_feats / gallery_norms
        
        # 余弦相似度
        similarities = np.dot(gallery_feats_normalized, query_feat)
        similarities = np.maximum(0.0, similarities)
        
        # 获取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'person_id': self.person_ids[idx],
                'similarity': float(similarities[idx]),
                'image_path': self.image_paths[idx]
            })
        
        return results
    
    def save_gallery(self, path):
        """保存特征库"""
        np.savez(path, 
                 features=np.array(self.feature_gallery),
                 person_ids=np.array(self.person_ids),
                 image_paths=np.array(self.image_paths, dtype=object))
    
    def load_gallery(self, path):
        """加载特征库"""
        data = np.load(path, allow_pickle=True)
        self.feature_gallery = data['features'].tolist()
        self.person_ids = data['person_ids'].tolist()
        self.image_paths = data['image_paths'].tolist()


if __name__ == '__main__':
    # 测试代码
    import sys
    print("=" * 50)
    print("ReIDSystem 测试")
    print("=" * 50)
    
    # 检查模型文件
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"
    if ckpt_path.exists():
        system = ReIDSystem(model_path=str(ckpt_path), device='cpu')
    else:
        print("模型文件不存在，使用随机初始化模型")
        system = ReIDSystem(device='cpu')
    
    # 测试特征提取
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    feat = system.extract_features(test_img, is_infrared=False)
    print(f"特征维度: {feat.shape}")
    print(f"特征范数: {np.linalg.norm(feat):.4f}")
    
    # 测试搜索
    system.add_to_gallery(test_img, 'person_001')
    results = system.search(test_img)
    print(f"搜索结果: {results}")
    
    print("系统初始化成功！")
