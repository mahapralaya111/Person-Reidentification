# API接口文档

## 概述

本系统提供基于Flask的RESTful API服务，支持特征提取、行人检索、图片比较等功能。

- **基础URL**: `http://localhost:5000`
- **数据格式**: JSON
- **编码**: UTF-8

---

## 接口列表

### 1. 健康检查

```
GET /health
```

检查API服务是否正常运行。

#### 请求示例
```bash
curl http://localhost:5000/health
```

#### 响应格式
```json
{
    "status": "ok",
    "service": "ReID API",
    "version": "2.0.0",
    "timestamp": "2026-05-05T18:00:00"
}
```

#### 状态码
| 状态码 | 说明 |
|--------|------|
| 200 | 服务正常 |
| 503 | 服务不可用 |

---

### 2. 提取特征

```
POST /feature
```

提取单张图片的特征向量。

#### 请求体
```json
{
    "image": "base64编码的图片数据",
    "image_url": "http://example.com/person.jpg"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | string | 二选一 | Base64编码的图片二进制数据 |
| `image_url` | string | 二选一 | 图片URL地址 |

#### 请求示例
```bash
# 使用base64编码
curl -X POST http://localhost:5000/feature \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -w0 person.jpg)'"}'

# 使用图片URL
curl -X POST http://localhost:5000/feature \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/person.jpg"}'
```

#### 响应格式
```json
{
    "feature": [0.123, -0.456, 0.789, ...],
    "norm": 1.000,
    "dimension": 512,
    "time_ms": 45.2
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `feature` | array | 特征向量数组，长度为512或2048 |
| `norm` | float | 特征向量的L2范数（通常接近1.0） |
| `dimension` | int | 特征维度 |
| `time_ms` | float | 处理耗时（毫秒） |

---

### 3. 比较两张图片

```
POST /compare
```

比较两张图片的相似度，判断是否为同一人。

#### 请求体
```json
{
    "image1": "base64编码的图片",
    "image2": "base64编码的图片"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image1` | string | 是 | 第一张图片的Base64编码 |
| `image2` | string | 是 | 第二张图片的Base64编码 |

#### 请求示例
```bash
curl -X POST http://localhost:5000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "image1": "'$(base64 -w0 person1.jpg)'",
    "image2": "'$(base64 -w0 person2.jpg)'"
  }'
```

#### 响应格式
```json
{
    "similarity": 0.8567,
    "same_person": true,
    "threshold": 0.7
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `similarity` | float | 余弦相似度，范围[-1, 1] |
| `same_person` | bool | 是否判定为同一人（基于阈值） |
| `threshold` | float | 当前判定阈值（可配置） |

#### 置信度参考
| 相似度区间 | 置信度 | 说明 |
|-----------|--------|------|
| > 0.7 | 高置信度 | 极可能是同一人 |
| 0.4 - 0.7 | 中置信度 | 可能是同一人 |
| 0.2 - 0.4 | 低置信度 | 不确定 |
| < 0.2 | 极低 | 很可能不是同一人 |

---

### 4. 行人检索

```
POST /search
```

在预加载的底库中搜索相似行人。

#### 请求体
```json
{
    "image": "base64编码的图片",
    "top_k": 10,
    "threshold": 0.2
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image` | string | 是 | - | 查询图片的Base64编码 |
| `top_k` | int | 否 | 10 | 返回前K个结果 |
| `threshold` | float | 否 | 0.2 | 相似度阈值，低于此值的结果将被过滤 |

#### 请求示例
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w0 query.jpg)'",
    "top_k": 5
  }'
```

#### 响应格式
```json
{
    "results": [
        {
            "id": "person_001",
            "similarity": 0.9567,
            "confidence": "高置信度"
        },
        {
            "id": "person_002",
            "similarity": 0.5345,
            "confidence": "中置信度"
        },
        {
            "id": "person_003",
            "similarity": 0.1123,
            "confidence": "低置信度"
        }
    ],
    "total_gallery": 100,
    "time_ms": 123.4
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `results[].id` | string | 人员ID（来自底库文件名或子文件夹名） |
| `results[].similarity` | float | 与查询图片的相似度 |
| `results[].confidence` | string | 置信度等级：高/中/低 |
| `total_gallery` | int | 底库中总图片数 |
| `time_ms` | float | 查询耗时（毫秒） |

---

### 5. 加载底库

```
POST /load_gallery
```

预加载底库图片到内存，提高后续搜索效率。

#### 请求体
```json
{
    "gallery_dir": "rescue_gallery"
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `gallery_dir` | string | 否 | "data/gallery" | 底库图片目录路径 |

#### 请求示例
```bash
curl -X POST http://localhost:5000/load_gallery \
  -H "Content-Type: application/json" \
  -d '{"gallery_dir": "rescue_gallery"}'
```

#### 响应格式
```json
{
    "status": "success",
    "gallery_size": 150,
    "feature_dim": 512,
    "persons_count": 25
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | "success" 或 "error" |
| `gallery_size` | int | 底库中的图片总数 |
| `feature_dim` | int | 特征维度 |
| `persons_count` | int | 不同人员的数量（按子文件夹统计） |

---

## 错误码说明

| HTTP状态码 | 错误类型 | 说明 |
|-----------|----------|------|
| 400 | Bad Request | 请求参数错误（如缺少必要字段） |
| 404 | Not Found | 请求的资源不存在 |
| 422 | Unprocessable Entity | 请求格式不正确（如Base64解码失败） |
| 500 | Internal Server Error | 服务器内部错误（如模型加载失败） |
| 503 | Service Unavailable | 服务暂不可用 |

#### 错误响应格式
```json
{
    "error": "错误描述信息",
    "code": "ERROR_CODE",
    "detail": "详细错误信息（调试模式）"
}
```

---

## 启动API服务

### 基本启动
```bash
python main.py api
# 或
cd web && python api.py
```

### 带参数启动
```bash
python web/api.py --port 5000 --model checkpoints/best_model.pth --gallery rescue_gallery
```

### 命令行参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--port` | int | 5000 | 服务端口号 |
| `--host` | string | "0.0.0.0" | 监听地址 |
| `--model` | string | "checkpoints/best_model.pth" | 模型文件路径 |
| `--gallery` | string | None | 预加载底库目录路径 |

---

## Python SDK

```python
import requests
import base64

class ReIDAPIClient:
    """ReID API客户端"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> dict:
        """健康检查"""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()
    
    def extract_feature(self, image_path: str = None, image_url: str = None) -> dict:
        """
        提取特征
        
        Args:
            image_path: 图片文件路径
            image_url: 图片URL
        """
        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            data = {"image": img_b64}
        elif image_url:
            data = {"image_url": image_url}
        else:
            raise ValueError("请提供 image_path 或 image_url")
        
        resp = requests.post(f"{self.base_url}/feature", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def compare_images(self, image_path1: str, image_path2: str) -> dict:
        """比较两张图片"""
        def encode(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        
        data = {
            "image1": encode(image_path1),
            "image2": encode(image_path2)
        }
        resp = requests.post(f"{self.base_url}/compare", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def search(self, query_path: str, top_k: int = 10) -> dict:
        """行人检索"""
        with open(query_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        data = {"image": img_b64, "top_k": top_k}
        resp = requests.post(f"{self.base_url}/search", json=data)
        resp.raise_for_status()
        return resp.json()
    
    def load_gallery(self, gallery_dir: str) -> dict:
        """加载底库"""
        resp = requests.post(f"{self.base_url}/load_gallery", 
                           json={"gallery_dir": gallery_dir})
        resp.raise_for_status()
        return resp.json()


# 使用示例
if __name__ == "__main__":
    client = ReIDAPIClient()
    
    # 健康检查
    print(client.health_check())
    
    # 提取特征
    feat = client.extract_feature(image_path="person.jpg")
    print(f"特征维度: {feat['dimension']}")
    
    # 比较图片
    result = client.compare_images("person1.jpg", "person2.jpg")
    print(f"相似度: {result['similarity']:.4f}")
    
    # 加载底库
    client.load_gallery("rescue_gallery")
    
    # 搜索
    results = client.search("query.jpg", top_k=5)
    for r in results["results"]:
        print(f"ID: {r['id']}, 相似度: {r['similarity']:.4f}")
```