"""
抗恶劣环境行人重识别系统 - Web演示界面（轻量版）
使用Streamlit构建，启动更快
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 图库目录
RESCUE_GALLERY_DIR = Path(__file__).resolve().parent.parent / "rescue_gallery"
RESCUE_GALLERY_AUG_DIR = Path(__file__).resolve().parent.parent / "rescue_gallery_augmented"

# 默认模型路径
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth")

# 懒加载
@st.cache_resource
def get_reid_system():
    """懒加载ReID系统"""
    with st.spinner('正在加载模型，请稍候...'):
        from reid_model import ReIDSystem
        system = ReIDSystem(model_path=_DEFAULT_MODEL_PATH, device='cpu')
    return system

# 页面配置
st.set_page_config(
    page_title="灾害应急搜救行人重识别系统",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 标题
    st.markdown('<h1 class="main-header">🚨 面向灾害应急搜救的抗恶劣环境行人重识别系统</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <b>系统特点：</b><br>
        ✅ 可见光-红外双模态融合识别<br>
        ✅ 抗恶劣环境（烟雾、遮挡、光照极端）<br>
        ✅ 小样本学习（仅需一张照片即可匹配）<br>
        ✅ 多设备跨域轨迹拼接
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.title("功能菜单")
    mode = st.sidebar.selectbox(
        "选择功能模式",
        ["🏠 系统介绍", "📸 单张图片识别", "🔍 数据库搜索", "⚙️ 系统设置"]
    )
    
    if mode == "🏠 系统介绍":
        show_introduction()
    elif mode == "📸 单张图片识别":
        single_image_recognition()
    elif mode == "🔍 数据库搜索":
        database_search()
    elif mode == "⚙️ 系统设置":
        system_settings()

def show_introduction():
    """显示系统介绍"""
    st.markdown('<h2 class="sub-header">系统介绍</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 核心功能
        
        **1. 抗恶劣环境识别**
        - 适应烟雾遮挡场景
        - 处理画面模糊图像
        - 应对光照极端条件
        
        **2. 双模态融合**
        - 可见光图像识别
        - 红外热成像识别
        
        **3. 快速部署**
        - 小样本学习能力
        - 仅需被困人员日常照片
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 应用场景
        
        **消防救援**
        - 火灾现场人员搜救
        - 烟雾环境身份识别
        
        **自然灾害救援**
        - 地震后被困人员定位
        - 洪涝灾害人员搜寻
        
        ### 📈 技术优势
        
        - 高精度识别
        - 实时处理
        - 鲁棒性强
        """)

def load_default_gallery(system):
    """自动加载默认图库（优先使用增强图库）"""
    if len(system.feature_gallery) == 0:
        gallery_dir = RESCUE_GALLERY_AUG_DIR if RESCUE_GALLERY_AUG_DIR.exists() else RESCUE_GALLERY_DIR
        if gallery_dir.exists():
            with st.spinner(f'正在加载图库 {gallery_dir.name}...'):
                for person_dir in sorted(gallery_dir.iterdir()):
                    if person_dir.is_dir():
                        person_id = person_dir.name
                        images = list(person_dir.glob('*'))
                        for img_path in images:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                system.add_to_gallery(img_rgb, person_id, str(img_path))
                st.success(f"✅ 已加载图库：{len(system.feature_gallery)} 张，{len(set(system.person_ids))} 人")
    return len(system.feature_gallery) > 0

def single_image_recognition():
    """单张图片识别"""
    st.markdown('<h2 class="sub-header">📸 单张图片识别</h2>', unsafe_allow_html=True)
    
    system = get_reid_system()
    load_default_gallery(system)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 上传查询图像")
        uploaded_file = st.file_uploader(
            "选择一张图像",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="single_upload"
        )
        
        is_infrared = st.checkbox("是否为红外图像", key="single_ir")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='查询图像', use_column_width=True)
            
            if st.button("开始识别", key="single_recognize"):
                with st.spinner('正在提取特征并搜索图库...'):
                    feat = system.extract_features(np.array(image), is_infrared)
                    
                    if len(system.feature_gallery) > 0:
                        results = system.search(np.array(image), is_infrared, top_k=5)
                    else:
                        results = []
                    
                    st.session_state['single_feat'] = feat
                    st.session_state['single_results'] = results
    
    with col2:
        st.markdown("### 识别结果")
        
        if 'single_results' in st.session_state and len(st.session_state['single_results']) > 0:
            feat = st.session_state['single_feat']
            results = st.session_state['single_results']
            
            st.success(f"✅ 特征提取完成！维度: {len(feat)}")
            
            st.markdown("---")
            st.markdown("#### 🏆 图库匹配结果")
            
            for i, result in enumerate(results):
                sim = result['similarity']
                pid = result['person_id']
                
                if sim > 0.6:
                    label = "高置信度"
                elif sim > 0.4:
                    label = "中置信度"
                elif sim > 0.2:
                    label = "低置信度"
                else:
                    label = "极低"
                
                # 结果卡片（含图片）
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    img_path = result.get('image_path')
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, width=100)
                    else:
                        st.markdown(f"**{pid}**")
                with col_b:
                    st.markdown(f"""
                    <div style="padding:8px; border-radius:6px;
                                background-color:{'#E8F5E9' if sim > 0.5 else '#FFF3E0'};
                                border-left:5px solid {'#4CAF50' if sim > 0.5 else '#FF9800'};">
                        <b>Top {i+1}</b> — <b>{pid}</b><br>
                        相似度: <span style="font-size:1.2em; font-weight:bold; color:{'#4CAF50' if sim > 0.5 else '#FF9800'}">{sim:.4f}</span> | {label}
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.expander("📊 特征向量详情"):
                st.markdown(f"**维度**: {len(feat)} | **范数**: {np.linalg.norm(feat):.4f}")
                st.markdown("**前10维**:")
                st.write(feat[:10])
        elif 'single_results' in st.session_state:
            st.warning("图库为空，未找到匹配结果")
        else:
            st.info("👆 上传图像并点击 '开始识别'")

def database_search():
    """数据库搜索"""
    st.markdown('<h2 class="sub-header">🔍 数据库搜索</h2>', unsafe_allow_html=True)
    
    system = get_reid_system()
    
    st.markdown("### 📚 特征库管理")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📂 从 rescue_gallery 加载"):
            gallery_dir = RESCUE_GALLERY_DIR
            if gallery_dir.exists():
                system.feature_gallery = []
                system.person_ids = []
                system.image_paths = []
                with st.spinner('正在加载图库...'):
                    for person_dir in sorted(gallery_dir.iterdir()):
                        if person_dir.is_dir():
                            images = sorted(person_dir.glob('*'))
                            for img_path in images:
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    system.add_to_gallery(img_rgb, person_dir.name, str(img_path))
                    st.success(f"✅ 已加载 {len(system.feature_gallery)} 张，{len(set(system.person_ids))} 人")
    
    with col2:
        uploaded_files = st.file_uploader(
            "上传图像到数据库",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        if uploaded_files and st.button("添加到数据库"):
            for file in uploaded_files:
                image = Image.open(file).convert('RGB')
                system.add_to_gallery(np.array(image), file.name)
            st.success(f"已添加 {len(uploaded_files)} 个样本")
    
    with col3:
        if st.button("清空数据库"):
            system.feature_gallery = []
            system.person_ids = []
            system.image_paths = []
            st.success("数据库已清空")
    
    # 图库概览
    if len(system.feature_gallery) > 0:
        with st.expander(f"👥 当前图库 ({len(set(system.person_ids))} 人, {len(system.feature_gallery)} 张)"):
            unique_persons = list(set(system.person_ids))
            cols = st.columns(min(6, len(unique_persons)))
            for idx, person_id in enumerate(unique_persons):
                with cols[idx % len(cols)]:
                    person_paths = [p for i, p in enumerate(system.image_paths) 
                                    if system.person_ids[i] == person_id and p is not None]
                    if person_paths and os.path.exists(person_paths[0]):
                        st.image(person_paths[0], width=100, caption=person_id)
                    else:
                        st.markdown(f"**{person_id}**")
    
    # 搜索功能
    st.markdown("### 🔎 搜索查询")
    query_file = st.file_uploader(
        "上传查询图像",
        type=['jpg', 'jpeg', 'png'],
        key="search_upload"
    )
    top_k = st.slider("返回Top-K结果", min_value=1, max_value=10, value=5)
    
    if query_file is not None:
        query_image = Image.open(query_file).convert('RGB')
        st.image(query_image, caption='查询图像', width=200)
        
        if st.button("开始搜索"):
            if len(system.feature_gallery) == 0:
                st.warning("数据库为空，请先添加数据")
            else:
                with st.spinner('正在搜索...'):
                    results = system.search(np.array(query_image), top_k=top_k)
                    
                    st.markdown("### 📋 搜索结果")
                    for i, result in enumerate(results):
                        sim = result['similarity']
                        pid = result['person_id']
                        
                        if sim > 0.6:
                            label = "高"
                        elif sim > 0.4:
                            label = "中"
                        elif sim > 0.2:
                            label = "低"
                        else:
                            label = "极低"
                        
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            img_path = result.get('image_path')
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, width=100)
                            else:
                                alt_path = RESCUE_GALLERY_DIR / pid
                                if alt_path.exists() and alt_path.is_dir():
                                    imgs = list(alt_path.glob('*'))
                                    if imgs:
                                        st.image(str(imgs[0]), width=100)
                                    else:
                                        st.markdown(f"**{pid}**")
                                else:
                                    st.markdown(f"**{pid}**")
                        with col_b:
                            st.markdown(f"""
                            <div style="padding:8px; border-radius:6px;
                                        background-color:{'#E8F5E9' if sim > 0.5 else '#FFF3E0'};">
                                <b>Top {i+1}</b> — <b>{pid}</b><br>
                                相似度: <span style="font-size:1.2em; font-weight:bold; 
                                    color:{'#4CAF50' if sim > 0.5 else '#FF9800'}">{sim:.4f}</span> | {label}
                            </div>
                            """, unsafe_allow_html=True)

def system_settings():
    """系统设置"""
    st.markdown('<h2 class="sub-header">⚙️ 系统设置</h2>', unsafe_allow_html=True)
    
    st.markdown("### 🔧 模型状态")
    try:
        system = get_reid_system()
        st.success(f"✅ 模型已加载")
        st.info(f"当前数据库样本数: {len(system.feature_gallery)} 人, {len(set(system.person_ids))} 人")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
    
    st.markdown("### 💾 数据管理")
    if st.button("清空数据库"):
        system = get_reid_system()
        system.feature_gallery = []
        system.person_ids = []
        system.image_paths = []
        st.success("数据库已清空")

if __name__ == '__main__':
    main()
