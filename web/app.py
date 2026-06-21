"""
抗恶劣环境行人重识别系统 - Web演示界面
使用Streamlit构建
"""

import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path
import pandas as pd

# 添加 src 目录到 Python 路径，确保模块可以导入
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 图库目录
RESCUE_GALLERY_DIR = Path(__file__).resolve().parent.parent / "rescue_gallery"
RESCUE_GALLERY_AUG_DIR = Path(__file__).resolve().parent.parent / "rescue_gallery_augmented"

# 懒加载自定义模块（只在需要时才导入，避免启动缓慢）
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth")

def get_reid_system():
    """懒加载ReID系统"""
    from reid_model import ReIDSystem
    @st.cache_resource
    def load():
        print(f"[Web] 加载模型: {_DEFAULT_MODEL_PATH}")
        system = ReIDSystem(model_path=_DEFAULT_MODEL_PATH, device='cpu')
        if _DEFAULT_MODEL_PATH and not os.path.exists(_DEFAULT_MODEL_PATH):
            print(f"[Web] WARNING: 模型文件不存在: {_DEFAULT_MODEL_PATH}")
        return system
    return load()

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
        ["🏠 系统介绍", "📸 单张图片识别", "🔍 数据库搜索", "🎬 视频识别", "📊 批量处理", "⚙️ 系统设置"]
    )
    
    # 加载系统（懒加载，避免启动缓慢）
    system = get_reid_system()
    
    if mode == "🏠 系统介绍":
        show_introduction()
    elif mode == "📸 单张图片识别":
        single_image_recognition(system)
    elif mode == "🔍 数据库搜索":
        database_search(system)
    elif mode == "🎬 视频识别":
        video_recognition(system)
    elif mode == "📊 批量处理":
        batch_processing(system)
    elif mode == "⚙️ 系统设置":
        system_settings(system)

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
        - 识别人体姿态畸变（匍匐、摔倒）
        - 处理衣物破损情况
        
        **2. 双模态融合**
        - 可见光图像识别
        - 红外热成像识别
        - 跨模态特征匹配
        
        **3. 快速部署**
        - 小样本学习能力
        - 仅需被困人员日常照片
        - 甚至模糊证件照也可匹配
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
        
        **应急管理**
        - 多摄像头轨迹追踪
        - 跨域人员重识别
        
        ### 📈 技术优势
        
        - **高精度**: 在恶劣环境下保持高识别率
        - **实时性**: 支持实时视频流处理
        - **鲁棒性**: 抗干扰能力强
        - **易部署**: 支持边缘设备部署
        """)
    
    # 技术架构图（用文字描述）
    st.markdown('<h2 class="sub-header">技术架构</h2>', unsafe_allow_html=True)
    st.markdown("""
    ```
    输入图像/视频
        ↓
    [预处理模块] - 去噪、增强、归一化
        ↓
    [特征提取网络] - ResNet50 backbone
        ↓         ↓
    可见光特征   红外特征
        ↓         ↓
    [特征融合模块] - 自适应融合
        ↓
    [特征库检索] - 余弦相似度匹配
        ↓
    识别结果输出
    ```
    """)

def load_default_gallery(system):
    """自动加载默认图库到内存（优先使用增强图库）"""
    if len(system.feature_gallery) == 0:
        # 优先使用增强图库（更多图片/角度）
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
                st.success(f"✅ 已自动加载图库：{len(system.feature_gallery)} 张图片，{len(set(system.person_ids))} 人")
    return len(system.feature_gallery) > 0


def single_image_recognition(system):
    """单张图片识别"""
    st.markdown('<h2 class="sub-header">📸 单张图片识别</h2>', unsafe_allow_html=True)
    
    # 自动加载图库
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
                    
                    # 在图库中搜索
                    if len(system.feature_gallery) > 0:
                        results = system.search(np.array(image), is_infrared, top_k=5)
                    else:
                        results = []
                    
                    # 存到 session_state 中持久化
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
                
                # 设置置信度标签
                if sim > 0.6:
                    label = "🔴 高置信度"
                elif sim > 0.4:
                    label = "🟡 中置信度"
                elif sim > 0.2:
                    label = "🟢 低置信度"
                else:
                    label = "⚪ 极低"
                
                # 显示结果卡片
                st.markdown(f"""
                <div style="padding:10px; margin:5px 0; border-radius:8px; 
                            background-color:{'#E8F5E9' if sim > 0.5 else '#FFF3E0'};
                            border-left:5px solid {'#4CAF50' if sim > 0.5 else '#FF9800'};">
                    <b>Top {i+1}</b> | <b>{pid}</b> | 相似度: <b>{sim:.4f}</b> | {label}
                </div>
                """, unsafe_allow_html=True)
                
                # 如果保存了路径，显示匹配到的图库图片
                if result.get('image_path') and os.path.exists(result['image_path']):
                    from PIL import Image as PILImage
                    img_match = PILImage.open(result['image_path']).convert('RGB')
                    st.image(img_match, width=140, caption=f"{pid} (图库图片)")
            
            # 展示特征向量简表
            with st.expander("📊 查看特征向量详情"):
                st.markdown(f"**特征维度**: {len(feat)}")
                st.markdown("**特征范数**: {:.4f}".format(np.linalg.norm(feat)))
                st.markdown("**前20维**:")
                feat_df = pd.DataFrame({
                    '维度': [f'dim_{j}' for j in range(min(20, len(feat)))],
                    '值': feat[:20]
                })
                st.table(feat_df)
        elif 'single_results' in st.session_state:
            st.warning("图库为空，未找到匹配结果，请先在'数据库搜索'中添加数据")
        else:
            st.info("👆 上传图像并点击 '开始识别'")

def database_search(system):
    """数据库搜索"""
    st.markdown('<h2 class="sub-header">🔍 数据库搜索</h2>', unsafe_allow_html=True)
    
    # 数据库管理
    st.markdown("### 📚 特征库管理")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📂 从 rescue_gallery 加载", key="init_db"):
            gallery_dir = RESCUE_GALLERY_DIR
            if gallery_dir.exists():
                system.feature_gallery = []
                system.person_ids = []
                system.image_paths = []
                with st.spinner(f'正在加载图库...'):
                    person_count = 0
                    for person_dir in sorted(gallery_dir.iterdir()):
                        if person_dir.is_dir():
                            images = sorted(person_dir.glob('*'))
                            for img_path in images:
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    system.add_to_gallery(img_rgb, person_dir.name, str(img_path))
                                    person_count += 1
                    st.success(f"✅ 已加载 {len(system.feature_gallery)} 张图片，共 {len(set(system.person_ids))} 人")
    
    with col2:
        uploaded_files = st.file_uploader(
            "批量上传图像到数据库",
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
        if st.button("清空数据库", key="clear_db"):
            system.feature_gallery = []
            system.person_ids = []
            system.image_paths = []
            st.success("数据库已清空")
    
    # 显示图库人员预览
    if len(system.feature_gallery) > 0:
        with st.expander(f"👥 当前图库概览 ({len(set(system.person_ids))} 人, {len(system.feature_gallery)} 张)"):
            unique_persons = list(set(system.person_ids))
            cols = st.columns(min(6, len(unique_persons)))
            for idx, person_id in enumerate(unique_persons):
                col_idx = idx % len(cols)
                with cols[col_idx]:
                    # 找到该人的第一张图
                    person_paths = [p for i, p in enumerate(system.image_paths) 
                                    if system.person_ids[i] == person_id and p is not None]
                    if person_paths and os.path.exists(person_paths[0]):
                        img = Image.open(person_paths[0]).convert('RGB')
                        st.image(img, width=100, caption=person_id)
                    else:
                        st.markdown(f"**{person_id}**")
    
    # 搜索功能
    st.markdown("### 🔎 搜索查询")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        query_file = st.file_uploader(
            "上传查询图像",
            type=['jpg', 'jpeg', 'png'],
            key="search_upload"
        )
        top_k = st.slider("返回Top-K结果", min_value=1, max_value=10, value=5)
        
        if query_file is not None:
            query_image = Image.open(query_file).convert('RGB')
            st.image(query_image, caption='查询图像', use_column_width=True)
    
    with col2:
        if query_file is not None and st.button("开始搜索"):
            if len(system.feature_gallery) == 0:
                st.warning("数据库为空，请先添加数据")
            else:
                with st.spinner('正在搜索...'):
                    results = system.search(np.array(query_image), top_k=top_k)
                    
                    st.markdown("### 📋 搜索结果")
                    for i, result in enumerate(results):
                        sim = result['similarity']
                        pid = result['person_id']
                        
                        # 置信度标签
                        if sim > 0.6:
                            label = "🔴 高"
                        elif sim > 0.4:
                            label = "🟡 中"
                        elif sim > 0.2:
                            label = "🟢 低"
                        else:
                            label = "⚪ 极低"
                        
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            # 显示匹配到的图库图片
                            img_path = result.get('image_path')
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, width=100)
                            else:
                                # 尝试从 rescue_gallery 查找
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
                                    color:{'#4CAF50' if sim > 0.5 else '#FF9800'}">{sim:.4f}</span>
                                &nbsp;|&nbsp; {label}
                            </div>
                            """, unsafe_allow_html=True)

def video_recognition(system):
    """视频识别处理"""
    st.markdown('<h2 class="sub-header">🎬 视频识别</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        上传视频文件，逐帧进行行人检测与识别。<br>
        支持 YOLOv8 行人检测 + ReID 特征匹配 + 卡尔曼跟踪。
    </div>
    """, unsafe_allow_html=True)

    # 自动加载图库
    load_default_gallery(system)

    uploaded_video = st.file_uploader("选择视频文件", type=['mp4','avi','mov','mkv'], key="video_upload")
    
    if uploaded_video is not None:
        # 保存上传的视频到临时文件
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        tfile.close()

        st.video(video_path)
        st.info(f"视频大小: {len(uploaded_video.getvalue())/1024/1024:.1f} MB")

        if st.button("开始视频识别", key="video_start"):
            import time as _time
            from pathlib import Path
            import torch, cv2, numpy as np
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

            # 加载与视频处理一致的模型和检索器
            from video_processor import load_reid_model, ReIDRetriever, VideoProcessor

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            vp_model = load_reid_model(str(Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"), device)

            # 构建底库（使用与视频处理相同的模型）
            from inference import ReidInferencer
            # 从checkpoint读取模型信息
            ckpt_for_web = torch.load(str(Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"),
                                      map_location='cpu', weights_only=False)
            ckpt_backbone = ckpt_for_web.get('backbone', 'resnet50')
            ckpt_num_classes = ckpt_for_web.get('num_classes', 751)
            temp_infer = ReidInferencer(
                model_path=str(Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"),
                backbone=ckpt_backbone,
                num_classes=ckpt_num_classes
            )
            gallery_dir = RESCUE_GALLERY_AUG_DIR if RESCUE_GALLERY_AUG_DIR.exists() else RESCUE_GALLERY_DIR
            if not gallery_dir.exists():
                st.error("底库目录不存在!")
                os.unlink(video_path)
                return
            gallery_feat, gallery_ids, gallery_paths = temp_infer.build_gallery(str(gallery_dir))
            if len(gallery_ids) == 0:
                st.error("底库为空!")
                os.unlink(video_path)
                return
            st.info(f"底库: {len(gallery_ids)}样本, {len(set(gallery_ids))}人")

            # 处理视频
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_placeholder = st.empty()

            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            vp = VideoProcessor(source=video_path, model=vp_model, device=device,
                                gallery_features=gallery_feat, gallery_ids=gallery_ids,
                                gallery_paths=gallery_paths, debug_mode=False)

            results_summary = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    result, tracks, nd = vp.process_frame(frame)
                except Exception as e:
                    continue
                if tracks:
                    for trk in tracks:
                        gm = trk.get('gallery_match', {})
                        if gm and gm.get('id') and gm.get('similarity', 0) > 0.1:
                            results_summary.append({
                                'frame': frame_idx,
                                'track_id': trk['id'],
                                'person': gm['id'],
                                'similarity': round(gm['similarity'], 4)
                            })
                frame_idx += 1
                if frame_idx % max(1, total//20) == 0:
                    progress_bar.progress(min(1.0, frame_idx/total))
                    status_text.text(f"处理中: {frame_idx}/{total} 帧")

            cap.release()
            vp.stop()
            os.unlink(video_path)

            progress_bar.progress(1.0)
            status_text.text(f"处理完成: {frame_idx} 帧")

            if results_summary:
                import pandas as pd
                df = pd.DataFrame(results_summary)
                # 帧号转为时间戳
                df['time'] = df['frame'].apply(lambda x: f"{int(x//fps//60):02d}:{int(x//fps%60):02d}")
                df['时间'] = df['frame'].apply(lambda x: f"{int(x//fps//3600):02d}:{int(x//fps%60//60):02d}:{int(x//fps%60):02d}")

                st.success(f"检测到 {len(set(r['person'] for r in results_summary))} 个不同人员, 共 {len(results_summary)} 帧匹配")

                # ⏱ 时间线报告（核心功能）—— 合并连续片段
                st.markdown("### ⏱ 人员出现时间线")
                from collections import defaultdict
                person_time_ranges = defaultdict(list)
                person_sims = defaultdict(list)
                for r in results_summary:
                    p = r['person']
                    t = r['frame'] / fps
                    person_time_ranges[p].append(t)
                    person_sims[p].append(r['similarity'])

                # 构建合并后的时间片段表格
                merged_rows = []
                for person, times in sorted(person_time_ranges.items()):
                    times = sorted(set(times))
                    sims = person_sims[person]
                    avg_sim = round(sum(sims) / len(sims), 4)
                    max_sim = round(max(sims), 4)
                    min_sim = round(min(sims), 4)

                    # 合并连续时间段（间隔>3秒算新一段）
                    segments = []
                    seg_start = times[0]
                    seg_end = times[0]
                    for t in times[1:]:
                        if t - seg_end > 3:
                            segments.append((seg_start, seg_end))
                            seg_start = t
                        seg_end = t
                    segments.append((seg_start, seg_end))

                    seg_str = []
                    for s, e in segments:
                        s_str = f"{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d}"
                        e_str = f"{int(e//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}"
                        dur = round(e - s, 1)
                        seg_str.append(f"**{s_str} ~ {e_str}** ({dur}秒)")
                        merged_rows.append({
                            '人员': person,
                            '开始时间': s_str,
                            '结束时间': e_str,
                            '持续时长(秒)': dur,
                            '平均相似度': avg_sim,
                            '最高相似度': max_sim
                        })
                    st.markdown(f"**{person}**：{' → '.join(seg_str)}")

                # 合并后的时间线表
                if merged_rows:
                    st.markdown("### 时间线明细（按片段合并）")
                    df_merged = pd.DataFrame(merged_rows)
                    st.dataframe(df_merged, use_container_width=True, hide_index=True)

                # 按人员统计
                st.markdown("### 📊 人员出现统计")
                person_stats = df.groupby('person').agg(
                    出现帧数=('frame', 'count'),
                    平均相似度=('similarity', 'mean'),
                    最高相似度=('similarity', 'max')
                ).sort_values('出现帧数', ascending=False)
                person_stats['出现时长(秒)'] = (person_stats['出现帧数'] / fps).round(1)
                st.dataframe(person_stats, use_container_width=True)

                # 下载结果（合并格式）
                if merged_rows:
                    csv_merged = pd.DataFrame(merged_rows).to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下载时间线报告 (CSV)", csv_merged, "timeline_merged.csv", "text/csv",
                                      help="按时间片段合并的简洁报告，适合导入Excel分析")
                # 也提供原始明细下载
                csv_raw = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下载原始帧级数据 (CSV)", csv_raw, "reid_frames.csv", "text/csv",
                                  help="每帧每个人的原始匹配数据，数据量较大")

                st.info("💡 建议下载「时间线报告」—— 已按人员合并连续片段，更简洁清晰")
            else:
                st.warning("视频中未识别到已知人员")
                st.info("提示: 请先在'数据库搜索'中添加待搜索的人员照片，或使用默认底库")

    st.markdown("""
    <div class="warning-box">
        <b>说明:</b><br>
        1. 视频识别使用 YOLOv8 检测行人 + ReID 特征匹配<br>
        2. 匹配结果基于底库中已有的人员照片<br>
        3. 处理速度受CPU性能影响，建议使用短视频<br>
        4. 可通过'数据库搜索'页面上传更多人员照片
    </div>
    """, unsafe_allow_html=True)


def batch_processing(system):
    """批量处理"""
    st.markdown('<h2 class="sub-header">📊 批量处理</h2>', unsafe_allow_html=True)
    
    st.markdown("### 上传批量图像")
    batch_files = st.file_uploader(
        "选择多张图像",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="batch_process"
    )
    
    if batch_files:
        st.info(f"已上传 {len(batch_files)} 张图像")
        
        if st.button("开始批量处理"):
            progress_bar = st.progress(0)
            results = []
            
            for i, file in enumerate(batch_files):
                image = Image.open(file).convert('RGB')
                feat = system.extract_features(np.array(image))
                
                results.append({
                    'filename': file.name,
                    'feature_shape': feat.shape,
                    'feature_mean': float(np.mean(feat))
                })
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            st.success("批量处理完成！")
            st.markdown("### 处理结果")
            st.table(results)

def system_settings(system):
    """系统设置"""
    st.markdown('<h2 class="sub-header">⚙️ 系统设置</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔧 模型配置")
        device = st.selectbox(
            "计算设备",
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
            key="device_select"
        )
        
        feat_dim = st.slider(
            "特征维度",
            min_value=128,
            max_value=2048,
            value=512,
            step=128,
            key="feat_dim"
        )
        
        st.markdown("### 📊 数据库状态")
        st.info(f"当前数据库样本数: {len(system.feature_gallery)}")
    
    with col2:
        st.markdown("### 💾 数据管理")
        
        if st.button("保存特征库", key="save_gallery"):
            system.save_gallery('gallery.npz')
            st.success("特征库已保存")
        
        if st.button("加载特征库", key="load_gallery"):
            if os.path.exists('gallery.npz'):
                system.load_gallery('gallery.npz')
                st.success(f"特征库已加载，包含 {len(system.feature_gallery)} 个样本")
            else:
                st.warning("未找到保存的特征库文件")
        
        st.markdown("### 🎨 视觉效果")
        show_confidence = st.checkbox("显示置信度", value=True)
        show_feature_map = st.checkbox("显示特征图", value=False)

if __name__ == '__main__':
    main()
