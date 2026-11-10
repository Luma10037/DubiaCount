"""DubiaCount Streamlit Web Application."""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.models.sahi_inference import SahiDubiaDetector

st.set_page_config(
    page_title="DubiaCount | 杜比亞蟑螂智能計數系統",
    page_icon="🪳",
    layout="wide",
)

st.title("🪳 DubiaCount: 杜比亞蟑螂密集目標檢測與計數")
st.markdown("基於 YOLO 與電腦視覺的高密度群體自動計數系統，專為生物飼育與科研場景設計。")

# Discover existing models, newest fine-tune first.
available_models: list[str] = []
discovered = [Path("weights/best.pt"), *Path("runs/detect").glob("*/weights/best.pt")]
discovered = [path for path in discovered if path.exists()]
discovered.sort(key=lambda path: path.stat().st_mtime, reverse=True)
for path in discovered:
    if str(path) not in available_models:
        available_models.append(str(path))

available_models.extend(["yolov8n.pt", "yolo11n.pt"])

# Sidebar configurations
st.sidebar.header("⚙️ 偵測與模型參數")
model_choice = st.sidebar.selectbox(
    "選擇模型權重",
    available_models,
    index=0,
    help="訓練完成後的權重會寫入 weights/best.pt",
)
use_sahi = st.sidebar.checkbox(
    "啟用切塊推論 (整張飼養箱照片請打開)",
    value=True,
    help="以 512 切塊偵測後，用重疊比例合併同一隻蟲在相鄰切塊上的框。",
)
conf_threshold = st.sidebar.slider("信心度閾值 (Confidence Threshold)", 0.05, 1.0, 0.25, 0.05)
iou_label = "切塊重疊合併閾值" if use_sahi else "IoU 閾值 (NMS IoU Threshold)"
iou_threshold = st.sidebar.slider(iou_label, 0.1, 0.9, 0.5 if use_sahi else 0.45, 0.05)

uploaded_file = st.file_uploader("請上傳飼養箱或托盤照片 (JPG / PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1.subheader("📸 原始影像")
    col1.image(image, use_container_width=True)

    with st.spinner("正在進行偵測與計數..."):
        try:
            start_t = time.time()
            if use_sahi:
                detector = SahiDubiaDetector(
                    model_path=model_choice,
                    confidence_threshold=conf_threshold,
                    device="cpu",
                    slice_height=512,
                    slice_width=512,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    postprocess_match_threshold=iou_threshold,
                    image_size=512,
                )
                res = detector.predict(np.array(image))
                total_count = res["count"]
                annotated_frame = res["visualized_image"]
                elapsed_ms = (time.time() - start_t) * 1000
                avg_conf = "SAHI 聚合"
            else:
                model = YOLO(model_choice)
                results = model.predict(
                    source=image,
                    imgsz=512,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                result = results[0]
                boxes = result.boxes
                total_count = len(boxes)
                annotated_frame = result.plot()
                elapsed_ms = result.speed["inference"]
                avg_conf = f"{(boxes.conf.mean().item() * 100):.1f}%" if total_count > 0 else "0%"

            col2.subheader("🎯 檢測結果")
            col2.image(annotated_frame, use_container_width=True)

            # Metric Cards
            st.markdown("---")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("預測個體總數", f"{total_count} 隻")
            metric_col2.metric("平均信心度", avg_conf)
            metric_col3.metric("推論耗時", f"{elapsed_ms:.1f} ms")

        except Exception as e:
            st.error(f"推論過程發生錯誤: {e}")
            st.info("提示：若使用客製化模型，請確保權重檔案存在於指定路徑。")
else:
    st.info("👈 請於左側或上方上傳照片以開始進行自動計數。")
