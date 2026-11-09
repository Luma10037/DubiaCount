"""DubiaCount Streamlit Web Application."""

import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="DubiaCount | 杜比亞蟑螂智能計數系統",
    page_icon="🪳",
    layout="wide",
)

st.title("🪳 DubiaCount: 杜比亞蟑螂密集目標檢測與計數")
st.markdown("基於 YOLO 與電腦視覺的高密度群體自動計數系統，專為生物飼育與科研場景設計。")

# Sidebar configurations
st.sidebar.header("⚙️ 偵測與模型參數")
model_choice = st.sidebar.selectbox(
    "選擇模型權重",
    ["weights/best.pt", "yolov8n.pt", "yolo11n.pt"],
    index=1,
    help="訓練好的客製模型通常存放在 weights/best.pt",
)
conf_threshold = st.sidebar.slider("信心度閾值 (Confidence Threshold)", 0.05, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU 閾值 (NMS IoU Threshold)", 0.1, 0.9, 0.45, 0.05)
use_sahi = st.sidebar.checkbox("啟用 SAHI 切圖推論 (適合密集小目標)", value=False)

uploaded_file = st.file_uploader("請上傳飼養箱或托盤照片 (JPG / PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1.subheader("📸 原始影像")
    col1.image(image, use_container_width=True)

    with st.spinner("正在進行偵測與計數..."):
        try:
            model = YOLO(model_choice)
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes
            total_count = len(boxes)

            # Render visualization
            annotated_frame = result.plot()

            col2.subheader("🎯 檢測結果")
            col2.image(annotated_frame, use_container_width=True)

            # Metric Cards
            st.markdown("---")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("預測個體總數", f"{total_count} 隻")
            avg_conf = f"{(boxes.conf.mean().item() * 100):.1f}%" if total_count > 0 else "0%"
            metric_col2.metric("平均信心度", avg_conf)
            metric_col3.metric("推論耗時", f"{(result.speed['inference']):.1f} ms")

        except Exception as e:
            st.error(f"推論過程發生錯誤: {e}")
            st.info("提示：若使用客製化模型，請確保權重檔案存在於指定路徑。")
else:
    st.info("👈 請於左側或上方上傳照片以開始進行自動計數。")
