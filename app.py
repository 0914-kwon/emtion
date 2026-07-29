import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
import tf2onnx
import tensorflow as tf

st.set_page_config(page_title="오늘 나의 기분은? 🎈", page_icon="😊", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #FFF9F3; }
    .title-text { color: #6C5CE7; text-align: center; font-family: 'Comic Sans MS', cursive, sans-serif; }
    .result-box { background-color: #FFFFFF; padding: 20px; border-radius: 15px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05); text-align: center; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>✨ AI 기분 측정기 ✨</h1>", unsafe_allow_html=True)
st.write("카메라로 얼굴 사진을 찍어주세요! 당신의 기분을 알려드릴게요 🌸")

# ONNX 세션 로드 함수 (TensorFlow 없이 가볍게 실행)
@st.cache_resource
def load_onnx_model():
    # keras_model.h5를 ONNX 형식으로 메모리상에서 변환 후 로드
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    session = ort.InferenceSession(onnx_model.SerializeToString())
    
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    return session, class_names, session.get_inputs()[0].name

try:
    session, class_names, input_name = load_onnx_model()
except Exception as e:
    st.error("⚠️ 모델(keras_model.h5) 또는 라벨 파일(labels.txt)을 찾을 수 없습니다.")

img_file_buffer = st.camera_input("📸 사진 찍기")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    with st.spinner("기분을 분석 중이에요... 🔮"):
        outputs = session.run(None, {input_name: data})
        prediction = outputs[0]
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

    clean_label = class_name.split(' ', 1)[-1] if ' ' in class_name else class_name

    st.markdown("---")
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    
    if "웃음" in clean_label or "smile" in clean_label.lower():
        st.balloons()
        st.subheader("😊 당신은 지금 **행복한 웃음**을 짓고 계시네요!")
        st.write(f"확신도: **{confidence_score * 100:.1f}%**")
    elif "슬픔" in clean_label or "sad" in clean_label.lower():
        st.subheader("🥺 혹시 무슨 일 있으신가요? **슬픈 기분**이 느껴져요.")
        st.write(f"확신도: **{confidence_score * 100:.1f}%**")
    else:
        st.subheader(f"결과: **{clean_label}**")
        st.write(f"확신도: **{confidence_score * 100:.1f}%**")

    st.markdown("</div>", unsafe_allow_html=True)
