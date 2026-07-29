import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. 페이지 기본 설정 (귀여운 타이틀 및 아이콘)
st.set_page_config(
    page_title="오늘 나의 기분은? 🎈",
    page_icon="😊",
    layout="centered"
)

# 2. 커스텀 CSS로 귀여운 분위기 연출
st.markdown("""
    <style>
    .main {
        background-color: #FFF9F3;
    }
    .stButton>button {
        background-color: #FFB7B2;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF9AA2;
        color: white;
    }
    .title-text {
        color: #6C5CE7;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .result-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>✨ AI 기분 측정기 ✨</h1>", unsafe_allow_html=True)
st.write("카메라로 얼굴 사진을 찍어주세요! 당신의 기분을 알려드릴게요 🌸")

# 3. 모델 및 라벨 로드 (캐싱을 이용해 로딩 속도 최적화)
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = load_my_model()
except Exception as e:
    st.error("⚠️ 모델(keras_model.h5) 또는 라벨 파일(labels.txt)을 찾을 수 없습니다. 파일이 같은 폴더에 있는지 확인해 주세요.")

# 4. 카메라 입력 기능
img_file_buffer = st.camera_input("📸 사진 찍기")

if img_file_buffer is not None:
    # 이미지 불러오기
    image = Image.open(img_file_buffer).convert('RGB')
    
    # Teachable Machine 전처리 규격 (224x224 정규화)
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 예측 수행
    with st.spinner("기분을 분석 중이에요... 🔮"):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

    # 클래스 명 정리 (보통 라벨 파일에 '0 웃음', '1 슬픔' 형태로 들어있으므로 숫자를 제거)
    clean_label = class_name.split(' ', 1)[-1] if ' ' in class_name else class_name

    # 결과 출력
    st.markdown("---")
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    
    if "웃음" in clean_label or "smile" in clean_label.lower():
        st.balloons()
        st.subheader("😊 당신은 지금 **행복한 웃음**을 지꼬 계시네요!")
        st.write(f"확신도: **{confidence_score * 100:.1f}%**")
        st.write("오늘 하루도 이 미소 그대로 행복하게 보내세요! ✨")
    elif "슬픔" in clean_label or "sad" in clean_label.lower():
        st.subheader("🥺 혹시 무슨 일 있으신가요? **슬픈 기분**이 느껴져요.")
        st.write(f"확신도: **{confidence_score * 100:.1f}%**")
        st.write("따뜻한 차 한 잔과 함께 편안한 휴식을 취해보는 건 어떨까요? 💕")
    else:
        st.subheader(f"결과: **{clean_label}**")
        st.write(f"확신도: **{confidence_score * 100:.1f}%**")

    st.markdown("</div>", unsafe_allow_html=True)