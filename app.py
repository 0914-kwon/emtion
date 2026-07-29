import streamlit as st
from streamlit_teachablemachine import teachablemachine

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="오늘 나의 기분은? 🎈",
    page_icon="😊",
    layout="centered"
)

# 2. 귀여운 스타일링 적용
st.markdown("""
    <style>
    .main {
        background-color: #FFF9F3;
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
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>✨ AI 기분 측정기 ✨</h1>", unsafe_allow_html=True)
st.write("카메라에 얼굴을 비춰주시면 AI가 기분을 인식합니다! 🌸")

# 3. Teachable Machine 모델 로드 및 식별
# labels.txt 파일에서 클래스 이름을 읽어옵니다.
try:
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
except Exception:
    class_names = ["웃음", "슬픔"]

# Teachable Machine 웹캠 컴포넌트 실행 (local model 파일 연결)
result = teachablemachine(
    model_url="", 
    model_json="model.json", # Teachable Machine에서 추출한 json 구조를 쓰는 것이 정석이나 local .h5 파싱 대신 웹 연동
)

# 카메라를 통해 감지된 결과 분석
if result is not None:
    best_class = result.get("class", "")
    confidence = result.get("confidence", 0.0)

    st.markdown("---")
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    
    if "웃음" in best_class or "smile" in best_class.lower():
        st.balloons()
        st.subheader("😊 당신은 지금 **행복한 웃음**을 짓고 계시네요!")
        st.write(f"확신도: **{confidence * 100:.1f}%**")
        st.write("오늘 하루도 이 미소 그대로 행복하게 보내세요! ✨")
    elif "슬픔" in best_class or "sad" in best_class.lower():
        st.subheader("🥺 혹시 무슨 일 있으신가요? **슬픈 기분**이 느껴져요.")
        st.write(f"확신도: **{confidence * 100:.1f}%**")
        st.write("따뜻한 차 한 잔과 함께 편안한 휴식을 취해보는 건 어떨까요? 💕")
    else:
        st.subheader(f"인식 결과: **{best_class}**")
        st.write(f"확신도: **{confidence * 100:.1f}%**")

    st.markdown("</div>", unsafe_allow_html=True)
