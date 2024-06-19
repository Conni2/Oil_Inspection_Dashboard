import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

# 페이지 설정
st.set_page_config(page_title="오일 상태 점검 데모", page_icon="⛑️")

st.title("⛑️ 오일 상태 점검 데모")
st.write("🗨️ 각 컴포넌트에 대해 랜덤 값을 넣고 오일 상태를 점검합니다.")
st.write("아래 버튼을 눌러 랜덤 값을 생성하여 예측 결과를 확인하세요.")

# Proba 및 예측결과 표 생성
data = {
    "Proba": ["~0.40", "0.40~0.45", "0.45~0.50", "0.50~"],
    "예측결과": ["🟢 정상", "🟡 주의", "🟠 경고", "🔴 위험"]
}
tabel_side = pd.DataFrame(data)

# 사이드바에 표 삽입
st.sidebar.write("### 🚦예측 결과에 따른 상태")
st.sidebar.write(tabel_side.to_markdown(index=False), unsafe_allow_html=True)

# 컴포넌트 목록
components = ['COMPONENT1', 'COMPONENT2', 'COMPONENT3', 'COMPONENT4']

# 랜덤으로 값을 생성하기 위해 데이터 불러오기
data = pd.read_csv("./data/casting.csv")

def load_model(model_path):
    return joblib.load(model_path)

def predict_fault(model, scaler, input_values, features):
    data = np.zeros((1, len(features)))
    for i, feature in enumerate(features):
        data[0, i] = input_values[i]
    data_scaled = scaler.transform(data)
    prediction = model.predict_proba(data_scaled)[:, 1]
    if prediction[0] < 0.40:
        return f"🟢정상: {prediction[0]:.2f} 확률로 불량입니다. 계속 사용하세요."
    elif 0.40 <= prediction[0] < 0.45:
        return f"🟡주의: {prediction[0]:.2f} 확률로 불량입니다. 사용에 주의하세요."
    elif 0.45 <= prediction[0] < 0.50:
        return f"🟠경고: {prediction[0]:.2f} 확률로 불량입니다. 마음의 준비가 필요할 것 같습니다."
    else:
        return f"🔴위험: {prediction[0]:.2f} 확률로 불량입니다. 당장 교체하세요."

# 랜덤 값 생성 함수
def generate_random_values(features):
    random_values = np.random.rand(len(features))
    return dict(zip(features, random_values))

# 클러스터에 따라 모델 경로 결정
def get_model_path(cluster):
    return f"models/cat_3{chr(97+cluster)}_model.joblib"

# Streamlit 앱 실행
if st.button("오일 상태 점검"):
    predictions = []
    warning_components = []

    for component, model_name in zip(components, ['1', '2', '3', '4']):
        st.write(f"### {component}")

        # 데이터 준비
        component_data = data[data['COMPONENT_ARBITRARY'] == component]
        if not component_data.empty:  # Check if data is not empty
            if component in ['COMPONENT1', 'COMPONENT2', 'COMPONENT4']:
                component_data = component_data.drop(columns=['ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY', 'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4'])
                component_data['CD'] = component_data['CD'].fillna(component_data['CD'].mode()[0])
                component_data['K'] = component_data['K'].fillna(component_data['K'].mean())
            else:  # COMPONENT3
                component_data = component_data.drop(columns=['ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY', 'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4', 'CD'])
                component_data['K'] = component_data['K'].fillna(component_data['K'].mean())

            X = component_data
        else:
            st.write(f"{component}: 데이터가 없습니다.")
            continue

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # 랜덤 값 생성
        input_values = generate_random_values(X.columns)

        if component == 'COMPONENT3':
            cluster = 0  # Assume cluster 0 as default
            # 클러스터 예측 로직을 여기에 추가하여 cluster 값을 설정
            # 여기서는 임시로 cluster 0으로 설정했습니다.
            model_path = get_model_path(cluster)
        else:
            model_path = f"models/cat_{model_name}_model.joblib"

        model = load_model(model_path)
        features = X.columns

        # 랜덤 값을 표시하는 토글
        expander_text = "🎰 랜덤 값 보기"
        with st.expander(expander_text):
            st.table(pd.DataFrame.from_dict(input_values, orient='index', columns=['값']))

        prediction = predict_fault(model, scaler, list(input_values.values()), features)

        st.write(f"{component}: {prediction}")
        predictions.append(prediction)
        if "위험" in prediction or "경고" in prediction:
            warning_components.append(component)

    if not warning_components:
        st.write("🥳모든 컴포넌트에 문제가 없습니다!")

    else:
        st.write(f"⚠️오일 교체가 필요한 컴포넌트: {', '.join(warning_components)}")