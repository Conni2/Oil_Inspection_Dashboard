import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="오일 상태 점검 데모", page_icon="⛑️")

st.title("✍️데이터 수집 및 축적")
st.write("데이터를 모을 수 있도록 도와드립니다.")

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

# 데이터프레임 초기화 또는 로드
if 'component_dfs' not in st.session_state:
    st.session_state.component_dfs = {component: pd.DataFrame(columns=['Date', 'Random_Values', 'Prediction']) for component in components}

component_dfs = st.session_state.component_dfs

def load_model(model_path):
    return joblib.load(model_path)

def predict_fault(model, scaler, input_values, features):
    data = np.zeros((1, len(features)))
    for i, feature in enumerate(features):
        data[0, i] = input_values[i]
    data_scaled = scaler.transform(data)
    prediction = model.predict_proba(data_scaled)[:, 1]
    return prediction[0]

# 랜덤 값 생성 함수 (데이터 분포 고려)
def generate_random_values(component_data):
    random_values = {}
    for column in component_data.columns:
        if column in ['ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY']:
            continue
        if component_data[column].dtype == 'object':
            random_values[column] = np.random.choice(component_data[column].dropna().values)
        else:
            mean = component_data[column].mean()
            std = component_data[column].std()
            random_values[column] = np.random.normal(mean, std)
    return random_values

# 클러스터에 따라 모델 경로 결정 (임시로 설정)
def get_model_path(cluster):
    return f"models/cat_3{chr(97+cluster)}_model.joblib"

# 날짜 선택
selected_date = st.date_input("점검 날짜를 선택하세요:", datetime.today())

# Streamlit 앱 실행
if st.button("오일 상태 점검"):
    for component, model_name in zip(components, ['1', '2', '3', '4']):
        st.write(f"### {component}")

        # 데이터 준비
        component_data = data[data['COMPONENT_ARBITRARY'] == component]
        if not component_data.empty:  # Check if data is not empty
            # Drop unnecessary columns
            component_data = component_data.drop(columns=['ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY'])
            
            # Fill missing values and preprocess data
            component_data = component_data.fillna(component_data.mean())  # Fill missing values with mean

            X = component_data  # Use all columns as features

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            # 랜덤 값 생성
            input_values = generate_random_values(component_data)

            if component == 'COMPONENT3':
                cluster = 0  # Assume cluster 0 as default
                # 클러스터 예측 로직을 여기에 추가하여 cluster 값을 설정
                # 여기서는 임시로 cluster 0으로 설정했습니다.
                model_path = get_model_path(cluster)
            else:
                model_path = f"models/cat_{model_name}_model.joblib"

            model = load_model(model_path)
            features = X.columns

            # 예측 결과
            prediction = predict_fault(model, scaler, list(input_values.values()), features)

            # 예측 결과를 데이터프레임에 추가
            result_dict = {'Date': selected_date, 'Random_Values': input_values, 'Prediction': prediction}
            component_dfs[component] = component_dfs[component].append(result_dict, ignore_index=True)

            # 예측 결과를 아이콘으로 시각화
            status = "🟢 정상" if prediction < 0.25 else "🟡 주의" if prediction < 0.5 else "🟠 경고" if prediction < 0.75 else "🔴 위험"
            st.write(f"{component}: 예측 결과 - {status}")

    # 세션 상태에 데이터프레임 저장
    st.session_state.component_dfs = component_dfs

# 모든 컴포넌트에 대한 예측 결과를 보여줌
with st.expander("전체 예측 결과 보기"):
    st.write("전체 예측 결과:")
    for component, df in component_dfs.items():
        st.write(f"#### {component}")
        st.write(df)

# Proba를 기반으로 한 선 그래프
selected_components = st.multiselect("그래프를 보고 싶은 컴포넌트를 선택하세요:", components)
if selected_components:
    st.write("### Prediction 기반 선 그래프")
    for component in selected_components:
        st.write(f"#### {component}")
        df = component_dfs[component].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        st.line_chart(df['Prediction'])
