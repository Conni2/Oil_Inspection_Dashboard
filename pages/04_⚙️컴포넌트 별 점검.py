import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import joblib

st.set_page_config(
    page_title="건설 기계 오일 상태 진단",
    page_icon="🔧",
)

# Proba 및 예측결과 표 생성
data = {
    "Proba": ["~0.40", "0.40~0.45", "0.45~0.50", "0.50~"],
    "예측결과": ["🟢 정상", "🟡 주의", "🟠 경고", "🔴 위험"]
}
tabel_side = pd.DataFrame(data)

# 사이드바에 표 삽입
st.sidebar.write("### 🚦예측 결과에 따른 상태")
st.sidebar.write(tabel_side.to_markdown(index=False), unsafe_allow_html=True)


st.title("⚙️컴포넌트 별 건설 기계 오일 상태 진단")

############################ Component 1 ############################
# 데이터 로드
st.subheader("🛢️COMPONENT1 모델링 결과")
df = pd.read_csv("./data/casting.csv")

# 컴포넌트 1 분리
df_1 = df[df['COMPONENT_ARBITRARY'] == 'COMPONENT1']

# 불필요한 컬럼 제거
df_1 = df_1.drop(columns=['ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY', 'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4'])

# 결측치 대치
df_1['CD'] = df_1['CD'].fillna(df_1['CD'].mode()[0])
df_1['K'] = df_1['K'].fillna(df_1['K'].mean())

# 독립변수와 종속변수 분할
y = df_1['Y_LABEL']
X = df_1.drop(columns=['Y_LABEL'])

# 로버스트 스케일링
scaler = RobustScaler()
X = scaler.fit_transform(X)

# 테스트 데이터 분할
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.12, random_state=42)

# CatBoost 모델 설정 및 학습
cat_1 = CatBoostClassifier(
    iterations=140,
    learning_rate=0.17,
    depth=5,
    l2_leaf_reg=3,
    border_count=30,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=10
)
cat_1.fit(X_tr, y_tr)

# 저장할 디렉토리 경로 설정
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 모델 저장
joblib.dump(cat_1, os.path.join(save_dir, 'cat_1_model.joblib'))

# 테스트 세트에 대해 예측
y_pred = cat_1.predict(X_val)

# F1 스코어 계산
f1 = f1_score(y_val, y_pred)
print(f"F1 Score: {f1:.2f}")

# 변수 중요도 확인
feature_importance = cat_1.get_feature_importance()
important_features = np.argsort(feature_importance)[-3:]

print(f"가장 중요한 변수: {df_1.columns[important_features]}")

# 예측 함수 정의
def predict_fault(input_values, selected_features):
    data = np.zeros((1, X.shape[1]))
    for i, feature in enumerate(selected_features):
        data[0, list(df_1.columns).index(feature)] = input_values[i]
    data_scaled = scaler.transform(data)
    prediction = cat_1.predict_proba(data_scaled)[:, 1]
    if prediction[0] < 0.40:
        return f"🟢:정상: {prediction[0]:.2f} 확률로 불량입니다.계속 사용하세요."
    elif 0.40 <= prediction[0] < 0.45:
        return f"🟡주의: {prediction[0]:.2f} 확률로 불량입니다.사용에 주의하세요."
    elif 0.45 <= prediction[0] < 0.50:
        return f"🟠경고: {prediction[0]:.2f} 확률로 불량입니다.마음의 준비가 필요할 것 같습니다."
    else:
        return f"🔴위험: {prediction[0]:.2f} 확률로 불량입니다.당장 교체하세요."


# 모델 학습 버튼
if st.button("모델 학습", key="train_model"):
    cat_1.fit(X_tr, y_tr)
    st.write("모델 학습이 완료되었습니다.")

# 변수 선택 및 슬라이더 설정
features = list(df_1.columns)
selected_features = st.multiselect("변수 선택", features)

# 선택한 변수들에 대해 슬라이더 설정
input_data = []
for feature in selected_features:
    min_val = float(df_1[feature].min())
    max_val = float(df_1[feature].max())
    value = st.slider(f"{feature} 값을 입력하세요:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
    input_data.append(value)

# 입력 데이터로 예측 실행
if st.button("예측 실행", key="predict"):
    if 'cat_1' in locals():
        input_scaled = np.array(input_data).reshape(1, -1)
        prediction_result = predict_fault(input_scaled[0], selected_features)
        st.write(prediction_result)
    else:
        st.write("모델이 학습되지 않았습니다. '모델 학습' 버튼을 클릭하여 모델을 학습하세요.")
        
        
        
st.divider()


############################ Component 2 ############################
# 데이터 로드
st.subheader("🛢️COMPONENT2 모델링 결과")
df = pd.read_csv("./data/casting.csv")

# 컴포넌트 2 분리
df_2 = df[df['COMPONENT_ARBITRARY'] == 'COMPONENT2']

# 불필요한 컬럼 제거
df_2 = df_2.drop(columns=['ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY', 'FH2O', 'FNOX', 'FOPTIMETHGLY', 'FOXID', 'FSO4', 'FTBN', 'FUEL', 'SOOTPERCENTAGE', 'U4', 'U6', 'U14', 'U20', 'U25', 'U50', 'U75', 'U100', 'V100'])

# 결측치 대치
df_2['CD'] = df_2['CD'].fillna(df_2['CD'].mode()[0])
df_2['K'] = df_2['K'].fillna(df['K'].mean())

# 독립변수와 종속변수 분할
y = df_2['Y_LABEL']
X = df_2.drop(columns=['Y_LABEL'])

# 로버스트 스케일링
scaler = RobustScaler()
X = scaler.fit_transform(X)

# 테스트 데이터 분할
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# CatBoost 모델 설정 및 학습
cat_2 = CatBoostClassifier(
    iterations=120,
    learning_rate=0.7,
    depth=4,
    l2_leaf_reg=4,
    border_count=150,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=10
)
cat_2.fit(X_tr, y_tr)

# 저장할 디렉토리 경로 설정
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 모델 저장
joblib.dump(cat_2, os.path.join(save_dir, 'cat_2_model.joblib'))

# 테스트 세트에 대해 예측
y_pred = cat_2.predict(X_val)

# F1 스코어 계산
f1 = f1_score(y_val, y_pred)
print(f"F1 Score: {f1:.2f}")

# 변수 중요도 확인
feature_importance = cat_2.get_feature_importance()
important_features = np.argsort(feature_importance)[-3:]

print(f"가장 중요한 변수: {df_2.columns[important_features]}")

# 예측 함수 정의
def predict_fault(input_values, selected_features):
    data = np.zeros((1, X.shape[1]))
    for i, feature in enumerate(selected_features):
        data[0, list(df_2.columns).index(feature)] = input_values[i]
    data_scaled = scaler.transform(data)
    prediction = cat_2.predict_proba(data_scaled)[:, 1]
    if prediction[0] < 0.40:
        return f"🟢:정상: {prediction[0]:.2f} 확률로 불량입니다. 계속 사용하세요."
    elif 0.40 <= prediction[0] < 0.45:
        return f"🟡주의: {prediction[0]:.2f} 확률로 불량입니다. 사용에 주의하세요."
    elif 0.45 <= prediction[0] < 0.50:
        return f"🟠경고: {prediction[0]:.2f} 확률로 불량입니다. 마음의 준비가 필요할 것 같습니다."
    else:
        return f"🔴위험: {prediction[0]:.2f} 확률로 불량입니다. 당장 교체하세요."

# 모델 학습 버튼
if st.button("🛫모델 학습", key="train_model_button"):
    cat_2.fit(X_tr, y_tr)
    st.write("모델 학습이 완료되었습니다.")

# 변수 선택 및 슬라이더 설정
features = list(df_2.columns)
feature_selector_key = "feature_selector_" + str(np.random.randint(10000))  # 고유한 key 생성
selected_features = st.multiselect("변수 선택", features, key=feature_selector_key)

# 선택한 변수들에 대해 슬라이더 설정
input_data = []
for feature in selected_features:
    min_val = float(df_2[feature].min())
    max_val = float(df_2[feature].max())
    slider_key = f"slider_{feature}_{str(np.random.randint(10000))}"  # 고유한 key 생성
    value = st.slider(f"⭐{feature} 값을 입력하세요:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, key=slider_key)

    input_data.append(value)

# 입력 데이터로 예측 실행
predict_button_key = "predict_button_" + str(np.random.randint(10000))  # 고유한 key 생성
if st.button("🛬예측 실행", key=predict_button_key):
    if 'cat_2' in locals():
        input_scaled = np.array(input_data).reshape(1, -1)
        prediction_result = predict_fault(input_scaled[0], selected_features)
        st.write(prediction_result)
    else:
        st.write("모델이 학습되지 않았습니다. '모델 학습' 버튼을 클릭하여 모델을 학습하세요.")
        
        
st.divider()

############################ Component 3A ############################
# 데이터 로드
st.subheader("🛢️COMPONENT3A 모델링 결과")
df = pd.read_csv("./data/casting.csv")

# 컴포넌트 분리
df_3 = df[df['COMPONENT_ARBITRARY'] == 'COMPONENT3']

# 결측치 비율이 1인 변수들 목록
columns_to_drop = ['FUEL', 'FOPTIMETHGLY', 'FH2O', 'FOXID', 'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4', 'FSO4', 'V100', 'FTBN', 'SOOTPERCENTAGE', 'FNOX', 'ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY']

# 결측치 비율이 1인 변수들 제거
df_3 = df_3.drop(columns=columns_to_drop)

# 결측치 대치
cd_mode = df_3['CD'].mode()[0]
df_3['CD'].fillna(cd_mode, inplace=True)

# K 변수 모델링 (결측치 대치)
k_missing = df_3[df_3['K'].isnull()]
k_not_missing = df_3.dropna(subset=['K'])

# 피처와 타겟 변수 설정
X = k_not_missing[['NA', 'SI']]
y = k_not_missing['K']

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 결측치를 가지는 데이터에 대한 예측
X_missing = k_missing[['NA', 'SI']]
k_missing['K'] = model.predict(X_missing)

# 결측치를 포함한 데이터프레임을 업데이트
df_3.update(k_missing)

# 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=0)
df_3['Cluster'] = kmeans.fit_predict(df_3[['S', 'P']])

# 각 클러스터를 별도의 데이터프레임으로 저장
df_3a = df_3[df_3['Cluster'] == 0]

# 특성(Feature)와 라벨(Label) 분리
Xa = df_3a.drop(columns=['Y_LABEL'])
ya = df_3a['Y_LABEL']

# Robust Scaler 적용
scaler = RobustScaler()
X_scaleda = scaler.fit_transform(Xa)
X_scaled_df_3a = pd.DataFrame(X_scaleda, columns=Xa.columns) 

# 독립변수 종속변수 분할
Xa = X_scaled_df_3a
ya = df_3a['Y_LABEL']

# 테스트 데이터 분할
X_a, X_vala, y_a, y_vala = train_test_split(Xa, ya, test_size=0.2, random_state=42)

# CatBoost 분류 모델 생성 및 훈련
cat_3a = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0, border_count=32, l2_leaf_reg=5)
cat_3a.fit(X_a, y_a)

# 저장할 디렉토리 경로 설정
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 모델 저장
joblib.dump(cat_3a, os.path.join(save_dir, 'cat_3a_model.joblib'))

# 예측 확률 계산
pred_ca_proba = cat_3a.predict_proba(X_vala)[:, 1]  # 클래스 1 확률

# F1 Score 계산
pred_ca = (pred_ca_proba >= 0.5).astype(int)  # 임계값 0.5 기준으로 클래스 예측
f1 = f1_score(y_vala, pred_ca)

print(f"F1 Score: {f1:.2f}")

# 변수 중요도 확인
feature_importance = cat_3a.get_feature_importance()
important_features = np.argsort(feature_importance)[-3:]

print(f"가장 중요한 변수: {Xa.columns[important_features]}")

# 예측 함수 정의
def predict_fault(input_values, selected_features):
    # 전체 데이터의 길이를 맞추기 위해 0으로 초기화
    data = np.zeros((1, len(Xa.columns)))
    
    # 선택한 변수에 대한 값 할당
    for i, feature in enumerate(selected_features):
        data[0, Xa.columns.get_loc(feature)] = input_values[i]

    # 스케일러를 사용하여 데이터 스케일링
    data_scaled = scaler.transform(data)
    prediction = cat_3a.predict_proba(data_scaled)[:, 1]
    if prediction[0] < 0.40:
        return f"🟢:정상: {prediction[0]:.2f} 확률로 불량입니다.계속 사용하세요."
    elif 0.40 <= prediction[0] < 0.45:
        return f"🟡주의: {prediction[0]:.2f} 확률로 불량입니다.사용에 주의하세요."
    elif 0.45 <= prediction[0] < 0.50:
        return f"🟠경고: {prediction[0]:.2f} 확률로 불량입니다.마음의 준비가 필요할 것 같습니다."
    else:
        return f"🔴위험: {prediction[0]:.2f} 확률로 불량입니다.당장 교체하세요."

# 모델 학습 버튼
if st.button("🛫모델 학습"):
    cat_3a.fit(X_a, y_a)
    st.write("모델 학습이 완료되었습니다.")

# 변수 선택 및 슬라이더 설정
features = list(Xa.columns)
selected_features = st.multiselect("⭐변수 선택", features)

# 선택한 변수들에 대해 슬라이더 설정
input_data = []
for feature in selected_features:
    min_val = float(df_3[feature].min())
    max_val = float(df_3[feature].max())
    value = st.slider(f"💫{feature} 값을 입력하세요:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
    input_data.append(value)

# 입력 데이터로 예측 실행
if st.button("예측 실행"):
    if 'cat_3a' in locals():
        input_scaled = np.array(input_data).reshape(1, -1)
        prediction_result = predict_fault(input_scaled[0], selected_features)
        st.write(prediction_result)
    else:
        st.write("모델이 학습되지 않았습니다. '모델 학습' 버튼을 클릭하여 모델을 학습하세요.")
        
        
st.divider()
############################ Component 3B ############################

# 데이터 로드
st.subheader("🛢️COMPONENT3B 모델링 결과")
df = pd.read_csv("./data/casting.csv")

# 컴포넌트 분리
df_3 = df[df['COMPONENT_ARBITRARY'] == 'COMPONENT3']

# 결측치 비율이 1인 변수들 목록
columns_to_drop = ['FUEL', 'FOPTIMETHGLY', 'FH2O', 'FOXID', 'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4', 'FSO4', 'V100', 'FTBN', 'SOOTPERCENTAGE', 'FNOX', 'ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY']

# 결측치 비율이 1인 변수들 제거
df_3 = df_3.drop(columns=columns_to_drop)

# 결측치 대치
cd_mode = df_3['CD'].mode()[0]
df_3['CD'].fillna(cd_mode, inplace=True)

# K 변수 모델링 (결측치 대치)
k_missing = df_3[df_3['K'].isnull()]
k_not_missing = df_3.dropna(subset=['K'])

# 피처와 타겟 변수 설정
X = k_not_missing[['NA', 'SI']]
y = k_not_missing['K']

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 결측치를 가지는 데이터에 대한 예측
X_missing = k_missing[['NA', 'SI']]
k_missing['K'] = model.predict(X_missing)

# 결측치를 포함한 데이터프레임을 업데이트
df_3.update(k_missing)

# 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=0)
df_3['Cluster'] = kmeans.fit_predict(df_3[['S', 'P']])

# 각 클러스터를 별도의 데이터프레임으로 저장
df_3b = df_3[df_3['Cluster'] == 1]

# 특성(Feature)와 라벨(Label) 분리
Xb = df_3b.drop(columns=['Y_LABEL'])
yb = df_3b['Y_LABEL']

# Robust Scaler 적용
scaler = RobustScaler()
X_scaledb = scaler.fit_transform(Xb)
X_scaled_df_3b = pd.DataFrame(X_scaledb, columns=Xb.columns)

# 독립변수 종속변수 분할
Xb = X_scaled_df_3b
yb = df_3b['Y_LABEL']

# 테스트 데이터 분할
X_b, X_valb, y_b, y_valb = train_test_split(Xb, yb, test_size=0.2, random_state=42)

# CatBoost 모델 설정 및 학습
best_cat_3b = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=4,
    border_count=50,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=10
)

best_cat_3b.fit(X_b, y_b)

# 저장할 디렉토리 경로 설정
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 모델 저장
joblib.dump(best_cat_3b, os.path.join(save_dir, 'cat_3b_model.joblib'))

# 테스트 세트에 대해 예측
y_predb = best_cat_3b.predict(X_valb)

# F1 스코어 계산
f1 = f1_score(y_valb, y_predb)
print("F1 Score of y_predb(Best): {:.2f}".format(f1))
# F1 Score: 0.71

# 모델 학습 버튼
if st.button("🛫모델 학습", key="model_train_button"):
    best_cat_3b.fit(X_b, y_b)
    st.write("모델 학습이 완료되었습니다.")

# 변수 선택 및 슬라이더 설정
features = list(Xb.columns)
selected_features = st.multiselect("⭐변수 선택", features, key="feature_selector")

# 선택한 변수들에 대해 슬라이더 설정
input_data = []
for feature in selected_features:
    min_val = float(df_3b[feature].min())
    max_val = float(df_3b[feature].max())
    value = st.slider(f"💫{feature} 값을 입력하세요:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, key=f"slider_{feature}")

# 예측 함수 정의
def predict_fault(input_values, selected_features):
    # 전체 데이터의 길이를 맞추기 위해 0으로 초기화
    data = np.zeros((1, len(Xb.columns)))
    
    # 선택한 변수에 대한 값 할당
    for i, feature in enumerate(selected_features):
        data[0, Xb.columns.get_loc(feature)] = input_values[i]

    # 스케일러를 사용하여 데이터 스케일링
    data_scaled = scaler.transform(data)
    prediction = best_cat_3b.predict_proba(data_scaled)[:, 1]
    if prediction[0] < 0.40:
        return f"🟢:정상: {prediction[0]:.2f} 확률로 불량입니다. 계속 사용하세요."
    elif 0.40 <= prediction[0] < 0.45:
        return f"🟡주의: {prediction[0]:.2f} 확률로 불량입니다. 사용에 주의하세요."
    elif 0.45 <= prediction[0] < 0.50:
        return f"🟠경고: {prediction[0]:.2f} 확률로 불량입니다. 마음의 준비가 필요할 것 같습니다."
    else:
        return f"🔴위험: {prediction[0]:.2f} 확률로 불량입니다. 당장 교체하세요."

# 입력 데이터로 예측 실행
if st.button("🛬예측 실행", key="predict_button"):
    if 'best_cat_3b' in locals():
        input_scaled = np.array(input_data).reshape(1, -1)
        prediction_result = predict_fault(input_scaled[0], selected_features)
        st.write(prediction_result)
    else:
        st.write("모델이 학습되지 않았습니다. '모델 학습' 버튼을 클릭하여 모델을 학습하세요.")
        
        
st.divider()

############################ Component 3C ############################
# 데이터 로드
st.subheader("🛢️COMPONENT3C 모델링 결과")
df = pd.read_csv("./data/casting.csv")

# 컴포넌트 분리
df_3 = df[df['COMPONENT_ARBITRARY'] == 'COMPONENT3']

# 결측치 비율이 1인 변수들 목록
columns_to_drop = ['FUEL', 'FOPTIMETHGLY', 'FH2O', 'FOXID', 'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4', 'FSO4', 'V100', 'FTBN', 'SOOTPERCENTAGE', 'FNOX', 'ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY']

# 결측치 비율이 1인 변수들 제거
df_3 = df_3.drop(columns=columns_to_drop)

# 결측치 대치
cd_mode = df_3['CD'].mode()[0]
df_3['CD'].fillna(cd_mode, inplace=True)

# K 변수 모델링 (결측치 대치)
k_missing = df_3[df_3['K'].isnull()]
k_not_missing = df_3.dropna(subset=['K'])

# 피처와 타겟 변수 설정
X = k_not_missing[['NA', 'SI']]
y = k_not_missing['K']

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 결측치를 가지는 데이터에 대한 예측
X_missing = k_missing[['NA', 'SI']]
k_missing['K'] = model.predict(X_missing)

# 결측치를 포함한 데이터프레임을 업데이트
df_3.update(k_missing)

# 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=0)
df_3['Cluster'] = kmeans.fit_predict(df_3[['S', 'P']])

# 각 클러스터를 별도의 데이터프레임으로 저장
df_3c = df_3[df_3['Cluster'] == 2]

# 특성(Feature)와 라벨(Label) 분리
Xc = df_3c.drop(columns=['Y_LABEL'])
yc = df_3c['Y_LABEL']

# Robust Scaler 적용
scaler = RobustScaler()
X_scaledc = scaler.fit_transform(Xc)
X_scaled_df_3c = pd.DataFrame(X_scaledc, columns=Xc.columns)

# 독립변수 종속변수 분할
Xc = X_scaled_df_3c
yc = df_3c['Y_LABEL']

# 테스트 데이터 분할
X_c, X_valc, y_c, y_valc = train_test_split(Xc, yc, test_size=0.2, random_state=42)

# CatBoost 모델 설정 및 학습
best_cat_3c = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3,
    border_count=100,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=10
)

best_cat_3c.fit(X_c, y_c)

# 저장할 디렉토리 경로 설정
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 모델 저장
joblib.dump(best_cat_3c, os.path.join(save_dir, 'cat_3c_model.joblib'))

# 테스트 세트에 대해 예측
y_predc = best_cat_3c.predict(X_valc)

# F1 스코어 계산
f1 = f1_score(y_valc, y_predc)
print("F1 Score of y_predc(Best): {:.2f}".format(f1))
# F1 Score: 0.71

# Streamlit 대시보드
# 모델 학습 버튼
if st.button("🛫모델 학습", key="model_train_button_c"):
    best_cat_3c.fit(X_c, y_c)
    st.write("모델 학습이 완료되었습니다.")

# 변수 선택 및 슬라이더 설정
features = list(Xc.columns)
selected_features = st.multiselect("⭐변수 선택", features, key="feature_selector_c")

# 선택한 변수들에 대해 슬라이더 설정
input_data = []
for feature in selected_features:
    min_val = float(df_3c[feature].min())
    max_val = float(df_3c[feature].max())
    value = st.slider(f"💫{feature} 값을 입력하세요:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, key=f"slider_{feature}_c")

# 예측 함수 정의
def predict_fault(input_values, selected_features):
    # 전체 데이터의 길이를 맞추기 위해 0으로 초기화
    data = np.zeros((1, len(Xc.columns)))
    
    # 선택한 변수에 대한 값 할당
    for i, feature in enumerate(selected_features):
        data[0, Xc.columns.get_loc(feature)] = input_values[i]

    # 스케일러를 사용하여 데이터 스케일링
    data_scaled = scaler.transform(data)
    prediction = best_cat_3c.predict_proba(data_scaled)[:, 1]
    if prediction[0] < 0.40:
        return f"🟢:정상: {prediction[0]:.2f} 확률로 불량입니다. 계속 사용하세요."
    elif 0.40 <= prediction[0] < 0.45:
        return f"🟡주의: {prediction[0]:.2f} 확률로 불량입니다. 사용에 주의하세요."
    elif 0.45 <= prediction[0] < 0.50:
        return f"🟠경고: {prediction[0]:.2f} 확률로 불량입니다. 마음의 준비가 필요할 것 같습니다."
    else:
        return f"🔴위험: {prediction[0]:.2f} 확률로 불량입니다. 당장 교체하세요."

# 입력 데이터로 예측 실행
if st.button("🛬예측 실행", key="predict_button_c"):
    if 'best_cat_3c' in locals():
        input_scaled = np.array(input_data).reshape(1, -1)
        prediction_result = predict_fault(input_scaled[0], selected_features)
        st.write(prediction_result)
    else:
        st.write("모델이 학습되지 않았습니다. '모델 학습' 버튼을 클릭하여 모델을 학습하세요.")
        
        
st.divider()

############################ Component ALL & 4 ############################       
        
# 데이터 로드
st.subheader("🛢️COMPONENT ALL (4) 모델링 결과")
df = pd.read_csv("./data/casting.csv")

# 제거할 열 목록
columns_to_drop = ['ID', 'YEAR', 'SAMPLE_TRANSFER_DAY', 'COMPONENT_ARBITRARY', 'U100', 'U75', 'U50', 'U25', 'U14', 'U6', 'U4', 'FH2O', 'FNOX', 'FOPTIMETHGLY', 'FOXID', 'FSO4', 'FTBN', 'FUEL', 'SOOTPERCENTAGE']

# 열 제거
df = df.drop(columns=columns_to_drop)

# CD 열의 결측치를 0으로 대치
df['CD'] = df['CD'].fillna(df['CD'].mode()[0])

# K 변수의 결측치를 가지는 행만 선택
k_missing = df[df['K'].isnull()]

# 결측치를 가지지 않는 K 변수와 관련 변수들 선택
k_not_missing = df.dropna(subset=['K'])

# 피처와 타겟 변수 설정
X = k_not_missing[['NA', 'SI']]
y = k_not_missing['K']

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 결측치를 가지는 데이터에 대한 예측
X_missing = k_missing[['NA', 'SI']]
k_missing['K'] = model.predict(X_missing)

# 결측치를 포함한 데이터프레임을 업데이트
df.update(k_missing)

# 독립변수와 종속변수 분할
X = df.drop(columns=['Y_LABEL'])
y = df['Y_LABEL']

# Robust Scaler 적용
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 테스트 데이터 분할
X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# CatBoost 모델 설정 및 학습
best_cat = CatBoostClassifier(
    iterations=100,
    learning_rate=0.2,
    depth=4,
    l2_leaf_reg=3,
    border_count=32,
    verbose=0
)

best_cat.fit(X_tr, y_tr)

# 저장할 디렉토리 경로 설정
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 모델 저장
joblib.dump(best_cat, os.path.join(save_dir, 'cat_4_model.joblib'))

# 테스트 세트에 대해 예측
y_pred = best_cat.predict(X_val)

# F1 스코어 계산
f1 = f1_score(y_val, y_pred, average='macro')
print("F1 Score: {:.2f}".format(f1))

# 예측 함수 정의
def predict_fault(input_values, selected_features):
    data = np.zeros((1, X_scaled.shape[1]))
    for i, feature in enumerate(selected_features):
        data[0, list(df.columns).index(feature)] = input_values[i]
    data_scaled = scaler.transform(data)
    prediction = best_cat.predict_proba(data_scaled)[:, 1]
    if prediction[0] < 0.40:
        return f"🟢:정상: {prediction[0]:.2f} 확률로 불량입니다. 계속 사용하세요."
    elif 0.40 <= prediction[0] < 0.45:
        return f"🟡주의: {prediction[0]:.2f} 확률로 불량입니다. 사용에 주의하세요."
    elif 0.45 <= prediction[0] < 0.50:
        return f"🟠경고: {prediction[0]:.2f} 확률로 불량입니다. 마음의 준비가 필요할 것 같습니다."
    else:
        return f"🔴위험: {prediction[0]:.2f} 확률로 불량입니다. 당장 교체하세요."

# 모델 학습 버튼
train_button_key = "train_model_button_" + str(np.random.randint(10000))  # 고유한 key 생성
if st.button("🛫모델 학습", key=train_button_key):
    best_cat.fit(X_tr, y_tr)
    st.write("모델 학습이 완료되었습니다.")

# 변수 선택 및 슬라이더 설정
features = list(df.columns)
feature_selector_key = "feature_selector_" + str(np.random.randint(10000))  # 고유한 key 생성
selected_features = st.multiselect("⭐변수 선택", features, key=feature_selector_key)

# 선택한 변수들에 대해 슬라이더 설정
input_data = []
for feature in selected_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    slider_key = f"slider_{feature}_{str(np.random.randint(10000))}"  # 고유한 key 생성
    value = st.slider(f"💫{feature} 값을 입력하세요:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, key=slider_key)

    input_data.append(value)

# 입력 데이터로 예측 실행
predict_button_key = "predict_button_" + str(np.random.randint(10000))  # 고유한 key 생성
if st.button("🛬예측 실행", key=predict_button_key):
    if 'best_cat' in locals():
        input_scaled = np.array(input_data).reshape(1, -1)
        prediction_result = predict_fault(input_scaled[0], selected_features)
        st.write(prediction_result)
    else:
        st.write("모델이 학습되지 않았습니다. '모델 학습' 버튼을 클릭하여 모델을 학습하세요.")
