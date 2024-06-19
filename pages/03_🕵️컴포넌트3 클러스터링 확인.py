import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
# 페이지 설정
st.set_page_config(page_title="Component3 클러스터링", page_icon="👥")

# 데이터 로드
df = pd.read_csv('data/casting.csv')

# 컴포넌트 분리
df_3 = df[df['COMPONENT_ARBITRARY'] == 'COMPONENT3']

# 결측치 비율이 1인 변수들 제거
columns_to_drop = ['FUEL', 'FOPTIMETHGLY', 'FH2O', 'FOXID', 'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4', 'FSO4', 'V100', 'FTBN', 'SOOTPERCENTAGE', 'FNOX', 'ID', 'COMPONENT_ARBITRARY', 'YEAR', 'SAMPLE_TRANSFER_DAY']
df_3 = df_3.drop(columns=columns_to_drop)

# 결측치 대치
cd_mode = df_3['CD'].mode()[0]
df_3['CD'].fillna(cd_mode, inplace=True)

# K 변수 모델링 (결측치 대치)
k_missing = df_3[df_3['K'].isnull()]
k_not_missing = df_3.dropna(subset=['K'])
X = k_not_missing[['NA', 'SI']]
y = k_not_missing['K']
model = LinearRegression()
model.fit(X, y)
X_missing = k_missing[['NA', 'SI']]
k_missing['K'] = model.predict(X_missing)
df_3.update(k_missing)

# 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=0)
df_3['Cluster'] = kmeans.fit_predict(df_3[['S', 'P']])

# Streamlit 앱 생성
st.title("💁Component 3의 클러스터 예측 및 모델 추천")
st.write("❓Component 3의 데이터를 먼저 넣기 전에, 어떤 모델을 사용해야할지 추천해드려요!")
# 사용자 입력 받기
st.write("⭐아래 열들을 입력해주세요:")

# 입력받을 열 목록 (S와 P)
input_columns = ['S', 'P']

# 사용자 입력값을 저장할 딕셔너리
input_data = {}

for column in input_columns:
    min_val = float(df_3[column].min())
    max_val = float(df_3[column].max())
    value = st.number_input(f"💫{column} 값을 입력하세요:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, key=f"input_{column}")
    input_data[column] = value

# 예측 버튼
if st.button("🤔클러스터 예측"):
    # 입력 데이터를 데이터프레임으로 변환
    input_df = pd.DataFrame([input_data])
    
    # 클러스터 예측
    cluster = kmeans.predict(input_df)[0]

    # 결과 출력
    if cluster == 0:
        st.write("🔵 이 데이터는 모델 3A를 사용해주세요")
    elif cluster == 1:
        st.write("🟢 이 데이터는 모델 3B를 사용해주세요")
    else:
        st.write("🟣 이 데이터는 모델 3C를 사용해주세요")
