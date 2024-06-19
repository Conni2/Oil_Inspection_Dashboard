# dashboard_new.py

import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.title('🥸변수 별 분포 확인')
    

    # 데이터 로드
    df = pd.read_csv("./data/casting.csv")

    # 첫 번째 필터: COMPONENT_ARBITRARY
    selected_components = st.multiselect('⭐Component 선택', df['COMPONENT_ARBITRARY'].unique())

    # 두 번째 필터: 다른 열 선택
    columns_to_plot = st.selectbox('⭐열 선택', df.columns[~df.columns.isin(['Y_LABEL', 'COMPONENT_ARBITRARY', 'ID'])])

    # 필터링된 데이터
    filtered_data = df[df['COMPONENT_ARBITRARY'].isin(selected_components)]

    st.divider()
    
    # 전체 그래프
    st.write("### 🔧Component 별 전체 데이터 분포")
    st.write(f"📈{columns_to_plot}의 전체 분포")
    st.bar_chart(filtered_data[columns_to_plot].value_counts())

    st.divider()

    # 0이랑 1 나눈 그래프
    st.subheader("💧오일상태에 따른 분포")

    # Y_LABEL이 1인 데이터와 0인 데이터 분리
    label_1_data = filtered_data[filtered_data['Y_LABEL'] == 1]
    label_0_data = filtered_data[filtered_data['Y_LABEL'] == 0]

    # Plotly를 사용하여 그래프 그리기
    fig = px.histogram(filtered_data, x=columns_to_plot, color='Y_LABEL', barmode='overlay', 
                       labels={'Y_LABEL': 'Y_LABEL', columns_to_plot: 'Frequency'}, 
                       title=f"{columns_to_plot}의 전체 분포",
                       color_discrete_map={1: 'pink', 0: 'skyblue'})  # 색상 설정
    fig.update_traces(opacity=0.7)  # 투명도 설정
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
