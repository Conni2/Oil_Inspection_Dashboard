import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="건설 기계 데이터 분석통한 건설 기계 오일 상태 진단",
    page_icon="🏗️",
)

# Streamlit 대시보드
st.title('🏗️건설 기계 데이터 분석통한 건설 기계 오일 상태 진단')
st.balloons()
st.divider() # 구분선
image_url = "https://www.hd-hyundaice.com/common_ko/images/img-product-intro-5.jpg"

# HTML을 사용하여 이미지 크기 조정
st.markdown(f'<img src="{image_url}" width="700" height="350">', unsafe_allow_html=True)

st.markdown(
    """
    
    ### 🔍 분석 주제 선정 보고서

    [이 곳](https://drive.google.com/file/d/1uoVZMtQpyBGPH-z4w2SuTcFnBq0kv5Ns/view?usp=drive_link)에서 분석 주제 선정 보고서를 확인하세요.

    ### 🎯 건설 기계 데이터 수집을 통한 오일 상태 확인

    - **🤔선정배경**:
      - 오일 교체 중요성: 기계 성능 하락, 고장 위험, 부품 마모 증가, 안전 문제
      - 기존 오일 검사 방식 문제점: Sample Transfer Day 에 많은 시간이 소요되어 샘플 상태가 변경되고 기계 가동 중지 시간이 길어지면서 발생하는 손해

    - **🌟기대효과**:
      - 안전한 기계 운행
      - 부품 마모로 인한 기계 성능하락 및 고장 위험 미연에 방지
      - 정확한 오일 진단으로 필요시 오일 교체하여 ESG경영 및 운영 관리비 감소
      - 오일 검사 및 교체로 인한 작업 중단 시간 단축
    """

)
st.divider()

import streamlit as st
import pandas as pd

# 데이터 로드
file_path = 'data\data_info.xlsx - Features.csv'
df = pd.read_csv(file_path)

st.markdown(
    """
    ### 🕵️Data Overview
    """
)

with st.expander("컬럼 별 정보"):
    st.table(df)

st.divider()

st.markdown(
    """
    ### 🛹 Dashboard Overview
    """
)
st.image("src/그림3.png")
st.divider()

st.markdown(
    """
    
    ### 🎤 결과 보고서

    [이 곳](https://drive.google.com/file/d/1uoVZMtQpyBGPH-z4w2SuTcFnBq0kv5Ns/view?usp=drive_link)에서 결과 보고서를 확인하세요.
    """
)
