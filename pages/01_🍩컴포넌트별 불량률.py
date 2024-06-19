import streamlit as st
import pandas as pd
import plotly.express as px

def load_data():
    # 데이터 로드 (예: casting.csv 파일이 존재해야 함)
    df = pd.read_csv("./data/casting.csv")
    return df

def main():
    st.title("🚜Component별 오일 상태 불량")
    
    # 데이터 로드
    df = load_data()
    
    # 고유 컴포넌트 목록
    unique_components = df['COMPONENT_ARBITRARY'].unique()
    
    # 컴포넌트 선택 박스
    selected_component = st.selectbox('⭐ Component 선택', unique_components)
    st.divider()
    
    # 선택된 컴포넌트에 해당하는 데이터 필터링
    filtered_data = df[df['COMPONENT_ARBITRARY'] == selected_component]
    
    if not filtered_data.empty:
        # 불량품과 정상품 개수 계산
        defect_count = filtered_data['Y_LABEL'].sum()
        pass_count = len(filtered_data) - defect_count
        
        # 불량률 계산
        total_count = len(filtered_data)
        defect_ratio = defect_count / total_count * 100
        
        # 파이 차트 그리기
        fig = px.pie(names=['정상품', '불량품'], values=[pass_count, defect_count],
                     title=f'🍩 Component {selected_component}의 불량률 ({defect_ratio:.2f}%)', hole=0.4)
        st.plotly_chart(fig)
        
        # 불량품 데이터 표시
        st.subheader('데이터 상세 정보')
        st.write(filtered_data)
    else:
        st.warning('선택한 컴포넌트에 대한 데이터가 없습니다.')

if __name__ == '__main__':
    main()