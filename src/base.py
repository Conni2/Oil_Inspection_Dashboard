import warnings
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import Config
from src.data import DataManager
from src.charts import MakeChart


def system_setting():
    warnings.filterwarnings(action='ignore')
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False


def initialize_objects():
    conf = Config()
    dtm = DataManager(conf)
    return dtm


def load_data(dtm):
    data_mart = dtm.create_data_mart()
    return data_mart


def base_dashboard(make_chart, data_mart):
    st.title("🚜Component별 오일 상태 불량")
    
    unique_components = data_mart['COMPONENT_ARBITRARY'].unique()
    selected_component = st.selectbox('⭐Component 선택', unique_components)
    st.divider()
    filtered_data = data_mart[data_mart['COMPONENT_ARBITRARY'] == selected_component]

    if not filtered_data.empty:
        defect_count = filtered_data['Y_LABEL'].sum()
        pass_count = len(filtered_data) - defect_count

        fig = px.pie(names=['정상품', '불량품'], values=[pass_count, defect_count],
                     title=f'🍩Component {selected_component}의 불량률', hole=0.4)
        st.plotly_chart(fig)

        st.write(filtered_data)


def main():
    system_setting()

    dtm = initialize_objects()
    data_mart = load_data(dtm)
    
    make_chart = MakeChart()
    
    base_dashboard(make_chart, data_mart)
    

if __name__ == "__main__":
    main()
