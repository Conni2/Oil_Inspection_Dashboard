from src.base import system_setting
from src.base import initialize_objects
from src.base import load_data
from src.base import base_dashboard
from src.charts import MakeChart

def main():
    system_setting()

    # 데이터 불러오기
    dtm = initialize_objects()
    data_mart = load_data(dtm)
    
    # 챠트 객체 생성
    make_chart = MakeChart()
    
    # 대시보드 생성
    base_dashboard(make_chart, data_mart)
    

if __name__ == "__main__":
    main()
