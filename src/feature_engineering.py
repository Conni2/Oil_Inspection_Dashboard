from datetime import datetime
import pandas as pd


class FeatureEngineer:
    def __init__(self):
        self.dt_format = "%Y-%m-%d %H:00:00"
        pass

    def create_time_features(self, data_before):
        data_after = data_before.copy()
        rtime = pd.to_datetime(data_after['registration_time'])
        data_after['date'] = rtime.dt.date
        data_after['hour'] = rtime.dt.hour
        data_after['weekday'] = rtime.dt.weekday.map({0:'월', 1:'화', 2:'수', 3:'목', 4:'금', 5:'토', 6:'일'})
        data_after['date_time'] = pd.to_datetime(rtime.apply(lambda t: datetime.strftime(t, self.dt_format)))
        return data_after
    
    def create_cycletime_feature(self, data_before):
        data_after = data_before.copy()
        data_after['average_cycle_time'] = (data_after['facility_operation_cycletime'] +\
                                             data_after['production_cycletime']) / 2
        return data_after
    
    def create_pass_fail_feature(self, data_before):
        data_after = data_before.copy()
        data_after['pass'] = data_after['Y_LABEL'].apply(lambda x: 1 if x == 0 else 0)
        data_after['fail'] = data_after['Y_LABEL'].apply(lambda x: 1 if x == 1 else 0)
        return data_after
    
    def aggregate(self, data_mart, key_cols=None):
        if key_cols is None:
            key_cols = ['date','date_time','mold_code','weekday','hour']
        gdf = data_mart.groupby(key_cols)

        statistic_df = gdf['average_cycle_time'].agg(['mean','median','count']).reset_index()
        pass_df = gdf['pass'].sum().reset_index(name="pass_count")
        fail_df = gdf['fail'].sum().reset_index(name="error_count")
        
        agg_df = statistic_df.merge(pass_df, on=key_cols, how='left')
        agg_df = agg_df.merge(fail_df, on=key_cols, how='left')
        return agg_df
    
    def create_ratio_features(self, data_before):
        data_after = data_before.copy()
        data_after['error_ratio'] = (data_after['error_count'] / data_after['count'])
        data_after['pass_ratio'] = 1 - data_after['error_ratio']
        return data_after

    def rounding(self, data_before):
        data_after = data_before.copy()
        data_after = data_after.round({
            'mean': 1,
            'median': 1,
            'error_ratio': 2,
            'pass_ratio': 2,
        })
        return data_after
    
    def create_features(self, data_mart, key_cols=None):
        df = data_mart.copy()
        
        # 제조 데이터의 시간 정보 변환
        df = self.create_time_features(df)
        
        # 평균 생산시간 계산
        df = self.create_cycletime_feature(df)
        
        # 정상품, 불량품 생성
        df = self.create_pass_fail_feature(df)
        
        # 그룹(금형코드, 시간)별 집계 (평균생산시간, 평균생산량, 전체횟수, 성공횟수, 실패횟수)
        features = self.aggregate(df, key_cols)

        # 불량률, 정상률 생성
        features = self.create_ratio_features(features)

        # 자릿수 지정 (반올림)
        features = self.rounding(features)
        
        return features
