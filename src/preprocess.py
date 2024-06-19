import numpy as np
import pandas as pd


class Preprocess:
    def __init__(self, conf):
        self.key_cols = ["registration_time","count"]
        self.conf = conf
        pass

    def normalize_column_name(self, data: pd.DataFrame) -> pd.DataFrame:
        data_cols = list(map(lambda s: s.lower(), data.columns))
        data.columns = data_cols
        return data

    def remove_trivial_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        rm_cols = [
            'line',
            'name',
            'mold_name',
            'time',
            'date',
            'tryshot_signal',
            'working',
            'emergency_stop',
        ]
        rm_cols = [c for c in rm_cols if c in data.columns]
        try:
            data = data.drop(columns=rm_cols)
        except KeyError:
            pass
        return data

    def drop_duplicated_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop_duplicates(subset=self.key_cols, keep='first')
        return data

    @property
    def data_type_dict(self) -> dict:
        type_dict = {
            'registration_time': pd.to_datetime,
            'count': np.vectorize(lambda x: int(x)),
            'mold_code': np.vectorize(lambda x: "CODE_" + str(x))
        }
        return type_dict
    
    def data_type_casting(self, data: pd.DataFrame) -> pd.DataFrame:
        def as_float(ser): return ser.astype('float', errors='ignore')
        type_dict = self.data_type_dict
        for c in data.columns:
            func = type_dict.get(c, as_float)
            data[c] = func(data[c])
        return data

    def sort_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values(self.key_cols)
        return data

    def mask_abnormal_values(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            normal_range_dict = self.conf.normal_range_dict
            for c in data:
                normal_range = normal_range_dict.get(data[c].name, None)
                if normal_range is None:
                    continue
                cond = data[c].between(*sorted(normal_range), inclusive='both')
                data[c] = data[c].mask(~cond)
        except TypeError as err:
            print(c)
            print(err)
        return data

    def drop_trivial_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        cond = data['y_label'].notnull()  # 'passorfail' 대신 'y_label' 사용
        data = data.loc[cond]
        return data

    def create_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        data['hour'] = data['registration_time'].apply(lambda t: t.hour)
        return data

    def preprocess_data(self, data_before: pd.DataFrame) -> pd.DataFrame:
        data_after = data_before.copy()
        data_after = self.normalize_column_name(data_after)
        data_after = self.remove_trivial_columns(data_after)
        data_after = self.drop_duplicated_rows(data_after)
        data_after = self.data_type_casting(data_after)
        data_after = self.sort_data(data_after)
        data_after = self.mask_abnormal_values(data_after)
        data_after = self.drop_trivial_rows(data_after)
        return data_after
