import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


def predict():
    CAT_FEATURES = ['gender', 'region', 'city', 'education', 'mcc_cd']

    try:
        model = CatBoostRegressor().load_model('models/ana_model.cbm', 'cbm')
    except:
        model = CatBoostRegressor().load_model('ana_model.cbm', 'cbm')
    
    data = pd.read_pickle('final_version.pickle')
    data_pool = Pool(data = data, cat_features=CAT_FEATURES)

    target = (model.predict(data_pool) > 0).astype(int)
    submission = pd.DataFrame({'client_id': data.index, 'target': target})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    predict()