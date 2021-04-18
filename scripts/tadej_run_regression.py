import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


def predict():
    CAT_FEATURES = ['gender', 'region', 'city', 'education']

    try:
        model = CatBoostRegressor().load_model('models/tadej_model.cbm', 'cbm')
    except:
        model = CatBoostRegressor().load_model('tadej_model.cbm', 'cbm')
    
    data = pd.read_pickle('final_version.pickle')
    data_pool = Pool(data = data, cat_features=CAT_FEATURES)

    THRESHOLD = -2.3
    target = (model.predict(data_pool) > THRESHOLD).astype(int)
    submission = pd.DataFrame({'client_id': data.index, 'target': target})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    predict()