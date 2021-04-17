import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def predict():
    CAT_FEATURES = ['gender', 'region', 'city', 'education']

    try:
        model = CatBoostClassifier().load_model('models/tadej_model.cbm', 'cbm')
    except:
        model = CatBoostClassifier().load_model('tadej_model.cbm', 'cbm')
    
    data = pd.read_pickle('data/final_version.pickle')
    data_pool = Pool(data = data, cat_features=CAT_FEATURES)

    target = model.predict(data_pool)
    submission = pd.DataFrame({'client_id': data.index, 'target': target})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    predict()