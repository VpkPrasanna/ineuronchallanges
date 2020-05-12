import uvicorn
import pandas as pd

import pickle

from fastapi import FastAPI
import numpy as np

app = FastAPI()
saved_model = 'xgb_model.pkl'
with open('xgb_model.pkl', 'rb') as file:
    Pickled_XGB_Model = pickle.load(file)


@app.get("/predict")
def predict():
    # X_test = [['region', 'type', 'sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
    #            'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished', 'laundry_options', 'parking_options',
    # 'lat', 'long'],
    # [13, 11, 887, 2, 1.0, 0, 0, 1, 0, 0, 0, 1, 1, 35.3384, -119.063]]
    # column_names = X_test.pop(0)
    # datas = pd.DataFrame(X_test, columns=column_names)
    labels = ['region', 'type', 'sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
              'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished', 'laundry_options', 'parking_options',
              'lat', 'long']
    features = [[13, 11, 887, 2, 1.0, 0, 0, 1, 0, 0, 0, 1, 1, 35.3384, -119.063]]
    to_predict = pd.DataFrame(features, columns=labels)
    print(to_predict)
    values = Pickled_XGB_Model.predict(to_predict)
    print(values)
    return {'Output': values}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
