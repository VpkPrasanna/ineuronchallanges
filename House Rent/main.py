import uvicorn
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import LabelEncoder

app = FastAPI()
saved_model = 'House Rent/xgb_model.pkl'
with open(saved_model, 'rb') as file:
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


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    dataFrame = pd.read_csv(file.file)
    le = LabelEncoder()
    dataFrame['region'] = le.fit_transform(dataFrame['region'])
    dataFrame['laundry_options'] = le.fit_transform(dataFrame['laundry_options'])
    dataFrame['parking_options'] = le.fit_transform(dataFrame['parking_options'])
    dataFrame['type'] = le.fit_transform(dataFrame['type'])
    prediction = Pickled_XGB_Model.predict(dataFrame)
    return {"predicted": str(prediction)}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
