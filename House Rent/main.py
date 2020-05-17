import uvicorn
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile, Form
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

saved_model = 'House Rent/xgb_model.pkl'
with open(saved_model, 'rb') as file:
    Pickled_XGB_Model = pickle.load(file)


@app.route("/")
def welcome():
    return "Welcome all to the project"


@app.post(
    "/singlePredict/{region,type,sqfeet,beds,baths,cats_allowed,dogs_allowed,smoking_allowed,wheelchair_access,electric_vehicle_charge,comes_furnished,laundry_options,parking_options,lat,long}")
def predict(region: str, type: str, sqfeet: float, beds: int, baths: int, cats_allowed: int, dogs_allowed: int,
            smoking_allowed: int, wheelchair_access: int, electric_vehicle_charge: int, comes_furnished: int,
            laundry_options: str, parking_options: str, lat: float, long: float):
    labels = ['region', 'type', 'sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
              'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished', 'laundry_options', 'parking_options',
              'lat', 'long']
    features = [[region, type, sqfeet, beds, baths, cats_allowed, dogs_allowed, smoking_allowed, wheelchair_access, electric_vehicle_charge, comes_furnished, laundry_options, parking_options, lat, long]]
    to_predict = pd.DataFrame(features, columns=labels)
    le = LabelEncoder()
    to_predict['region'] = le.fit_transform(to_predict['region'])
    to_predict['laundry_options'] = le.fit_transform(to_predict['laundry_options'])
    to_predict['parking_options'] = le.fit_transform(to_predict['parking_options'])
    to_predict['type'] = le.fit_transform(to_predict['type'])
    values = Pickled_XGB_Model.predict(to_predict)
    return {'Output': int(values[0])}


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
