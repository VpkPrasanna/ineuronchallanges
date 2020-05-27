import uvicorn
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

model_name = "Banking subscription/Models/random_forest.pkl"
with open(model_name, 'rb') as file:
    random_forest_model = pickle.load(file)


# age	job	marital	education	default	housing	loan	contact	month	day_of_week	duration	campaign	pdays	previous	poutcome	emp.var.rate	cons.price.idx	cons.conf.idx	euribor3m	nr.employed
@app.post(
    "/predict/{age,job,marital,education,default,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed}")
def predict(age: int, job: str, marital: str, education: str, default: str, housing: str, loan: str, contact: str,
            month: str, day_of_week: str, duration: int, campaign: int, pdays: int, previous: int, poutcome: str,
            emp_var_rate: float, cons_price_idx: float, cons_conf_idx: float, euribor3m: float, nr_employed: float):
    labels = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
              "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx",
              "cons.conf.idx", "euribor3m", "nr.employed"]
    features = [
        [age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays,
         previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]]
    test_data = pd.DataFrame(features, columns=labels)
    le = LabelEncoder()
    test_data["job"] = le.fit_transform(test_data["job"])
    test_data["marital"] = le.fit_transform(test_data["marital"])
    test_data["education"] = le.fit_transform(test_data["education"])
    test_data["default"] = le.fit_transform(test_data["default"])
    test_data["housing"] = le.fit_transform(test_data["housing"])
    test_data["loan"] = le.fit_transform(test_data["loan"])
    test_data["contact"] = le.fit_transform(test_data["contact"])
    test_data["month"] = le.fit_transform(test_data["month"])
    test_data["day_of_week"] = le.fit_transform(test_data["day_of_week"])
    test_data["poutcome"] = le.fit_transform(test_data["poutcome"])
    output = random_forest_model.predict(test_data)
    print(output)
    return {"final_answer": str(output)}


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    dataframe = pd.read_csv(file.file)
    le = LabelEncoder()
    dataframe["job"] = le.fit_transform(dataframe["job"])
    dataframe["marital"] = le.fit_transform(dataframe["marital"])
    dataframe["education"] = le.fit_transform(dataframe["education"])
    dataframe["default"] = le.fit_transform(dataframe["default"])
    dataframe["housing"] = le.fit_transform(dataframe["housing"])
    dataframe["loan"] = le.fit_transform(dataframe["loan"])
    dataframe["contact"] = le.fit_transform(dataframe["contact"])
    dataframe["month"] = le.fit_transform(dataframe["month"])
    dataframe["day_of_week"] = le.fit_transform(dataframe["day_of_week"])
    dataframe["poutcome"] = le.fit_transform(dataframe["poutcome"])
    output = random_forest_model.predict(dataframe)
    return {"final_ansewer": str(output)}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
