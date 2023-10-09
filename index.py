
from flask import Flask, render_template, request,flash
import pandas as pd
from flask import session
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from RF import rf_evaluations
from LR import lr_evaluations
from LGBM import lgbm_evaluations
from DT import dt_evaluations
from XGBoost import xgboost_evaluations
import numpy as np



app = Flask(__name__)
app.secret_key = "abc"


dict={}




@app.route('/')
def index():
    return render_template('index.html')



@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/rank_prediction", methods =["GET", "POST"])
def rank_prediction():
    file = request.form.get('file')
    data_train = pd.read_csv('../Rank_Prediction/dataset/trainpubg.csv')

    y_train = data_train['winPlacePerc']

    del data_train['winPlacePerc']

    x_train = data_train

    lg_model = LGBMRegressor(random_state=42, n_jobs=1)

    lg_model.fit(x_train, y_train)

    data_test = pd.read_csv('../Rank_Prediction/dataset/'+file)
    testdata = np.array(data_test)
    testdata = testdata.reshape(len(testdata), -1)

    # test=[[0,0,0,0,0,0,60,1241,0,0,0,1306,28,26,-1,0,0,0,0,0,0,244.8,1,1466]]
    print(lg_model.predict(testdata))

    prediction_res=lg_model.predict(testdata)

    res=prediction_res[0]
    print(res)

    return render_template("prediction.html", result=res)



@app.route("/evaluations")
def evaluations():

    lr_list=[]
    lr_list.clear()

    rf_list = []
    rf_list.clear()

    dt_list = []
    dt_list.clear()

    xgbst_list = []
    xgbst_list.clear()


    lgbm_list = []
    lgbm_list.clear()

    metrics = []
    metrics.clear()

    data_train = pd.read_csv('../Rank_Prediction/dataset/trainpubg.csv')

    y = data_train['winPlacePerc']

    del data_train['winPlacePerc']

    X = data_train

    # Split train test: 70 % - 30 %
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)


    MAE_lr = lr_evaluations(X_train, X_test, y_train, y_test)

    MAE_rf = rf_evaluations(X_train, X_test, y_train, y_test)


    MAE_dt = dt_evaluations(X_train, X_test, y_train, y_test)

    MAE_xgboost = xgboost_evaluations(X_train, X_test, y_train, y_test)

    MAE_lgbm = lgbm_evaluations(X_train, X_test, y_train, y_test)


    lr_list.append("Linear Regression")
    lr_list.append(MAE_lr)

    rf_list.append("Random Forest")
    rf_list.append(MAE_rf)

    dt_list.append("Decision Tree")
    dt_list.append(MAE_dt)

    xgbst_list.append("XG Boost")
    xgbst_list.append(MAE_xgboost)

    lgbm_list.append("LightGBM")
    lgbm_list.append(MAE_lgbm)

    metrics.clear()
    metrics.append(lr_list)
    metrics.append(rf_list)
    metrics.append(dt_list)
    metrics.append(xgbst_list)
    metrics.append(lgbm_list)



    return render_template("evaluations.html", evaluations=metrics)






if __name__ == '__main__':
    app.run(host="localhost", port=2487, debug=True)
