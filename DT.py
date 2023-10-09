
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler

def dt_evaluations(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    dtr_clf =DecisionTreeRegressor()
    dtr_clf.fit(x_train, y_train)
    y_pred_test = dtr_clf.predict(x_test)

    print("DT_MEAN ABSOLUTE SCORE:-", mean_absolute_error(y_test, y_pred_test))


    MAE=mean_absolute_error(y_test, y_pred_test)
   

    return MAE


