
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler

def rf_evaluations(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    rfr_clf =RandomForestRegressor()
    rfr_clf.fit(x_train, y_train)
    y_pred_test = rfr_clf.predict(x_test)

    print("RF_MEAN ABSOLUTE SCORE:-", mean_absolute_error(y_test, y_pred_test))
  

    MAE=mean_absolute_error(y_test, y_pred_test)


    return MAE


