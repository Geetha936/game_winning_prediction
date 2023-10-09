import numpy as np
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler


def xgboost_evaluations(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    xgb_clf = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_clf.fit(x_train, y_train)
    y_pred_test = xgb_clf.predict(x_test)

    print("RF_MEAN ABSOLUTE SCORE:-", mean_absolute_error(y_test, y_pred_test))

    MAE = mean_absolute_error(y_test, y_pred_test)

    return MAE
