
import numpy as np
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,MinMaxScaler

def lgbm_evaluations(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    lgbm_clf =LGBMRegressor(random_state=42, n_jobs=1)
    lgbm_clf.fit(x_train, y_train)
    y_pred_test = lgbm_clf.predict(x_test)

    print("LGBM_MEAN ABSOLUTE SCORE:-", mean_absolute_error(y_test, y_pred_test))
   

    MAE=mean_absolute_error(y_test, y_pred_test)

    return MAE


