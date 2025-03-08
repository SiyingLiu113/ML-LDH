import xgboost
from xgboost import XGBRegressor
from sklearn.tree import  DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
import pickle
import joblib

data_final=pd.read_csv("./DATA_LDH.csv")  
data=data_final.iloc[:,:-1]
target=data_final.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.2, random_state=2)


transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)

X_train=pd.DataFrame(X_train,columns=['ρ Mg$^{2+}$','ρ Li$^{+}$','Z','χ','Valence','Rion','Ei','Nve','M/Al','d','Li/Al'])
X_test=pd.DataFrame(X_test,columns=['ρ Mg$^{2+}$','ρ Li$^{+}$','Z','χ','Valence','Rion','Ei','Nve','M/Al','d','Li/Al'])

joblib.dump(transfer, 'transfer.pkl')


# Train and evaluate the model and save the results
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    
    model.fit(X_train, y_train)

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    MSE_train=mean_squared_error(y_train,y_train_predict)
    MSE_test=mean_squared_error(y_test,y_test_predict)
    r2_score_train=r2_score(y_train,y_train_predict)
    r2_score_test=r2_score(y_test,y_test_predict)


    print("-------------------------\n")
    print("-------------------------\n")
    print(f"MODEL: {model_name}")
    print("TRAIN_MSE:\n",MSE_train)
    print("TEST_MSE:\n",MSE_test)
    print("TRAIN_R2:\n",r2_score_train)
    print("TEST_R2:\n",r2_score_test)
    print("-------------------------\n")
    print("-------------------------\n")


    np.savetxt(f"{model_name}_y_train.csv", y_train, delimiter=",")
    np.savetxt(f"{model_name}_y_test.csv", y_test, delimiter=",")
    np.savetxt(f"{model_name}_y_train_predict.csv", y_train_predict, delimiter=",")
    np.savetxt(f"{model_name}_y_test_predict.csv", y_test_predict, delimiter=",")

    joblib.dump(model, f"{model_name}.pkl")

    return model


svr_model = SVR(gamma=0.01)
train_and_evaluate_model(svr_model, X_train, y_train, X_test, y_test, "SVR")

dt_model = DecisionTreeRegressor(max_depth=7)
train_and_evaluate_model(dt_model, X_train, y_train, X_test, y_test, "DT")

xgb_model = XGBRegressor(reg_alpha=1, max_depth=3)
train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGB")

rf_model = RandomForestRegressor(criterion="friedman_mse",max_depth=8)
train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, "RF")

ada_model = AdaBoostRegressor(n_estimators=120)
train_and_evaluate_model(ada_model, X_train, y_train, X_test, y_test, "Ada")

gb_model = GradientBoostingRegressor(n_estimators=80)
train_and_evaluate_model(gb_model, X_train, y_train, X_test, y_test, "GB")