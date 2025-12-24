import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

train_data= pd.read_csv(r"E:\SRM_Projects\ML_pipeline_WP\data\processed\train_processed.csv")
#X_train= train_data.iloc[:,0:-1].values
#Y_train= train_data.iloc[:,-1].values

X_train= train_data.drop(columns=["Potability"],axis=1)
Y_train= train_data["Potability"]

n_estimators=yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]

model= RandomForestClassifier(n_estimators=n_estimators)

model.fit(X_train,Y_train)

pickle.dump(model,open("model.pkl","wb"))
