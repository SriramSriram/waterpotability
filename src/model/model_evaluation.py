import pandas as pd
import numpy as np
import os
import pickle

import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

test_data= pd.read_csv(r"E:\SRM_Projects\ML_pipeline_WP_CC\waterpotability\data\processed\test_processed_mean.csv")
X_test= test_data.iloc[:,0:-1].values
Y_test= test_data.iloc[:,-1].values

model= pickle.load(open("model.pkl","rb"))

Y_pred= model.predict(X_test)

acc=accuracy_score(Y_test,Y_pred)
pre=precision_score(Y_test,Y_pred)
f1=f1_score(Y_test,Y_pred)
rec= recall_score(Y_test,Y_pred)

metrics_dict= {
    "accuracy_score":acc,
    "precision_score":pre,
    "F1_score": f1,
    "recall_score":rec
}

with open("metrics.json","w") as file:
    json.dump(metrics_dict,file,indent=4)

