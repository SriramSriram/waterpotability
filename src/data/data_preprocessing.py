import pandas as pd
import numpy as np
import os

train_data= pd.read_csv(r"E:\SRM_Projects\ML_pipeline_WP\data\raw\train.csv")
test_data= pd.read_csv(r"E:\SRM_Projects\ML_pipeline_WP\data\raw\test.csv")

def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value= df[column].mean()
            df[column].fillna(mean_value,inplace=True)
    return df


train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

data_path=os.path.join("data","processed")

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,"train_processed_mean.csv"),index=False)
test_processed_data.to_csv(os.path.join(data_path,"test_processed_mean.csv"),index=False)