import numpy as np
import pandas as pd

def read_csv( path ):
    df = pd.read_csv( path, header=0, delimiter="," )
    return df

if __name__ == "__main__":
  print()
  dataframe_train = read_csv( "cv-valid-train.csv" )
  dataframe_test = read_csv( "cv-valid-test.csv" )
  dataframe_gender_train = dataframe_train[dataframe_train["gender"].notnull()]
  dataframe_gender_test = dataframe_test[dataframe_test["gender"].notnull()]

  dataframe_gender_train.to_csv( "dataframe_gender_train.csv", columns=["filename", "gender"])
  dataframe_gender_test.to_csv( "dataframe_gender_test.csv", columns=["filename", "gender"])
