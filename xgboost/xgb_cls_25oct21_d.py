# Jesus is my Saviour!

import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

#conda install -c anaconda py-xgboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV    

import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\DataSets1')

rain = pd.read_csv("weatherAUS.csv")
rain.head()
rain.info()

# target var = RainToday
#Dropping the Rainfall column is a must because it records the amount of rain in millimeters.
cols_to_drop = ["Date", "Location", "RainTomorrow", "Rainfall"]
rain.drop(cols_to_drop, axis=1, inplace=True)

#____missing values/proportions
missing_props = rain.isna().mean(axis=0)
missing_props

#If the proportion is higher than 40% we will drop the column

over_threshold = missing_props[missing_props >= 0.4]
over_threshold

rain.drop(over_threshold.index, axis=1, inplace=True)

# X and y
X = rain.drop("RainToday", axis=1)
y = rain.RainToday

# categorical vars, impute missing vals by mode=
# 'most_frequent', and encode them (one-hot-encoding)

'''
learners need to learn about PIPELINES
here is a good example
'''
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

# for continuous/numeric vars, we will standardize 
# and impute by median

from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")), 
           ("scale", StandardScaler())]
)

# after making pipelines, lets separte cat and cont vars
cat_cols = X.select_dtypes(exclude="number").columns
num_cols = X.select_dtypes(include="number").columns

'''
now see how we are using pipelines for
transforming catg and cont vars. its 
indeed very interesting and easy and making our 
task sooooo easy (in one shot!)
'''

# first we make ColumnTransformer
# which will transform catg+numeric vars
# by our pipelines
from sklearn.compose import ColumnTransformer

full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ]
)

# now we will apply on X (predictors)
X_processed = full_processor.fit_transform(X)
# see, now there are 60 columns (predictors!), X was 15 only!
# in one shot, all catg and cont vars treated!!! smile!!!!!

##_______________oh!, y is having null values, 
## we will impute by 'mode'
## and it is in series form
## which should be in array form, so we will reshape it

y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
              y.values.reshape(-1, 1))

# meaning of reshape(-1,1)
'''
our y is a series, which is transformed into an array
bcz, if you see, X_processed is an array, so y should also
be in array form.
Look at the following example-
''' 
k = [11, 12, 34, 45] # LIST
k # a list
'''
k
Out[50]: [11, 12, 34, 45]
'''
ks = pd.Series(k) # from list to Series
ks # a series
'''
ks
Out[54]: 
0    11
1    12
2    34
3    45
dtype: int64
'''
ks = ks.values.reshape(-1,1)
ks # now an array
'''
ks
Out[49]: 
array([[11],
       [12],
       [34],
       [45]], dtype=int64)'''

##________$$$
# so far we have done (i) missing value treatment
# (ii) one hot encoding of catg vars
# (iii) reshaping of y as an array

#______TIME TO GO FOR XGB MODELING!

# first thing first, train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, stratify=y_processed, random_state=1121218
)

y.value_counts() #no=1,10,319;; yes=31,880, too imbalanced
# that's why stratify used! 

#________building model
from sklearn.metrics import accuracy_score
# Init classifier
xgb_cl = xgb.XGBClassifier()
# Fit
xgb_cl.fit(X_train, y_train)
# Predict
preds = xgb_cl.predict(X_test)
# Score
accuracy_score(y_test, preds)
0.8507080984463082

#__________________________________

'''
GridSearch may take 
huge time, hence avoided in this demo
'''










