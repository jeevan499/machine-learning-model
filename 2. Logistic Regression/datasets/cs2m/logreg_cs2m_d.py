# Jesus is my Saviour!
import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\WD_python')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('cs2m.csv')
df.shape
df.head()
df.tail()
df.info()
import statsmodels.api as sm
import statsmodels.formula.api as smf
formula = 'DrugR ~ BP+Age+Chlstrl+Prgnt+AnxtyLH'
model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

predictions = result.predict() # 30 probabilities 
predictions_nominal = [0 if x < 0.5 else 1 for x in predictions]

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df["DrugR"], predictions_nominal))
print(classification_report(df["DrugR"], predictions_nominal, digits = 3))

#____________________roc curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

x = df.iloc[:, [0, 1, 2, 3, 4]] 
y = df.iloc[:, 5]
clf_reg = LogisticRegression()
clf_reg.fit(x, y)
y_score = clf_reg.predict_proba(x)[:,1]
false_positive_rate, true_positive_rate, threshold = roc_curve(y, y_score)
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y, y_score))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()