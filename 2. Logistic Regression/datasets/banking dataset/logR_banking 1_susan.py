# Jesus is my Saviour!

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_column',None)
 

data = pd.read_csv("C:/Users/Dr Vinod/Desktop/banking.csv")
data.isnull().sum() # no nulls
data = data.dropna()
print(data.shape) # 41,188
print(list(data.columns))

data.info()
data.sample(10)

# education has more categories, must reduce
data['education'].unique()
'''
#Let us group "basic.4y", "basic.9y" 
and "basic.6y" together and 
call them "basic".
'''
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])


data['education'].unique()

# balancing in y; 88:11; later done by smote

# mean across other vars
data.groupby('y').mean()
data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()
#________post completing good eda and categories minimization
# making dummy vars
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], drop_first=True, prefix=var)
    data1=data.join(cat_list)
    data=data1
    
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars] # keep only those which are not in the list of data_vars
data_final1=data[to_keep]
data_final1.columns.values # 52 are there, 10 catg gone!


data_final1.to_csv("C:/Users/Dr Vinod/Desktop/data_final1.csv")
#______smote
X = data_final1.loc[:, data_final1.columns != 'y']
y = data_final1.loc[:, data_final1.columns == 'y']

y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# oversampling to be done in TRAIN DATA ONLY

# 1st join x_train and y_train
train = X_train.join(y_train)

train.info()

not_subsc = train[train.y == 0] 
len(not_subsc) #25,567
subsc = train[train.y == 1] 
len(subsc) # 3264

# 2nd, upsample; minor catg 'subsc' to be incraesed to counts = not_subsc
from sklearn.utils import resample
subsc_os = resample(subsc,
                          replace=True, # sample with replacement
                          n_samples=len(not_subsc), # match number in majority class
                          random_state=27) # reproducible results

train_os = pd.concat([not_subsc, subsc_os]) 

train_os.y.value_counts()
'''
1    25567
0    25567
Name: y, dtype: int64
'''
# 3rd,  make x_trainos, y_trainos
X_trainos1 = train_os.loc[:, train_os.columns != 'y']
y_trainos1 = train_os.loc[:, train_os.columns == 'y']

#______________________________________________________________________
# RECURSIVE FEATURE ELIMINATION 
X_trainos1.to_csv("C:/Users/Dr Vinod/Desktop/X_trainos1.csv")
y_trainos1.to_csv("C:/Users/Dr Vinod/Desktop/y_trainos1.csv")

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_column',None)
from imblearn.over_sampling import SMOTE 


#X_trainos1 = pd.read_csv("C:/Users/Dr Vinod/Desktop/X_trainos1.csv")
#y_trainos1 = pd.read_csv("C:/Users/Dr Vinod/Desktop/y_trainos1.csv")
#__________________________________________________________________
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, n_features_to_select=20)
rfe = rfe.fit(X_trainos1, y_trainos1.values.ravel())

rfe.n_features_to_select
X_trainos1.columns[rfe.get_support()] # 20 
'''
Index(['previous', 'euribor3m', 'job_retired', 'job_student', 'job_unknown',
       'education_high.school', 'education_university.degree',
       'education_unknown', 'default_unknown', 'month_aug', 'month_jul',
       'month_jun', 'month_mar', 'month_may', 'month_oct', 'month_sep',
       'day_of_week_mon', 'day_of_week_wed', 'poutcome_nonexistent',
       'poutcome_success'],
      dtype='object')
'''

cols = X_trainos1.columns[rfe.get_support()] # 20 
cols.to_list() # to see easily
#________________________________sm model to see p_values 
x1 = X_trainos1[cols] # 51134
x1.info()

y_trainos1.info()
y_trainos1.y.value_counts()

y1 = y_trainos1.y # 51134

import statsmodels.api as sm
logit_model=sm.Logit(y1,x1)
result=logit_model.fit()
print(result.summary2())

#_____________now sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(solver= 'sag')
logreg.fit(x1, y1)


# x_test should also have 20 features! 

X_test20 = X_test[cols]

y_pred = logreg.predict(X_test20)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test20, y_test)))
#0.79

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#_________below run in block
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test20))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test20)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#_______above in one block 






