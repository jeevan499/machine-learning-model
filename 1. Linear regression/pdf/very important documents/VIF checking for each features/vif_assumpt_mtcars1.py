# Jesus is my saviour!
#Jesus is my Saviour! 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
pd.set_option('display.max_column',None)
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/Dr Vinod/Desktop/DataSets1/mtcars.csv")
data.info()

# correlations
dfc = data.iloc[:, [1,2,3,4]]
dfc.info()
dfc.corr()
dfc.corr()['mpg'][:] # so good!


#_____checking assumptions
#_______lets build a small model

model=smf.ols(formula='mpg ~ disp + hp + drat + wt',data=data).fit()
print(model.summary())

'''
print(model.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.838
Model:                            OLS   Adj. R-squared:                  0.814
Method:                 Least Squares   F-statistic:                     34.82
Date:                Wed, 29 Sep 2021   Prob (F-statistic):           2.70e-10
Time:                        09:37:51   Log-Likelihood:                -73.292
No. Observations:                  32   AIC:                             156.6
Df Residuals:                      27   BIC:                             163.9
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     29.1487      6.294      4.631      0.000      16.235      42.062
disp           0.0038      0.011      0.353      0.727      -0.018       0.026
hp            -0.0348      0.012     -2.999      0.006      -0.059      -0.011
drat           1.7680      1.320      1.340      0.192      -0.940       4.476
wt            -3.4797      1.078     -3.227      0.003      -5.692      -1.267
==============================================================================
Omnibus:                        5.267   Durbin-Watson:                   1.736
Prob(Omnibus):                  0.072   Jarque-Bera (JB):                4.327
Skew:                           0.899   Prob(JB):                        0.115
Kurtosis:                       3.102   Cond. No.                     4.26e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.26e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''
# read the note [2], indicating multicollinearity....must check vif

predict = model.predict() # these are predicted values, called y hat also
residuals = data.mpg - predict # these are errors

# add predict to your data
data['predict'] = predict # now you have a new column in your data
data.info() # look at 12th column
'''
RangeIndex: 32 entries, 0 to 31
Data columns (total 13 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  32 non-null     object 
 1   mpg         32 non-null     float64
 2   cyl         32 non-null     int64  
 3   disp        32 non-null     float64
 4   hp          32 non-null     int64  
 5   drat        32 non-null     float64
 6   wt          32 non-null     float64
 7   qsec        32 non-null     float64
 8   vs          32 non-null     int64  
 9   am          32 non-null     int64  
 10  gear        32 non-null     int64  
 11  carb        32 non-null     int64  
 12  predict     32 non-null     float64 
dtypes: float64(6), int64(6), object(1)
memory usage: 3.4+ KB

'''
# add residuals to your data
data['residuals'] = residuals # new column added

# create a new column as obsno [observation number]
obs = np.arange(32) # a numpy array carrying nos from 0 to 31 (last no will be ignored)
obs # you can see nos from 0 to 31

obsno = pd.DataFrame(obs) # convert array into a data frame 

data['obsno'] = obsno # craete a new variable/column 'obsno' in yr data

a = np.arange(10)
a
b = a+1
b

data.info() 
'''
Data columns (total 15 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  32 non-null     object 
 1   mpg         32 non-null     float64
 2   cyl         32 non-null     int64  
 3   disp        32 non-null     float64
 4   hp          32 non-null     int64  
 5   drat        32 non-null     float64
 6   wt          32 non-null     float64
 7   qsec        32 non-null     float64
 8   vs          32 non-null     int64  
 9   am          32 non-null     int64  
 10  gear        32 non-null     int64  
 11  carb        32 non-null     int64  
 12  predict     32 non-null     float64
 13  residuals   32 non-null     float64
 14  obsno       32 non-null     int32  
dtypes: float64(7), int32(1), int64(6), object(1)
memory usage: 3.8+ KB
'''
# look at 12, 13, 14 column nos ! wow!!!!!!!!!!!! you have done it!

#_______________________now ready for checking assumptions

# 1 _Normality
'''
make histogram of residuals 
and check whether its approximately
bell-shaped, symmetrical
'''

# 2_Linearity
'''
make scatter plots,
x = each continuous predictor [disp + hp + drat + wt]
one by one; 
y = response variable [mpg]

see whether a linear relationship is visible?
'''

# 3_Independence of observation
'''
make scatter plot,
x = obsno 
y = residuals

if, no geometric pattern is visible,
assumption of INDEPENDENCE OF ERROR is holding good and 
not violated
'''

# 4_Constant Error Variance [homoscadasticity]
'''
make scatter plot,
x = predict
y = residuals

if, no geometric pattern is visible,
assumption of homoscadasticity is holding good and 
not violated
'''
# 5 Durbin Watson Statistics
'''
in this model DWS = 1.736
create a scale from 0-4 
as discussed in class and take a call
'''

# 6_vif [this you have already learnt]

#___________________________________below was shared 28/9/21

from statsmodels.stats.outliers_influence import variance_inflation_factor

# first put your predictors in x
x = data.iloc[:, [2,3,4,5,6,7,8,9,10,11]]
x.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32 entries, 0 to 31
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   cyl     32 non-null     int64  
 1   disp    32 non-null     float64
 2   hp      32 non-null     int64  
 3   drat    32 non-null     float64
 4   wt      32 non-null     float64
 5   qsec    32 non-null     float64
 6   vs      32 non-null     int64  
 7   am      32 non-null     int64  
 8   gear    32 non-null     int64  
 9   carb    32 non-null     int64  
dtypes: float64(4), int64(6)
memory usage: 2.6 KB

'''
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
  
print(vif_data)
'''
print(vif_data)
  feature         VIF
0     cyl  112.629828
1    disp   98.930791
2      hp   56.047781
3    drat  132.214353
4      wt  182.948049
5    qsec  317.534376
6      vs    8.752581
7      am    7.412020
8    gear  119.804879
9    carb   32.213836
'''

