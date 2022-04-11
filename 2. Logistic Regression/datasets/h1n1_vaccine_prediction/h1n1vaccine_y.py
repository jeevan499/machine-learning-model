# Jesus is my Saviour!
import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\WD_python')
import pandas as pd 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('h1n1_vaccine_prediction.csv') #26,707
df.info()

# droping all will give 11k only 50% values 
dfnomissing = df.dropna()
dfnomissing.info()
df_vac = pd.read_csv('dfc_h1n1.csv') # 22,976
df_vac.info()

#h1n1_worry
#______histogram
#_run in block
plt.hist(df_vac.h1n1_worry, bins = 'auto', facecolor = 'red')
plt.xlabel('h1n1_worry')
plt.ylabel('counts')
plt.title('Histogram of h1n1_worry') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['h1n1_worry'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.h1n1_worry.isnull().sum() #0 Missing values
df_vac.h1n1_worry.value_counts() 
'''
2    9181
1    7181
3    3858
0    2756'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ h1n1_worry', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#7.331262e-92 ie p_value is <0.05; Ho Reject; Good Predictor

#h1n1_awareness
#______histogram
#_run in block
plt.hist(df_vac.h1n1_awareness, bins = 'auto', facecolor = 'red')
plt.xlabel('h1n1_awareness')
plt.ylabel('counts')
plt.title('Histogram of h1n1_awareness') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['h1n1_awareness'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.h1n1_awareness.isnull().sum() #0 Missing values
df_vac.h1n1_awareness.value_counts() 
'''
1    12662
2     8408
0     1906'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ h1n1_awareness', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#3.350592e-78 ie  p_value is <0.05; Ho Reject; Good Predictor

#antiviral_medication
#______histogram
#_run in block
plt.hist(df_vac.antiviral_medication, bins = 'auto', facecolor = 'red')
plt.xlabel('antiviral_medication')
plt.ylabel('counts')
plt.title('Histogram of antiviral_medication') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['antiviral_medication'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers

df_vac.antiviral_medication.isnull().sum() #0 Missing values
df_vac.antiviral_medication.value_counts() 
'''
0    21843
1     1133'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ antiviral_medication', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#4.740257e-09 ie p_value is <0.05; Ho Reject; Good Predictor

#contact_avoidance
#______histogram
#_run in block
plt.hist(df_vac.contact_avoidance, bins = 'auto', facecolor = 'red')
plt.xlabel('contact_avoidance')
plt.ylabel('counts')
plt.title('Histogram of contact_avoidance') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['contact_avoidance'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.contact_avoidance.isnull().sum() #0 Missing values
df_vac.contact_avoidance.value_counts() 
'''
1    16830
0     6146'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ contact_avoidance', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#2.021171e-13 ie p_value is <0.05; Ho Reject; Good Predictor

#bought_face_mask
#______histogram
#_run in block
plt.hist(df_vac.bought_face_mask, bins = 'auto', facecolor = 'red')
plt.xlabel('bought_face_mask')
plt.ylabel('counts')
plt.title('Histogram of bought_face_mask') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['bought_face_mask'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers

df_vac.bought_face_mask.isnull().sum() #0 Missing values
df_vac.bought_face_mask.value_counts() 
'''
0    21409
1     1567'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ bought_face_mask', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#3.276427e-27 ie p_value is <0.05; Ho Reject; Good Predictor

#wash_hands_frequently
#______histogram
#_run in block
plt.hist(df_vac.wash_hands_frequently, bins = 'auto', facecolor = 'red')
plt.xlabel('wash_hands_frequently')
plt.ylabel('counts')
plt.title('Histogram of wash_hands_frequently') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['wash_hands_frequently'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.wash_hands_frequently.isnull().sum() #0 Missing values
df_vac.wash_hands_frequently.value_counts() 
'''
1    19088
0     3888'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ wash_hands_frequently', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#4.778962e-29 ie p_value is <0.05; Ho Reject; Good Predictor

#avoid_large_gatherings
#______histogram
#_run in block
plt.hist(df_vac.avoid_large_gatherings, bins = 'auto', facecolor = 'red')
plt.xlabel('avoid_large_gatherings')
plt.ylabel('counts')
plt.title('Histogram of avoid_large_gatherings') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['avoid_large_gatherings'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.avoid_large_gatherings.isnull().sum() #0 Missing values
df_vac.avoid_large_gatherings.value_counts() 
'''
0    14734
1     8242'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ avoid_large_gatherings', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.009312 ie p_value is <0.05; Ho Reject; Good Predictor

#reduced_outside_home_cont
#______histogram
#_run in block
plt.hist(df_vac.reduced_outside_home_cont, bins = 'auto', facecolor = 'red')
plt.xlabel('reduced_outside_home_cont')
plt.ylabel('counts')
plt.title('Histogram of reduced_outside_home_cont') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['reduced_outside_home_cont'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.reduced_outside_home_cont.isnull().sum() #0 Missing values
df_vac.reduced_outside_home_cont.value_counts() 
'''
0    15233
1     7743'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ reduced_outside_home_cont', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.00523 ie p_value is <0.05; Ho Reject; Good Predictor

np.corrcoef(df_vac.reduced_outside_home_cont, df_vac.avoid_large_gatherings) #.58 Correlation only 1 will be taken

#avoid_touch_face
#______histogram
#_run in block
plt.hist(df_vac.avoid_touch_face, bins = 'auto', facecolor = 'red')
plt.xlabel('avoid_touch_face')
plt.ylabel('counts')
plt.title('Histogram of avoid_touch_face') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['avoid_touch_face'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.avoid_touch_face.isnull().sum() #0 Missing values
df_vac.avoid_touch_face.value_counts() 
'''
1    15752
0     7224'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ avoid_touch_face', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#1.079786e-26 ie p_value is <0.05; Ho Reject; Good Predictor

#dr_recc_h1n1_vacc
#______histogram
#_run in block
plt.hist(df_vac.dr_recc_h1n1_vacc, bins = 'auto', facecolor = 'red')
plt.xlabel('dr_recc_h1n1_vacc')
plt.ylabel('counts')
plt.title('Histogram of dr_recc_h1n1_vacc') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['dr_recc_h1n1_vacc'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers

df_vac.dr_recc_h1n1_vacc.isnull().sum() #0 Missing values
df_vac.dr_recc_h1n1_vacc.value_counts() 
'''
0    17870
1     5106'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ dr_recc_h1n1_vacc', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.0 ie p_value is <0.05; Ho Reject; Good Predictor

#dr_recc_seasonal_vacc
#______histogram
#_run in block
plt.hist(df_vac.dr_recc_seasonal_vacc, bins = 'auto', facecolor = 'red')
plt.xlabel('dr_recc_seasonal_vacc')
plt.ylabel('counts')
plt.title('Histogram of dr_recc_seasonal_vacc') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['dr_recc_seasonal_vacc'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.dr_recc_seasonal_vacc.isnull().sum() #0 Missing values
df_vac.dr_recc_seasonal_vacc.value_counts() 
'''
0    15347
1     7629'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ dr_recc_seasonal_vacc', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#1.150672e-224 ie p_value is <0.05; Ho Reject; Good Predictor
np.corrcoef(df_vac.dr_recc_h1n1_vacc, df_vac.dr_recc_seasonal_vacc) #.59 Correlation - only one will be taken

#chronic_medic_condition
#______histogram
#_run in block
plt.hist(df_vac.chronic_medic_condition, bins = 'auto', facecolor = 'red')
plt.xlabel('chronic_medic_condition')
plt.ylabel('counts')
plt.title('Histogram of chronic_medic_condition') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['chronic_medic_condition'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.chronic_medic_condition.isnull().sum() #0 Missing values
df_vac.chronic_medic_condition.value_counts() 
'''
0    16449
1     6527'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ chronic_medic_condition', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#4.599262e-53  ie p_value is <0.05; Ho Reject; Good Predictor

#cont_child_undr_6_mnths
#______histogram
#_run in block
plt.hist(df_vac.cont_child_undr_6_mnths, bins = 'auto', facecolor = 'red')
plt.xlabel('cont_child_undr_6_mnths')
plt.ylabel('counts')
plt.title('Histogram of cont_child_undr_6_mnths') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['cont_child_undr_6_mnths'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers

df_vac.cont_child_undr_6_mnths.isnull().sum() #0 Missing values
df_vac.cont_child_undr_6_mnths.value_counts() 
'''
0    21044
1     1932'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ cont_child_undr_6_mnths', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#6.084202e-26  ie p_value is <0.05; Ho Reject; Good Predictor

#is_health_worker
#______histogram
#_run in block
plt.hist(df_vac.is_health_worker, bins = 'auto', facecolor = 'red')
plt.xlabel('is_health_worker')
plt.ylabel('counts')
plt.title('Histogram of is_health_worker') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_health_worker'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.is_health_worker.isnull().sum() #0 Missing values
df_vac.is_health_worker.value_counts() 
'''
0    20355
1     2621'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ is_health_worker', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#1.690312e-163  ie p_value is <0.05; Ho Reject; Good Predictor

#is_h1n1_vacc_effective
#______histogram
#_run in block
plt.hist(df_vac.is_h1n1_vacc_effective, bins = 'auto', facecolor = 'red')
plt.xlabel('is_h1n1_vacc_effective')
plt.ylabel('counts')
plt.title('Histogram of is_h1n1_vacc_effective') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_h1n1_vacc_effective'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.is_h1n1_vacc_effective.isnull().sum() #0 Missing values
df_vac.is_h1n1_vacc_effective.value_counts() 
'''
4    10424
5     6474
3     3723
2     1618
1      737'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ is_h1n1_vacc_effective', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.0  ie p_value is <0.05; Ho Reject; Good Predictor

#is_h1n1_risky
#______histogram
#_run in block
plt.hist(df_vac.is_h1n1_risky, bins = 'auto', facecolor = 'red')
plt.xlabel('is_h1n1_risky')
plt.ylabel('counts')
plt.title('Histogram of is_h1n1_risky') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_h1n1_risky'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.is_h1n1_risky.isnull().sum() #0 Missing values
df_vac.is_h1n1_risky.value_counts() 
'''
2    8790
1    7075
4    4757
5    1533
3     821'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ is_h1n1_risky', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.0  ie p_value is <0.05; Ho Reject; Good Predictor

#sick_from_h1n1_vacc
#______histogram
#_run in block
plt.hist(df_vac.sick_from_h1n1_vacc, bins = 'auto', facecolor = 'red')
plt.xlabel('sick_from_h1n1_vacc')
plt.ylabel('counts')
plt.title('Histogram of sick_from_h1n1_vacc') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['sick_from_h1n1_vacc'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.sick_from_h1n1_vacc.isnull().sum() #0 Missing values
df_vac.sick_from_h1n1_vacc.value_counts() 
'''
2    8016
1    7869
4    5086
5    1915
3      90'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ sick_from_h1n1_vacc', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#8.723606e-35  ie p_value is <0.05; Ho Reject; Good Predictor

#is_seas_vacc_effective
#______histogram
#_run in block
plt.hist(df_vac.is_seas_vacc_effective, bins = 'auto', facecolor = 'red')
plt.xlabel('is_seas_vacc_effective')
plt.ylabel('counts')
plt.title('Histogram of is_seas_vacc_effective') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_seas_vacc_effective'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers

df_vac.is_seas_vacc_effective.isnull().sum() #0 Missing values
df_vac.is_seas_vacc_effective.value_counts() 
'''
4    10281
5     8811
2     1927
1     1021
3      936'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ is_seas_vacc_effective', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#3.359166e-178 ie p_value is <0.05; Ho Reject; Good Predictor
np.corrcoef(df_vac.is_seas_vacc_effective, df_vac.is_h1n1_vacc_effective) #.47 Correlation -only  one will be taken

#is_seas_risky
#______histogram
#_run in block
plt.hist(df_vac.is_seas_risky, bins = 'auto', facecolor = 'red')
plt.xlabel('is_seas_risky')
plt.ylabel('counts')
plt.title('Histogram of is_seas_risky') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_seas_risky'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers

df_vac.is_seas_risky.isnull().sum() #0 Missing values
df_vac.is_seas_risky.value_counts() 
'''
2    7869
4    6811
1    5169
5    2642
3     485'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ is_seas_risky', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.0 ie p_value is <0.05; Ho Reject; Good Predictor
np.corrcoef(df_vac.is_seas_risky, df_vac.is_h1n1_risky) #.56 Correlation

#sick_from_seas_vacc
#______histogram
#_run in block
plt.hist(df_vac.sick_from_seas_vacc, bins = 'auto', facecolor = 'red')
plt.xlabel('sick_from_seas_vacc')
plt.ylabel('counts')
plt.title('Histogram of sick_from_seas_vacc') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['sick_from_seas_vacc'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.sick_from_seas_vacc.isnull().sum() #0 Missing values
df_vac.sick_from_seas_vacc.value_counts() 
'''
1    10458
2     6686
4     4278
5     1489
3       65'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ sick_from_seas_vacc', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.171066 ie p_value is >0.05; Ho accepted; Bad Predictor

#age_bracket - Object
#______histogram
#_run in block
plt.hist(df_vac.age_bracket, bins = 'auto', facecolor = 'red')
plt.xlabel('age_bracket')
plt.ylabel('counts')
plt.title('Histogram of age_bracket') 

df_vac.age_bracket.isnull().sum() #0 Missing values
df_vac.age_bracket.value_counts() 
'''
65+ Years        5694
55 - 64 Years    4834
45 - 54 Years    4588
18 - 34 Years    4519
35 - 44 Years    3341'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['age_bracket']= le.fit_transform(df_vac['age_bracket']) 
df_vac.age_bracket.value_counts()
'''
4    5694
3    4834
2    4588
0    4519
1    3341'''
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ age_bracket', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#2.253509e-11 ie p_value is <0.05; Ho rejected; Good Predictor

#qualification - object
df_vac.qualification.isnull().sum() #478 Missing values
df_vac.qualification.value_counts() 
'''
College Graduate    9023
Some College        6286
12 Years            5169
< 12 Years          2020'''

#Replacing missing values with the value having highest frequency
df_vac.qualification = df_vac.qualification.fillna('College Graduate')
df_vac.qualification.isnull().sum() #No Missing values
df_vac.qualification.value_counts()
'''
College Graduate    9501
Some College        6286
12 Years            5169
< 12 Years          2020'''

'''Clubing into 2 categories - 12 & < 12 as 0 ; College graduate & Some College  as 1'''

df_vac['qualification']=df_vac.get('qualification').replace('< 12 Years','0')
df_vac['qualification']=df_vac.get('qualification').replace('12 Years','0')
df_vac['qualification']=df_vac.get('qualification').replace('College Graduate','1')
df_vac['qualification']=df_vac.get('qualification').replace('Some College','1')
df_vac.qualification.value_counts()
df_vac['qualification'] = df_vac['qualification'].astype('int64')
df_vac.info()
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ qualification', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#6.574124e-21 ie p_value is <0.05; Ho rejected; Good Predictor

#race - object
df_vac.race.isnull().sum() #No Missing values
df_vac.race.value_counts() 
'''
White                18282
Black                 1785
Hispanic              1556
Other or Multiple     1353'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['race']= le.fit_transform(df_vac['race']) 
df_vac.race.value_counts()
'''
3    18282
0     1785
1     1556
2     1353'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ race', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#1.184237e-12 ie p_value is <0.05; Ho rejected; Good Predictor

#sex - object
df_vac.sex.isnull().sum() #No Missing values
df_vac.sex.value_counts() 
'''
Female    13724
Male       9252'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['sex']= le.fit_transform(df_vac['sex']) 
df_vac.sex.value_counts()
'''
0    13724
1     9252'''
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ sex', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.000191 ie p_value is <0.05; Ho rejected; Good Predictor

#income_level - object
df_vac.income_level.isnull().sum() #3015 Missing values
df_vac.income_level.value_counts() 
'''
<= $75,000, Above Poverty    11386
> $75,000                     6207
Below Poverty                 2368'''

#Replacing missing values with the value having highest frequency
df_vac.income_level = df_vac.income_level.fillna('<= $75,000, Above Poverty')
df_vac.income_level.isnull().sum() #No Missing values
df_vac.income_level.value_counts()
'''
<= $75,000, Above Poverty    14401
> $75,000                     6207
Below Poverty                 2368'''

#Converting to numeric/ integer
df_vac['income_level']=df_vac.get('income_level').replace('Below Poverty','0')
df_vac['income_level']=df_vac.get('income_level').replace('<= $75,000, Above Poverty','1')
df_vac['income_level']=df_vac.get('income_level').replace('> $75,000','1')
df_vac['income_level'] = df_vac['income_level'].astype('int64')
df_vac.income_level.value_counts()
df_vac.info()

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ income_level', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.001439 ie p_value is <0.05; Ho rejected; Good Predictor

#marital_status - object
df_vac.marital_status.isnull().sum() #471 Missing values
df_vac.marital_status.value_counts() 
'''
Married        12170
Not Married    10335'''

#Replacing missing values with the value having highest frequency
df_vac.marital_status = df_vac.marital_status.fillna('Married')
df_vac.marital_status.isnull().sum() #No Missing values
df_vac.marital_status.value_counts()

'''
Married        12641
Not Married    10335'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['marital_status']= le.fit_transform(df_vac['marital_status']) 
df_vac.marital_status.value_counts()
'''
0    12641
1    10335'''
df_vac.info()
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ marital_status', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#2.494875e-15 ie p_value is <0.05; Ho rejected; Good Predictor

#housing_status - object
df_vac.housing_status.isnull().sum() #1031 Missing values
df_vac.housing_status.value_counts() 
'''
Own     16715
Rent     5230'''

#Replacing missing values with the value having highest frequency
df_vac.housing_status = df_vac.housing_status.fillna('Own')
df_vac.housing_status.isnull().sum() #No Missing values
df_vac.housing_status.value_counts()
'''
Own     17746
Rent     5230'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['housing_status']= le.fit_transform(df_vac['housing_status']) 
df_vac.housing_status.value_counts()
'''
0    17746
1     5230'''
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ housing_status', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#1.582484e-07 ie p_value is <0.05; Ho rejected; Good Predictor

#employment - object
df_vac.employment.isnull().sum() #525 Missing values
df_vac.employment.value_counts() 
'''
Employed              12207
Not in Labor Force     8926
Unemployed             1318'''

#Replacing missing values with the value having highest frequency
df_vac.employment = df_vac.employment.fillna('Employed')
df_vac.employment.isnull().sum() #No Missing values
df_vac.employment.value_counts()
'''
Employed              12732
Not in Labor Force     8926
Unemployed             1318'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['employment']= le.fit_transform(df_vac['employment']) 
df_vac.employment.value_counts()
'''
0    12732
1     8926
2     1318'''
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ employment', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.019751 ie p_value is <0.05; Ho rejected; Good Predictor

#census_msa - object
df_vac.census_msa.isnull().sum() #No Missing values
df_vac.census_msa.value_counts() 
'''
MSA, Not Principle  City    10072
MSA, Principle City          6677
Non-MSA                      6227'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ census_msa', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.951711 ie p_value is >0.05; Ho accepted; Bad Predictor

#no_of_adults 
df_vac.no_of_adults.isnull().sum() #No Missing values
df_vac.no_of_adults.value_counts() 
'''
1    12682
0     6780
2     2495
3     1019'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ no_of_adults', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.463975 ie p_value is >0.05; Ho accepted; Bad Predictor

#no_of_children 
df_vac.no_of_children.isnull().sum() #No Missing values
df_vac.no_of_children.value_counts() 
'''
0    16083
1     2794
2     2553
3     1546'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_vaccine ~ no_of_children', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod, type = 2)
print(aov_table)
#0.403791 ie p_value is >0.05; Ho accepted; Bad Predictor

#h1n1_vaccine - Target Variable
df_vac.h1n1_vaccine.isnull().sum() #No Missing values
df_vac.h1n1_vaccine.value_counts() 
'''
0    17791
1     5185'''

#With 23 predictors
#_____________________________________________Logistic Regression model - 1 method
from sklearn.metrics import classification_report, confusion_matrix
x = df_vac.iloc[:,[2,3,4,5,6,7,9,10,12,13,14,15,18,19,20,22,23,24,25,26,27,28,29]]
y = df_vac.iloc[:,33]  
#solver liblinear
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(solver='liblinear', random_state=0)
model1.fit(x, y)
model1.intercept_
model1.coef_

#Predictions
y_pred = model1.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17102   689]
 [ 4170  1015]]'''

#Accuracy Score - correct predictions / total number of data points
model1.score(x,y) #.0.78851

#ROC Curve - Receiver Operating Characteristic curve
#tpr = True Positive Rate 
#fpr = False Positive Rate
from sklearn.metrics import roc_curve, auc, roc_auc_score
y_pred_prob = model1.predict_proba(x)
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74911
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4) #Location of label
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.29      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.74     22976'''

#solver newton-cg
model1 = LogisticRegression(solver='newton-cg', random_state=0)
model1.fit(x, y)
model1.intercept_
model1.coef_

#Predictions
y_pred = model1.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17090   701]
 [ 4152  1033]]'''

#Accuracy Score - correct predictions / total number of data points
model1.score(x,y) #.0.78877

##ROC & AUC
y_pred_prob = model1.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74913
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#solver lbfgs
model1 = LogisticRegression(solver='lbfgs', random_state=0)
model1.fit(x, y)
model1.intercept_
model1.coef_

#Predictions
y_pred = model1.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17091   700]
 [ 4152  1033]]'''

#Accuracy Score - correct predictions / total number of data points
model1.score(x,y) #.0.78882

#ROC & AUC
y_pred_prob = model1.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74913
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#solver sag
model1 = LogisticRegression(solver='sag', random_state=0)
model1.fit(x, y)
model1.intercept_
model1.coef_

#Predictions
y_pred = model1.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17090   701]
 [ 4152  1033]]'''

#Accuracy Score - correct predictions / total number of data points
model1.score(x,y) #0.78877

#ROC & AUC
y_pred_prob = model1.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74913
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#solver saga
model1 = LogisticRegression(solver='saga', random_state=0)
model1.fit(x, y)
model1.intercept_
model1.coef_

#Predictions
y_pred = model1.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17091   700]
 [ 4152  1033]]'''

#Accuracy Score - correct predictions / total number of data points
model1.score(x,y) #0.78882

#ROC & AUC
y_pred_prob = model1.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74913
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#5 different solvers, output is slightly different

#Logistic Regression model - 2 method
import statsmodels.api as sm
import statsmodels.formula.api as smf
model2 = smf.glm(formula='''h1n1_vaccine~h1n1_worry+h1n1_awareness+antiviral_medication
                +contact_avoidance+bought_face_mask+wash_hands_frequently+
                reduced_outside_home_cont+avoid_touch_face+dr_recc_seasonal_vacc
                +chronic_medic_condition+cont_child_undr_6_mnths+is_health_worker
                +sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky
                +age_bracket+qualification+race+sex+marital_status+housing_status
                +employment''', data=df_vac, family=sm.families.Binomial())
result = model2.fit()
print(result.summary())

predictions = result.predict() 
predictions_nominal = [0 if x < 0.5 else 1 for x in predictions]
predictions_nominal 
#Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df_vac["h1n1_vaccine"], predictions_nominal))
'''
[[17098   693]
 [ 4149  1036]]'''

#Accuracy Score - correct predictions / total number of data points
(17098+1037)/(17098+693+4149+1037) #0.78926

#ROC & AUC
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], predictions)
roc_auc = auc(fpr, tpr) #Area under Curve 0.74913
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification Report
print(classification_report(df_vac["h1n1_vaccine"], predictions_nominal, digits = 3))

###############################################################################
#Values with p value > 0.05
'''contact_avoidance,wash_hands_frequently,avoid_touch_face,housing_status, 
employment'''

#With 17 Variables
#Removing  p values greater than 0.05
#Logistic Regression model - 1 method
from sklearn.metrics import classification_report, confusion_matrix
x = df_vac.iloc[:,[2,3,4,6,9,12,13,14,15,18,19,20,22,23,24,25,26,27]]
y = df_vac.iloc[:,33]  
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression(solver='liblinear', random_state=0)
model3.fit(x, y)
model3.intercept_
model3.coef_

#Predictions
y_pred = model3.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17124   667]
 [ 4178  1007]]'''

#Accuracy score - correct predictions / total number of data points
model3.score(x,y) #.0.78912

#ROC & AUC
y_pred_prob = model3.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74901
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.19      0.29      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.58     22976
weighted avg       0.76      0.79      0.74     22976'''

#Solver newton-cg
model3 = LogisticRegression(solver='newton-cg', random_state=0)
model3.fit(x, y)
model3.intercept_
model3.coef_

#Predictions
y_pred = model3.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17108   683]
 [ 4156  1029]]'''

#Accuracy score - correct predictions / total number of data points
model3.score(x,y) #0.78938

#ROC & AUC
y_pred_prob = model3.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74902
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#solver lbfgs
model3 = LogisticRegression(solver='lbfgs', random_state=0)
model3.fit(x, y)
model3.intercept_
model3.coef_

#Predictions
y_pred = model3.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17107   684]
 [ 4156  1029]]'''

#Accuracy score - correct predictions / total number of data points
model3.score(x,y) #.0.78934

#ROC & AUC
y_pred_prob = model3.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74902
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#solver sag
model3 = LogisticRegression(solver='sag', random_state=0)
model3.fit(x, y)
model3.intercept_
model3.coef_

#Predictions
y_pred = model3.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17108   683]
 [ 4156  1029]]'''

#Accuracy score - correct predictions / total number of data points
model3.score(x,y) #0.0.78938

#ROC Curve
y_pred_prob = model3.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74902
print(roc_auc)

#ROC & AUC
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#solver saga
model3 = LogisticRegression(solver='saga', random_state=0)
model3.fit(x, y)
model3.intercept_
model3.coef_

#Predictions
y_pred = model3.predict(x)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[17108   683]
 [ 4156  1029]]'''

#Accuracy score - correct predictions / total number of data points
model3.score(x,y) #0.78938

#ROC & AUC
y_pred_prob = model3.predict_proba(x)
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.74902
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support

           0       0.80      0.96      0.88     17791
           1       0.60      0.20      0.30      5185

    accuracy                           0.79     22976
   macro avg       0.70      0.58      0.59     22976
weighted avg       0.76      0.79      0.75     22976'''

#Logistic Regression model - 2 method
import statsmodels.api as sm
import statsmodels.formula.api as smf
model4 = smf.glm(formula='''h1n1_vaccine~h1n1_worry+h1n1_awareness+antiviral_medication
                +bought_face_mask+reduced_outside_home_cont+dr_recc_seasonal_vacc
                +chronic_medic_condition+cont_child_undr_6_mnths+is_health_worker
                +sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky
                +age_bracket+qualification+race+sex+marital_status''', data=df_vac, family=sm.families.Binomial())
result = model4.fit()
print(result.summary())

predictions = result.predict()
predictions_nominal = [0 if x < 0.5 else 1 for x in predictions]

#Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df_vac["h1n1_vaccine"], predictions_nominal))
'''
[[17110   681]
 [ 4153  1032]]'''

#Accuracy score - correct predictions / total number of data points
(17110+1032)/(17110+681+4153+1032) #0.78960

#ROC & AUC
from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], predictions)
roc_auc = auc(fpr, tpr) #Area under Curve 0.74900
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for h1n1 Vaccine Classifier')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification Report
print(classification_report(df_vac["h1n1_vaccine"], predictions_nominal, digits = 3))

















from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds =roc_curve(df_vac["h1n1_vaccine"], predictions)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
#Area under the ROC curve : 0.749003

####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

#ROC Curve
# Plot tpr vs 1-fpr
import pylab as pl
fig, ax = pl.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color = 'red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC Curve')
ax.set_xticklabels([])

















pd.set_option('display.max_column',None)
df_vac.iloc[:,2:21].corr()
