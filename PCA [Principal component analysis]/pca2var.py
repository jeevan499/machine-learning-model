# Jesus is my saviour!!
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("C:/Users/Dr Vinod/Desktop/DataSets1/pca2var.csv")
df.info()

df1 = scale(df)

#PCA 
pca2 = PCA(n_components=2)
pca2_comp = pca2.fit_transform(df1)
pca2_comp #Components
'''
array([[-1.08643242, -0.22352364],
       [ 2.3089372 ,  0.17808082],
       [-1.24191895,  0.501509  ],
       [-0.34078247,  0.16991864],
       [-2.18429003, -0.26475825],
       [-1.16073946,  0.23048082],
       [ 0.09260467, -0.45331721],
       [ 1.48210777,  0.05566672],
       [ 0.56722643,  0.02130455],
       [ 1.56328726, -0.21536146]])
'''
pca2_egvct = pca2.components_ #Eigen vectors
pca2_egvct
'''
array([[-0.70710678, -0.70710678],
       [-0.70710678,  0.70710678]])
'''
pca2_egvl = pca2.explained_variance_ #Eigen Values
pca2_egvl
'''
array([2.13992141, 0.08230081])
'''
#The amount of variance that each PC explains
var= pca2.explained_variance_ratio_
var
'''
array([0.96296464, 0.03703536])
'''
#Cumulative Variance explains
var1=np.cumsum(np.round(pca2.explained_variance_ratio_, decimals=4)*100)
var1
'''
array([ 96.3, 100. ])
'''
plt.plot(var1) # cumulative








































