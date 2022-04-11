# Jesus is Great!
#Assigning Predictors & Target Variable
mpm.info()
x = mpm.iloc[:,:-1] #14 Variables
x.info()
y = mpm.iloc[:,-1]y
#Splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state = 25,test_size=0.25)
len(x_train) #1500
len(x_test) #500
len(y_train) #1500
len(y_test) #500
#Building Model @ n_neighbors = 13
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13)
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)
#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)
#Prediction Score
mpm_knn.score(x_test, y_test)
#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #94.8
#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))































#________________________________________find optimal number of n_neigbors
error_rate = [] #an empty list
  
# Will take some time 
for i in range(1, 40): 
      
    knn = KNeighborsClassifier(n_neighbors = i) 
    knn.fit(x_train, y_train) 
    pred_i = knn.predict(x_test) 
    error_rate.append(np.mean(pred_i != y_test)) 
  
plt.figure(figsize =(10, 6)) 
plt.plot(range(1, 40), error_rate, color ='blue', 
                linestyle ='dotted', marker ='o', 
         markerfacecolor ='red', markersize = 10) 
  
plt.title('Error Rate vs. n_neighbors') 
plt.xlabel('n_neighbors') 
plt.ylabel('Error Rate')

'''n_neighbor value with least error is the optimal value of n_neighbor'''

#Calculation
 knn = KNeighborsClassifier(n_neighbors = 2) 
 knn.fit(x_train, y_train) 
 pred_i = knn.predict(x_test) 
 error_rate.append(np.mean(pred_i != y_test)) 
    
(pred_i != y_test).value_counts()
'''
False    449
True      51'''
np.mean = no of missclassified observations/total no of observations
51/500 #.102
################################################