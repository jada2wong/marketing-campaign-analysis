
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os


# Read file and reset y variable

# In[3]:


os.getcwd()
os.chdir('C:\\Users\\ym\\Downloads\\新建文件夹\\652')


# In[20]:


path="bank-full.csv"
df=pd.read_csv(path,sep=";")
df.rename(columns={"y":"dep"},inplace=True)


# In[21]:


print(df.describe(include="all"))
from sklearn.preprocessing import StandardScaler
sca=StandardScaler()

# In[22]:


df=pd.get_dummies(df,columns=["job",'marital','education','contact','month','poutcome'])#transfer categoreical variabled with k dummy
df=pd.get_dummies(df,columns=["default",'housing','loan','dep'],drop_first="Ture")#transfer variable with k-1 dummy

df[['age','balance','day','duration','campaign','pdays','previous']]=sca.fit_transform(df[['age','balance','day','duration','campaign','pdays','previous']])#standerdized numerical variable

print(df.head())

# In[24]:
# Split data set to train and test data set

from sklearn.model_selection import StratifiedShuffleSplit
# split the data into training and test sets use stratified shuffle split. means keep the same ratio of dep variable in test set 
sss=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in sss.split(df,df['dep_yes']):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_set,test_set=df.loc[train_index],df.loc[test_index]

y_train=train_set['dep_yes']#split y variable
y_test=test_set['dep_yes']










# In[31]:

#split x variables
X_train=train_set.drop(["dep_yes"],axis=1)
X_test=test_set.drop(["dep_yes"],axis=1)


# Model comparesion



print(X_test)


# In[27]:


import time

from sklearn.ensemble import RandomForestClassifier


from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score


# In[28]:
#set of model

n_estimators = 10
dict_clt = {
    
    "Linear SVM": BaggingClassifier(SVC(kernel='linear'), max_samples=1.0 / n_estimators, n_estimators=n_estimators),
    "RBF SVM":BaggingClassifier(SVC(kernel='rbf'), max_samples=1.0 / n_estimators, n_estimators=n_estimators),
    "Decision Tree": BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=1.0 / n_estimators, n_estimators=n_estimators),
    "Random Forest": RandomForestClassifier(),
    
}


# In[29]:

#funcation to create a df to restore the model accuracy


def batch_classify(X_train, y_train,X_test,y_test):
    df_results = pd.DataFrame(data=np.zeros(shape=(4,3)), columns = ['classifier', 'test_score', 'training_time'])#create empty 4*3 data frame 
    count = 0
    for key, clt in dict_clt.items():
        
            t_start = time.clock()#start time
            clt.fit(X_train, y_train)#fit x and y
            t_end = time.clock()#end time
            t_diff = t_end - t_start#time of model building
            train_score = clt.score(X_test, y_test)#accuracy score
            
            df_results.loc[count,'classifier'] = key#append data into df
            df_results.loc[count,'test_score'] = train_score
            df_results.loc[count,'training_time'] = t_diff
            #try:
                #f1score=f1_score(y_test,clt.predict(X_test))
                #df_results.loc[count,'f1']=f1score
     
            
            #except AttributeError:
                #df_results.loc[count,'f1']="NA"
            count+=1
           
            
    return df_results


# In[ ]:


df_results = batch_classify(X_train, y_train,X_test,y_test)#fit fucntion with test and training data set
print(df_results.sort_values(by='test_score', ascending=False))

#the warning here is because the version update of sklearn, it changes some default parameter and we did not set it. it will not influence our resul=t


# In[21]:


for key,clf in dict_clt.items() :
    n_estimators = 10
    clf.fit(X_train, y_train)
    pre=clf.predict(X_test)
    print(key,classification_report(y_test,pre))#print classification report for each model


# Avoiding Overfitting:
# 
# This is an error in the modeling algorithm that takes into consideration random noise in the fitting process rather than the pattern itself. 


# In[23]:
#10 folder cross validation

def batch_classify1(X_train, y_train):
    df_results = pd.DataFrame(data=np.zeros(shape=(5,2)), columns = ['classifier', 'cv_score'])
    count = 0
    for key, clt in dict_clt.items():
        
            
            
            
            train_score = cross_val_score(clt,X_train, y_train,cv=10).mean()
            
            df_results.loc[count,'classifier'] = key
            df_results.loc[count,'cv_score'] = train_score
            count+=1
           
            
    return df_results


# In[25]:


df_results = batch_classify1(X_train, y_train)
print(df_results)




# In[22]:
# Which Features Influence the Result of a Term Deposit Suscription?


clf= tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.feature_importances_)

#####################
#feature importance after drop duration
X_train_1=X_train.drop(["duration"],axis=1)
X_test_1=X_test.drop(["duration"],axis=1)



##################
clf= tree.DecisionTreeClassifier()
clf.fit(X_train_1, y_train)
print(clf.feature_importances_)

