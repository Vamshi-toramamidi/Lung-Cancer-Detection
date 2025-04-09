# Download the required packages using "pip install (package name)" command 
# in cmd if they are not present in your environment
# Importing Modules for Data Pre-processing and Visualization

# pandas - Analyzing the Data
import pandas as pd 
# NumPy - Arrays and Linear algebra
import numpy as np 
# Matplotlib - Data Visualization - Graphical Plotting
import matplotlib.pyplot as mp 
# seaborn - Data Visualization - Statistical PLotting
import seaborn as sb 
# Sklearn - Tools for machine learning and statistical modeling including classification, 
# regression,clustering and dimensionality reduction
from sklearn import preprocessing

# we are importing packages using "as" and two letters 
# to reffer the package easily in future use

# Setting the Statistical graph style - ggplot 
# mp.style.use('ggplot')
# Declaring up our custom color palette - Color Brewer - Qualitative - Paired 
# colors = sb.color_palette('Paired',5)
# As we declared our color palette now setting it up the palette
# sb.set_palette(colors)

# Creating a data frame for lung cancer csv file
df = pd.read_csv("Lung_Cancer_Dataset.csv") 


''' -------------------------------------- PRE - PROCESSING OF DATASET --------------------------------------'''


# Shape(row and columns) and information(Attributes and Datatype) about Dataset
# print(df.shape)
# print(df.info())
# Analysis on the numerical columns
# print(df.describe())

# Check for null values in the dataset
# print(df.isnull().sum())
# Used to drop Null Value rows 
# df.dropna()

# Check if there are any duplicates, if yes - how many ?
# print(df.duplicated().sum())
# Droping all the duplicate rows 
df.drop_duplicates(inplace =True)

# Encoding the Categorial data into numerical form as some of the Algorithm 
# dont work efficiently with categorial data 
label_encoder = preprocessing.LabelEncoder()
 # Male - 1 & Female - 0
df['GENDER']= label_encoder.fit_transform(df['GENDER'])
# Yes - 1 & No - 0
df['LUNG_CANCER']= label_encoder.fit_transform(df['LUNG_CANCER']) 
df.to_csv("Modified.csv",index=False)
mdf = pd.read_csv("Modified.csv")
# print(df)

# print(cat_col)

''' -------------------------------------- SPLITING THE DATA --------------------------------------'''

# Droping Lung Cancer Column as it is the result column for X
X=mdf.drop(['LUNG_CANCER'],axis=1)
# Adding Lung Cancer Column as it is Y
y=mdf['LUNG_CANCER']

# Changing values from 2 to 1 as it becomes easier and more efficient for to understand the data for us
for i in X.columns[2:]:
    temp=[]
    for j in X[i]:
        temp.append(j-1)
    X[i]=temp
# print(X.head())
from imblearn.over_sampling import RandomOverSampler
X_over,y_over=RandomOverSampler().fit_resample(X,y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_over,y_over,random_state=42,stratify=y_over)
# print(f'Train shape : {X_train.shape}\nTest shape: {X_test.shape}')

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train['AGE']=scaler.fit_transform(X_train[['AGE']])
X_test['AGE']=scaler.transform(X_test[['AGE']])
# print(X_train.head())

'''-------------------------------------- MODEL TRAINING --------------------------------------'''

import sklearn
import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
confusion=sklearn.metrics.confusion_matrix(y_test,y_pred)
print(confusion)
# mp.figure(figsize=(8,8))
# sb.heatmap(confusion,annot=True)
mp.xlabel("Predicted")
mp.ylabel("Actual")

'''-------------------------------------- TESTING MODEL --------------------------------------'''

# Classification Report for the Model 
print(" ------------------------------------------------------")
print(sklearn.metrics.classification_report(y_test,y_pred))
print(" ------------------------------------------------------")
# Accuracy of the model 
print("Accuracy using LBGM Classifier is : ",sklearn.metrics.accuracy_score(y_test, y_pred))
print(" ------------------------------------------------------")


input_data = (1,2,1,2,1,1,1,1,1,2,1,2,1,2,1)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Lung Cance')
else:
  print('The Person has Lung Cancer')