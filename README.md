<H3>ENTER YOUR NAME: SWETHA S</H3>
<H3>ENTER YOUR REGISTER NO: 212222230155</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```

## OUTPUT:

## DATASET:
![Screenshot 2024-02-25 211743](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/b947ff8a-0265-441a-b442-f7bb66042f60)

## X VALUES:
![Screenshot 2024-02-25 211754](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/2c8236a8-c12e-418c-ac6c-b0cff5eb98eb)

## Y VALUES:
![Screenshot 2024-02-25 211801](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/0b889e49-b7da-4c33-8e8c-3852f51d7c06)

## NULL VALUES:
![Screenshot 2024-02-25 211808](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/5ccd8377-9c88-4de2-9678-7882fc4ea104)

## DUPLICATED VALUES:
![Screenshot 2024-02-25 211820](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/8d768cf1-7cc1-4f78-83f4-8f66ee0f3c23)

## DESCRIPTION:
![Screenshot 2024-02-25 211850](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/6c868c87-3aaf-408a-acf9-7994c73e9695)

## NORMALIZED DATASET:
![Screenshot 2024-02-25 211925](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/ef5c94b3-e771-4e32-8c88-a74284783a70)

## TRAINING DATA:
![Screenshot 2024-02-25 211930](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/a327b6cd-6266-4df5-b742-a830cbe76909)

## TESTING DATA:
![Screenshot 2024-02-25 211936](https://github.com/swethaselvarajm/Ex-1-NN/assets/119525603/a242d90a-337f-4a1a-9fcc-b93d773c88ea)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


