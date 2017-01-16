# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:48:16 2016

@author: Vaibhav
"""
import os
os.getcwd()
os.chdir('D:/Analytics Vidhya/BigMartSales')
import pandas as pd
import matplotlib as plt
import numpy as np
train=pd.read_csv("D:/Analytics Vidhya/BigMartSales/train_bms.csv")
test=pd.read_csv("D:/Analytics Vidhya/BigMartSales/test_bms.csv")
df=pd.concat([train,test],ignore_index=True)
train.head(2)
train.shape,test.shape,df.shape
(train.isnull().sum()).Outlet_Size
sum(train.Outlet_Size.isnull())
df.head(10)
df.describe()
type('Item_Fat_Content')
df.apply(lambda x: type(x))
df.apply(lambda x: len(x.unique()))

##1) Imputing Item_Weight Missing values based on Item Identifier
## Applying the average wieght of all the items to missing values
item_avg_wt=df.pivot_table(values='Item_Weight', index='Item_Identifier', aggfunc=np.mean)
item_avg_wt

##Fill the mean values depending on the Identifier mean
miss_bool=df['Item_Weight'].isnull()
miss_bool
##Imputing the value
df.loc[miss_bool,'Item_Weight']=df.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_wt[x])
sum(df['Item_Weight'].isnull())

##2) Imputing for Outlet_size based on Outlet_type, Using mode to impute
from scipy.stats import mode
outlet_size_mode=df.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.dropna()).mode[0]))
miss_bool=df['Outlet_Size'].isnull()
df.loc[miss_bool, 'Outlet_Size']=df.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
sum(df['Outlet_Size'].isnull())

## There are no missing values in the dataset now

##Time for feature engineering
## If Outlet_Tyoe has same sales we can ignore them..lets c
df.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')
##All have different, so not making any changes

##Lets check on the visibility 0 part, replace them with mean
sum(df['Item_Visibility']==0)
visibility_avg=df.pivot_table(values='Item_Visibility', index='Item_Identifier')
zero_bool=(df['Item_Visibility']==0)
df.loc[zero_bool,'Item_Visibility']=df.loc[zero_bool,'Item_Identifier'].apply(lambda x: visibility_avg[x])

##Lets calculate the proportion of a product's visibility as compared to the mean visibility
df['Item_Visibility_MeanRatio']=df.apply(lambda x:x['Item_Visibility']/visibility_avg[x['Item_Identifier']],axis=1)
##If you see Item Identifier has FD, DR and NC i.e. Food,Drink and Non consumables..
##lets create a new column to divide the products

## Getting 1st 2 chars of the Identifier
df['Item_Type_Combined']=df['Item_Identifier'].apply(lambda x:x[0:2])
##Lets give them proper names
df['Item_Type_Combined']=df['Item_Type_Combined'].map({'FD':'Food','DR':'Drink','NC':'Non-Consumables'})
df['Item_Type_Combined'].value_counts()

#Lets determine the years of operation of the prodyct
df['Outlet_Years']=2013-df['Outlet_Establishment_Year']
df['Outlet_Years'].describe()

##There are some typos in Item_fat_content
df['Item_Fat_Content'].value_counts()
df['Item_Fat_Content']=df['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})

##There are some non consumbales as well for which fat content should not be defined
df.loc[df['Item_Type_Combined']=='Non-Consumables','Item_Fat_Content']='Non-Edible'
df['Item_Fat_Content'].value_counts()

##Our Data Exploration ends here,,,Lets start with coding

## Lets make every column as numerical as scipy only accpets numerical variables
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
##Creating new variable for outlet_Identifier as this is to be used in submission
df['Outlet']=le.fit_transform(df['Outlet_Identifier'])
var_mod=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    df[i]=le.fit_transform(df[i])
df.dtypes
##One hot Coding
df=pd.get_dummies(df, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet'])
df.dtypes
##Making train and test set again and droping the columns which are converted to others
df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

##Divide into Train and Test
train=df.loc[df['Item_Outlet_Sales'].isnull()==False]
test=df.loc[df['Item_Outlet_Sales'].isnull()]
train.shape,test.shape
test.drop(['Item_Outlet_Sales'],axis=1,inplace=True)
##Export the files
train.to_csv('train_modified.csv',index=False)
test.to_csv('test_modified.csv',index=False)

##Time to build models
##1) Lets start building a base model
mean_sale=train['Item_Outlet_Sales'].mean()
mean_sale
algo0=test[['Item_Identifier','Outlet_Identifier']]
algo0['Item_Outlet_Sales']=mean_sale
algo0.shape
algo0.head(3)
algo0.to_csv('base_model.csv',index=False)

##Base model is created to have a check on other models, they should have a score better than base model
Idcol=['Item_Identifier','Outlet_Identifier']
target='Item_Outlet_Sales'
predictor=[x for x in train.columns if x not in Idcol+target]
predictor
train.apply(lambda x: sum(x==0))
sum(train[target].isnull())

##1)Linear Regression
from sklearn.linear_model import LinearRegression
clf=LinearRegression(normalize=True)
clf.fit(train[predictor],train[target])
from sklearn import cross_validation
kf_total = cross_validation.KFold(len(train), n_folds=10,shuffle=True, random_state=4)
kf_total
score=cross_validation.cross_val_score(clf, train[predictor], train[target], cv=kf_total, n_jobs=1)
score.mean()