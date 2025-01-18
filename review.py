# Import The Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure,show
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# -------------------------------------------------------------
print("Sentimental Analysis on a dataset of Iphone customer review using TF-IDF Vectorization and Logistic Regression.")
# -------------------------------------------------------------

#--------------------------------------------------------------
# Loading the data and displaying its top 5 rows and columns 
db=pd.read_csv("iphone.csv")
print("\nTop 5 rows of dataset :\n",db.head())
print("\nThe number of rows and columns in dataset :",db.shape)
#-----------------------------------------------------------

#----------------------------------------------------------
#Data preprocessing
print("\nMissing values in each column :\n",db.isnull().sum())   # Displaying the missing values of each column in data set

db.dropna(inplace=True) #Dropping the missing values or null values

print("\nAgain checking if any null value is present or not :\n",db.isnull().sum())   #This time every column will show 0 as their is no null value we dropped everynull value

# Counting Unique rating
print("\nThe unique ratings are :\n",db["ratingScore"].value_counts())

# We will remove rating=3 assuming it as neutral 
print("\nDisplaying data without rating no 3 :\n",db[db["ratingScore"] != 3])
db=db[db["ratingScore"] != 3]
print("\nNow the unique ratings are :\n",db["ratingScore"].value_counts())

# We will create new column [New_Rating] where any entity greater than 3 is positive rating (1) and less 3 will be negative rating (0)
db["New_Rating"]=np.where(db["ratingScore"]>3,1,0)

print("\nCount of postive (1) and negtive (0) ratings :\n",db["New_Rating"].value_counts())  #Data is imbalanced as no of  1 is more than 0's

print("\nTop 5 rows of dataset :\n",db.head())  #to view new column added [New_Rating]
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#Displaying the barchart 
figure()
target="New_Rating"
sns.countplot(data=db,x=target)
show()
#----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
#Spliting the data here training parameter or independent variable is "reviewDescription" which is peoples comment or reviews on iphone in india 
x_train,x_test,y_train,y_test=train_test_split(db["reviewDescription"],db["New_Rating"],random_state=50,stratify=db["New_Rating"])
print("\nPeoples reviews about iphone [Training data] :\n",x_train)

#----------------------------------------------------------------------------------
# Now we have to do NLP Natural language processing by Tfidfvectorizer

vect=TfidfVectorizer()
vect.fit(x_train)       #builds a vocabulary assigning a unique column index
print("\nThe Features created by tfidf which are part of tfidf matrix\n",vect.get_feature_names_out()) #Returns the name of the feature i.e tokens which are a part of tf-idf matrix


#Transform the training data
x_train_vectorized=vect.transform(x_train)  #Contain sparse matrix row:-correspond to each review(row of dataset) & column:-tokens created


#-------------------------------------------------------------------------------------
# Using logistic regression to predict probability of 1 and 0
model=LogisticRegression(class_weight='balanced')   #As data is imbalanced we used class_weight='balanced'
model.fit(x_train_vectorized,y_train)
predictions=model.predict(vect.transform(x_test))

print("AUC (Area Under Curve) :",roc_auc_score(y_test,predictions))        #Area under curve 

feature_names=np.array(vect.get_feature_names_out())
sorted_coef_index=model.coef_[0].argsort()

print("\nSmallest Coef :\n",feature_names[sorted_coef_index[:10]])  #Features having least importance in contributing to logisticregression model.

print("\nLargest Coef :\n",feature_names[sorted_coef_index[:-11:-1]]) #Features having most importance in contributing to model.
#------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------
#Testing by giving my own input
ch='y'
while(ch=='y' or ch=='Y'):
    a =input("\nEnter The review :")
    ans=model.predict(vect.transform([a]))
    if(ans==1):
        print("Positive Review")
        ch=input("\nDo you want to continue 'y':-yes or 'n':- No :")

    else:
        print("Negative Review")
        ch=input("\nDo you want to continue 'y':-yes or 'n':- No :")
    
#--------------------------------------------------------------------------