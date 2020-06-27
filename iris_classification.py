import streamlit as st
## importing all the relevant libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

## Header line for the app
st.write("""
	# A Simple Iris Flower Prediction App 

	This App predicts the **Iris flower** type !! """)


##Header for the side bar
st.sidebar.header("User Input Parameters")

##Creating a function to take all the input values from the web page 
def user_input_features():
	sepal_length = st.sidebar.slider("Sepal Length",4.3,7.9,5.4) #first argument is name we want to display
																  #second argument is minimun value for side bar
																  #third argument is mmaximun value for side bar
																  #fourth argument is default value for side bar as we load the page
	sepal_width = st.sidebar.slider("Sepal Width",2.0,4.4,3.4)
	petal_length = st.sidebar.slider("Petal Length",1.0,6.9,1.3)
	petal_width = st.sidebar.slider("Petal Width",0.1,2.5,0.2)

	#creating dictionary for the input parameters
	data = {'sepal_length':sepal_length,'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width}

	#converting the dictionary to dataframe 
	features = pd.DataFrame(data,index=[0])
	return features

## Calling the above function 
df = user_input_features()

#writing the above dataset to display on to the webpage
st.subheader('User Input Parameters')
st.write(df)

#loading the dataset and declaring the feature and target variable
data = load_iris()
X = data.data
y = data.target

#creating standard scaler object 
sc = StandardScaler()

#creating a PCA object .PCA helps in dimensionality reduction
pca = PCA()

#creating object for logistic model
logistic= LogisticRegression()

#Creating a pipeline to execute the model
#first, we will standardize the data
#then , comes the PCA where we will select the optimum features
#finally , we will use the Logistic Regression model to train the data

pipeline = Pipeline(steps=[('sc',sc),
                           ('pca',pca),
                           ('logistic',logistic)])


#### Creating a parameter space
#### we will create a list of parameters from which the GridSearchCV algorithm will select these values to train the data
#### and it will then select the best optimum values of parameters


#since there are 4 features in this dataset , we will try to cover all the data features from the dataset to select the best features 
# using the PCA technique
n_components = list(range(1,X.shape[1]+1,1))

#The trade-off parameter of logistic regression that determines the strength of the regularization is called C, 
#and higher values of C correspond to less regularization (where we can specify the regularization function).
#C is actually the Inverse of regularization strength(lambda)
C = np.logspace(-4,4,50)

#Nextly , we will create a list of regularization penalty i.e. l1 and l2
penalty = ['l1','l2']

# let's create a dictionary of all the above parameters we defined
parameters = dict(pca__n_components = n_components,
                 logistic__C = C,
                 logistic__penalty = penalty)

#### Note that  , we can have access to the parameter values of the pipeline using the '__' operator

#Now , let's create a GridSearchCV object to fit the data using the parameter optimization techniques
# we use the pipeline we initialized and parameter values that we have set above
clf = GridSearchCV(pipeline,parameters)

## Fitting the gridsearch object to data
clf.fit(X,y)


##Displaying the target names on screen 
st.subheader("Class labels and their corresponding index no")
st.write(data.target_names)

## Predicting the inputs from the user 
prediction = clf.predict(df)
prediction_probability = clf.predict_proba(df)

##Printing the prediction on web page
st.subheader("Prediction")
st.write(data.target_names[prediction])

##Printing prediction probability 
st.subheader("Prediction Probability")
st.write(prediction_probability)
