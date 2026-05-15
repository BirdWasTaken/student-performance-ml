import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

df = pd.read_csv("StudentsPerformance.csv") #Loading Dataset
print(df.head())

#REGRESSION

print("\n--- Regression Model ---\n")

X = df[["reading score","writing score"]] #selecting Feature
y = df["math score"] #Selecting Target
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split( #Splitting the data into Training Set and Testing Set
    X, y, test_size= 0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)

model = LinearRegression() #Creating a model
model.fit(X_train,y_train) #Training the model

predictions = model.predict(X_test) #Making Predictions
print(predictions[:10])

new_data = pd.DataFrame([[80, 85]], columns=["reading score", "writing score"]) 
print(model.predict(new_data))  #Making Predictions on new data

print("Actual Values:")  #Comparing Actual Values with Predicted Values
print(y_test.head())
print("Predicted Values:")
print(predictions [:5])

error = mean_absolute_error(y_test, predictions)  #Calculating Error
print("Mean Absolute Error:",error)

print("Model Score", model.score(X_test,y_test)) #Calculating Accuracy (R^2)

new_dataa = pd.DataFrame([[60, 70]], columns=["reading score", "writing score"]) #prediction on new data
print(model.predict(new_dataa))

new_dataaa = pd.DataFrame([[95, 90]], columns=["reading score", "writing score"]) #prediction on new data
print(model.predict(new_dataaa))

plt.scatter(y_test,predictions) #Scatter plot of Actual Math Score and Predicted Math Score
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual VS Predicted Math Score")
plt.show()

#CLASSIFICATION

print("\n--- Classification Model ---\n")

df["pass_math"]= df["math score"] >= 40 #Creats a new column of pass_math and says True if math score >= 40 or else says False
print(df.head(10))

X = df[["reading score","writing score"]] # Selecting Feature
y = df["pass_math"] #selecting Target

X_train, X_test, y_train, y_test = train_test_split( #Dividing the model into training set and testing set
    X, y, test_size=0.2, random_state= 42
)

model = LogisticRegression() #Creating a model
model.fit(X_train,y_train) #training the model

predictions = model.predict(X_test) #Predicting Pass or Fail
print(predictions[:10])

print("Actual values")
print(y_test.head(20))
print("Predicted Values")
print(predictions[:20])

acc = accuracy_score(y_test,predictions) #Accuracy
print("Accuracy",acc)

df.groupby("pass_math")["math score"].mean().plot(kind="bar") #BarGraph of average math score of pass and fail
plt.show()

cm = confusion_matrix(y_test, predictions) #Confusion Matrix
print("Confusion Matrix\n", cm)