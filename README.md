# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neurons are the basic input/output units found in neural networks. These units are connected to one another, and each connection carries a weight. Because they are adaptable, neural networks can be applied to both classification and regression. We'll examine how neural networks can be used to tackle regression issues in this post.


A relationship between a dependent variable and one or more independent variables can be established with the aid of regression. Only when the regression equation is a good fit for the data can regression models perform well. Although sophisticated and computationally expensive, neural networks are adaptable and can choose the optimum form of regression dynamically. If that isn't sufficient, hidden layers can be added to enhance prediction.
Create your training and test sets using the dataset; in this case, we are creating a neural network with a second hidden layer that uses the activation layer as relu and contains its nodes. We will now fit our dataset before making a value prediction. 



## Neural Network Model

![Input](https://github.com/KoduruSanathKumarReddy/basic-nn-model/assets/69503902/47a829cd-5e11-4eaa-a5a4-24ab6a0a1248)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
~~~
Developed by: Koduru Sanath Kumar Reddy
Registration no: 212221240024
~~~
### Importing Required Packages
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
~~~
### Authentication and Creating DataFrame From DataSheet
~~~
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Exp1').sheet1
data = worksheet.get_all_values()
dataset = pd.DataFrame(data[1:], columns=data[0])
dataset = dataset.astype({'Input':'float'})
dataset = dataset.astype({'Output':'float'})
dataset.head()
~~~
### Assigning X and Y values
~~~
X = dataset[['Input']].values
y = dataset[['Output']].values
~~~
### Normalizing the data
~~~
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 20)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
X_train1 = Scaler.transform(x_train)
~~~
### Creating and Training the model
~~~
brainModel = Sequential([
    Dense(units = 5, activation = 'relu' , input_shape=[1]),
    Dense(units = 6),
    Dense(units = 1)

])
brainModel.compile(optimizer='rmsprop',loss='mse')
brainModel.fit(x=X_train1,y=y_train,epochs=20000)
~~~
### Plot the loss
~~~
loss_df = pd.DataFrame(brainModel.history.history)
loss_df.plot()
~~~
### Evaluate the Model
~~~
X_test1 = Scaler.transform(x_test)
brainModel.evaluate(X_test1,y_test)
~~~
### Prediction for a value
~~~
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
brainModel.predict(X_n1_1)
~~~
## Dataset Information
<img width="209" alt="Screenshot 2023-08-21 at 10 42 06 PM" src="https://github.com/KoduruSanathKumarReddy/basic-nn-model/assets/69503902/30b28dc8-60b0-4e24-b873-6baa4d38cf37">




## OUTPUT

### Training Loss Vs Iteration Plot

<img width="578" alt="Screenshot 2023-08-21 at 10 42 32 PM" src="https://github.com/KoduruSanathKumarReddy/basic-nn-model/assets/69503902/59b449b8-801a-45e2-bec2-2162d0996d62">



### Test Data Root Mean Squared Error
<img width="578" alt="Screenshot 2023-08-21 at 10 43 15 PM" src="https://github.com/KoduruSanathKumarReddy/basic-nn-model/assets/69503902/a2e0c53e-65b5-4b49-a329-e163c5c47f99">




### New Sample Data Prediction
<img width="578" alt="Screenshot 2023-08-21 at 10 43 33 PM" src="https://github.com/KoduruSanathKumarReddy/basic-nn-model/assets/69503902/976c6961-a06a-4e78-8a2d-39bda6976306">





## RESULT
Therefore a  Neural Network Regression Model is developed successfully for the given dataset.
