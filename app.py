import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logging import StreamHandler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



cal=fetch_california_housing()
df=pd.DataFrame(cal.data,columns=cal.feature_names)
df['Price']=cal.target

#Title of the APP
st.title('california house price')

#data overview
st.header('Data overview')
st.write(df.head (10))

# split the data into input and output
X = df.drop('Price', axis=1) # input features
y = df['Price'] # target
X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

mod=st.selectbox("select model",("Linear Regression","Decision Tree", "Random Forest"))

models= {
    "Linear Regression":LinearRegression(),
    "Random Forest":RandomForestRegressor(),
    "Decision Tree":DecisionTreeRegressor()
}

selected_model=models[mod]

selected_model.fit(X_train,y_train)

y_pred=selected_model.predict(X_test)

r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

# Display performance metrics
st.subheader('Model Performance Metrics')
st.write(f"**R² Score**: {r2:.2f}")
st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
st.write(f"**Mean Absolute Error (MAE)**: {mae:.2f}")

st.write("enter the input for pred")


user_input = {}
for column in X.columns:
    user_input[column] = st.number_input(
        column, 
        min_value=float(np.min(X[column])), 
        max_value=float(np.max(X[column])), 
        value=float(np.mean(X[column]))  # Default value is the column mean
    )

user_input_df=pd.DataFrame([user_input])

user_input_sc_df=scaler.transform(user_input_df)

predicted_price=selected_model.predict(user_input_sc_df)

st.subheader("Predicted House Price:")
st.write(f"${predicted_price[0] * 100000:,.2f}")