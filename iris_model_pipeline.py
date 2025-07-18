import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Species Prediction")

iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider('Sepal length', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width  = st.sidebar.slider('Sepal width', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.sidebar.slider('Petal length', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width  = st.sidebar.slider('Petal width', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
species = iris.target_names[prediction]

st.write("### Predicted Species:")
st.success(f"🌼 {species.capitalize()}")
