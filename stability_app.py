import streamlit as st
import pandas as pd
import joblib

# --- Load model and data mappings ---
@st.cache_resource
def load_model(path='models/regressor_pvalue.joblib'):
    return joblib.load(path)

@st.cache_data
def load_classes(class_file='data/data2_classified.csv'):
    df = pd.read_csv(class_file)
    # Classes are numeric labels from classification
    return sorted(df['Predicted_Class_All'].unique())

model = load_model()
classes = load_classes()

# --- App UI ---
st.title("Stability Index Predictor")

st.markdown("Select the two crude classes and their mixing ratio:")
class1 = st.selectbox('Sample 1 Class', classes, index=0)
class2 = st.selectbox('Sample 2 Class', classes, index=1)

p1 = st.slider('Percentage of Sample 1 (%)', 0, 100, 50)
p2 = 100 - p1
st.write(f"Percentage of Sample 2: {p2}%")

def classify_stability(p):
    if p > 2:
        return 'stable'
    elif p < 1.5:
        return 'unstable'
    else:
        return 'lower_stability'

# --- Prediction ---
if st.button('Predict Stability Index'):
    data = pd.DataFrame([{
        'Class1': class1,
        'Class2': class2,
        'P1': p1,
        'P2': p2
    }])
    pred = model.predict(data)[0]
    stability = classify_stability(pred)
    st.success(f"Predicted Stability Index: {pred:.2f}")
    st.info(f"Stability: **{stability}**")

# --- Footer ---
st.markdown("---")
st.caption("Model: RandomForestRegressor trained on P-Value mixtures")
