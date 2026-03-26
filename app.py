import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Placement Predictor", layout="wide")

# Load model
model = pickle.load(open("placement_model.pkl","rb"))

# Title
st.markdown("""
# 🎓 Student Placement Prediction Dashboard
### Predict student placement probability using Machine Learning
---
""")

# Create 3 columns layout
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male","Female"])
    ssc_p = st.slider("10th %", 0.0, 100.0)
    ssc_b = st.selectbox("10th Board", ["Central","Others"])

    hsc_p = st.slider("12th %", 0.0, 100.0)
    hsc_b = st.selectbox("12th Board", ["Central","Others"])
    hsc_s = st.selectbox("Stream", ["Arts","Commerce","Science"])

with col2:
    degree_p = st.slider("Degree %", 0.0, 100.0)
    degree_t = st.selectbox("Degree Type", ["Comm&Mgmt","Sci&Tech","Others"])
    workex = st.selectbox("Work Experience", ["Yes","No"])

    etest_p = st.slider("Employability Test %", 0.0, 100.0)
    specialisation = st.selectbox("MBA Specialisation", ["Mkt&Fin","Mkt&HR"])
    mba_p = st.slider("MBA %", 0.0, 100.0)

with col3:
    st.markdown("### 📊 Quick Summary")
    st.metric("10th %", ssc_p)
    st.metric("12th %", hsc_p)
    st.metric("Degree %", degree_p)
    st.metric("MBA %", mba_p)

# Convert inputs
gender = 1 if gender == "Male" else 0
ssc_b = 0 if ssc_b == "Central" else 1
hsc_b = 0 if hsc_b == "Central" else 1

hsc_s_map = {"Arts":0,"Commerce":1,"Science":2}
hsc_s = hsc_s_map[hsc_s]

degree_map = {"Comm&Mgmt":0,"Sci&Tech":2,"Others":1}
degree_t = degree_map[degree_t]

workex = 1 if workex == "Yes" else 0
specialisation = 0 if specialisation == "Mkt&Fin" else 1

# Prediction button
st.markdown("---")

if st.button("🔍 Predict Placement", use_container_width=True):

    data = pd.DataFrame([[gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,
                          degree_p,degree_t,workex,etest_p,
                          specialisation,mba_p]],
                        columns=[
                        "gender","ssc_p","ssc_b","hsc_p","hsc_b",
                        "hsc_s","degree_p","degree_t","workex",
                        "etest_p","specialisation","mba_p"])

    prediction = model.predict(data)
    prob = model.predict_proba(data)

    placement_prob = prob[0][1] * 100

    st.markdown("## 📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Placement Probability", f"{placement_prob:.2f}%")

    with colB:
        if prediction[0] == 1:
            st.success("🎉 Likely to be PLACED")
        else:
            st.error("❌ Not Likely to be Placed")

    # Progress bar
    st.progress(int(placement_prob))


# Feature Importance Section
st.markdown("---")
st.markdown("## 📈 Factors Affecting Placement")

importances = model.feature_importances_
features = [
"gender","ssc_p","ssc_b","hsc_p","hsc_b",
"hsc_s","degree_p","degree_t","workex",
"etest_p","specialisation","mba_p"
]

fig, ax = plt.subplots(figsize=(8,5))
ax.barh(features, importances)
ax.set_title("Feature Importance")
ax.set_xlabel("Impact")

st.pyplot(fig)