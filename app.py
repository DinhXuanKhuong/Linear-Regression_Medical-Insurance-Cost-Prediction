import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# -------------------------------------------------
# 1. Load pipeline
# -------------------------------------------------
pipeline = load("poly_regression_pipeline.joblib")

# -------------------------------------------------
# 2. Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="",
    layout="centered"
)

st.title("Medical Insurance Charge Prediction")
st.write(
    "Drag the **Age** and **BMI** sliders to instantly see their impact on insurance costs."
    "All charts and predictions update in **real time**."
)

# -------------------------------------------------
# 3. INPUTS - Thay number_input bằng SLIDER cho Age & BMI
# -------------------------------------------------
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("**Age**", min_value=18, max_value=70, value=30, step=1)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.slider("**BMI**", min_value=15.0, max_value=50.0, value=25.0, step=0.1)

    with col2:
        children = st.number_input("Children", min_value=0, max_value=15, value=0, step=1)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# -------------------------------------------------
# 4. Helper: build input DataFrame
# -------------------------------------------------
def build_input_df(age, bmi, children, smoker, sex, region):
    return pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "male" else 0],
        "bmi": [bmi],
        "children": [children],
        "smoker": [1 if smoker == "yes" else 0],
        "northeast": [1 if region == "northeast" else 0],
        "northwest": [1 if region == "northwest" else 0],
        "southeast": [1 if region == "southeast" else 0],
        "southwest": [1 if region == "southwest" else 0],
    })

model_cols = ["age", "bmi", "smoker"]

# -------------------------------------------------
# 5. TỰ ĐỘNG DỰ ĐOÁN (không cần button)
# -------------------------------------------------
input_df = build_input_df(age, bmi, children, smoker, sex, region)
try:
    pred = pipeline.predict(input_df[model_cols])[0]
    st.markdown(
        f"<h2 style='text-align: center; color: #2e8b57;'>"
        f"Estimated Charge: <span style='color:#d62728;'>${pred:,.2f}</span>"
        f"</h2>",
        unsafe_allow_html=True
    )
except Exception as e:
    st.error("Error!!. Please recheck pipeline.")
    st.exception(e)

# -------------------------------------------------
# 6. VISUALISATIONS - Cập nhật live theo slider
# -------------------------------------------------
st.markdown("---")
st.subheader("Effects of Age and BMI on Charge")

# Cache để tăng tốc
@st.cache_data(show_spinner=False)
def compute_partial_dependence(feature, grid, fixed_df):
    preds = []
    for val in grid:
        df = fixed_df.copy()
        df[feature] = val
        preds.append(pipeline.predict(df[model_cols])[0])
    return np.array(preds)

# Fixed DataFrame từ input hiện tại
fixed_df = build_input_df(age, bmi, children, smoker, sex, region)

# ------------------------------------------------------------------
# 6.1 Biểu đồ Age & BMI effect (có điểm hiện tại)
# ------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.write("**Age effect** (fixed others)")
    age_grid = np.linspace(18, 70, 100)
    age_pred = compute_partial_dependence("age", age_grid, fixed_df)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(age_grid, age_pred, color="#1f77b4", linewidth=2, label="Charge curve")
    ax.scatter([age], [pred], color="red", s=80, zorder=5, label=f"Current ({age})")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charge ($)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

with col_b:
    st.write("**BMI effect** (fixed others)")
    bmi_grid = np.linspace(15, 50, 100)
    bmi_pred = compute_partial_dependence("bmi", bmi_grid, fixed_df)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(bmi_grid, bmi_pred, color="#ff7f0e", linewidth=2, label="Charge curve")
    ax.scatter([bmi], [pred], color="red", s=80, zorder=5, label=f"Current ({bmi})")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charge ($)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ------------------------------------------------------------------
# 6.2 Heatmap Age × BMI (live theo smoker)
# ------------------------------------------------------------------
st.markdown("#### Heat Map: Age × BMI")
smoker_val = 1 if smoker == "yes" else 0
smoker_title = "Smoker" if smoker == "yes" else "Non-smoker"

age_grid_2d = np.linspace(18, 70, 60)
bmi_grid_2d = np.linspace(15, 50, 60)
A, B = np.meshgrid(age_grid_2d, bmi_grid_2d)
Z = np.zeros_like(A)

base_df = fixed_df.copy()
base_df["smoker"] = smoker_val

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        temp_df = base_df.copy()
        temp_df["age"] = A[i, j]
        temp_df["bmi"] = B[i, j]
        Z[i, j] = pipeline.predict(temp_df[model_cols])[0]

fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(A, B, Z, levels=30, cmap="RdYlBu_r")
fig.colorbar(cs, ax=ax, label="Charge ($)")
ax.scatter([age], [bmi], color="black", s=100, zorder=5, edgecolors='white', linewidth=1.5)
ax.set_xlabel("Age")
ax.set_ylabel("BMI")
ax.set_title(f"Charge Heatmap – {smoker_title}")
st.pyplot(fig, use_container_width=True)

# -------------------------------------------------
# 7. Footer
# -------------------------------------------------
st.markdown("---")
st.caption("NDLLS - 23127289 - 23127351 - 23127354 - 23127398 - 23127538")