import streamlit as st
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'xgboost_model_260102.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# 初始化SHAP解释器（XGBoost专用，放在模型加载后）
explainer = shap.TreeExplainer(model)  # 直接用原始XGB模型初始化
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 设置 Streamlit 应用的标题
st.title("Prediction of CVD based on XGBoost model")

st.sidebar.header("Selection Panel")
# st.sidebar.subheader("Picking up parameters")

Smoke_option = ["Yes", "No"]
Smoke_map = {"Yes": 1, "No": 0}
Smoke_sb = st.sidebar.selectbox("Smoke", Smoke_option, index=0)

Hypertension_option = ["Yes", "No"]
Hypertension_map = {"Yes": 1, "No": 0}
Hypertension_sb = st.sidebar.selectbox("Hypertension", Hypertension_option, index=1)

Diabetes_option = ["Yes", "No"]
Diabetes_map = {"Yes": 1, "No": 0}
Diabetes_sb = st.sidebar.selectbox("Diabetes", Diabetes_option, index=0)

CAD_FH_option = ["Yes", "No"]
CAD_FH_map = {"Yes": 1, "No": 0}
CAD_FH_sb = st.sidebar.selectbox("CAD_FH", CAD_FH_option, index=0)

Age = st.sidebar.slider("Age, year", min_value=18, max_value=80, value=55, step=1)

RBC = st.sidebar.slider("RBC", min_value=2.21, max_value=5.98, value=4.0, step=0.01)

SDAI = st.sidebar.slider("SDAI", min_value=0.0, max_value=62.8, value=10.0, step=0.1)

input_data = pd.DataFrame({
    'Age': [Age],
    'Smoke': [Smoke_map[Smoke_sb]],
    'Hypertension': [Hypertension_map[Hypertension_sb]],
    'Diabetes': [Diabetes_map[Diabetes_sb]],
    'CAD_FH': [CAD_FH_map[CAD_FH_sb]],
    'RBC': [RBC],
    'SDAI': [SDAI]
})

if st.button("Calculate"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    # st.write(f"Predictive Probability: {final_pred_proba:.2f}%")
    st.markdown(f"### Predictive Probability: {final_pred_proba:.2f}%", unsafe_allow_html=True)

    shap_values = explainer.shap_values(input_data)
    shap_vals = shap_values[0] if len(shap_values.shape) == 2 else shap_values
    base_val = float(explainer.expected_value)

    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=input_data.iloc[0],
        feature_names=input_data.columns
    )
    # 绘图展示（去掉ax参数）
    st.subheader("SHAP Feature Contribution (Waterfall Plot)")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf())
