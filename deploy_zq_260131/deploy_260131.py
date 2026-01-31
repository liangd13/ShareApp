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
model_path = os.path.join(current_dir, 'model_260131.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# 初始化SHAP解释器（XGBoost专用，放在模型加载后）
explainer = shap.TreeExplainer(model)  # 直接用原始XGB模型初始化
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 设置 Streamlit 应用的标题
st.title("Prediction of AKI based on XGBoost model")

st.sidebar.header("Selection Panel")
# st.sidebar.subheader("Picking up parameters")

# 二分类变量
HBP_option = ["Yes", "No"]
HBP_map = {"Yes": 1, "No": 0}
HBP_sb = st.sidebar.selectbox("HBP", HBP_option, index=0)

HF_option = ["Yes", "No"]
HF_map = {"Yes": 1, "No": 0}
HF_sb = st.sidebar.selectbox("HF", HF_option, index=0)

Pneumonia_option = ["Yes", "No"]
Pneumonia_map = {"Yes": 1, "No": 0}
Pneumonia_sb = st.sidebar.selectbox("Pneumonia", Pneumonia_option, index=0)

Sepsis_option = ["Yes", "No"]
Sepsis_map = {"Yes": 1, "No": 0}
Sepsis_sb = st.sidebar.selectbox("Sepsis", Sepsis_option, index=0)

Ventilation_option = ["Yes", "No"]
Ventilation_map = {"Yes": 1, "No": 0}
Ventilation_sb = st.sidebar.selectbox("Ventilation", Ventilation_option, index=0)

# 连续变量
RR = st.sidebar.slider("RR (times/minute)", min_value=9, max_value=40, value=20, step=1)

LDL = st.sidebar.slider("LDL (mg/dL)", min_value=17, max_value=182, value=100, step=1)

PO2 = st.sidebar.slider("PO2 (mmHg)", min_value=18, max_value=300, value=100, step=1)

Lactate = st.sidebar.slider("Lactate (mmol/L)", min_value=0.5, max_value=24.0, value=2.0, step=0.1)

OASIS = st.sidebar.slider("OASIS", min_value=11, max_value=59, value=30, step=1)

input_data = pd.DataFrame({
    'HBP': [HBP_map[HBP_sb]],
    'HF': [HF_map[HF_sb]],
    'Pneumonia': [Pneumonia_map[Pneumonia_sb]],
    'Sepsis': [Sepsis_map[Sepsis_sb]],
    'Ventilation': [Ventilation_map[Ventilation_sb]],
    'RR': [RR],
    'LDL': [LDL],
    'PO2': [PO2],
    'Lactate': [Lactate],
    'OASIS': [OASIS]
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
