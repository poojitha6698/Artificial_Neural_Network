import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.predict import predict

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

.metric-card{
    background:#ffffff;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
    text-align:center;
}

.big-title{
    font-size:42px;
    font-weight:800;
    color:#1f77b4;
}

.subtitle{
    font-size:18px;
    color:gray;
}

.prediction-box{
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <div class='big-title'>
    🚀 Employee Attrition Intelligence Dashboard
    </div>

    <div class='subtitle'>
    Artificial Neural Network powered HR Analytics System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("📋 Employee Information")

Age = st.sidebar.slider(
    "Age",
    18,
    60,
    30
)

MonthlyIncome = st.sidebar.slider(
    "Monthly Income",
    1000,
    50000,
    10000
)

DistanceFromHome = st.sidebar.slider(
    "Distance From Home",
    1,
    30,
    5
)

TotalWorkingYears = st.sidebar.slider(
    "Total Working Years",
    0,
    40,
    8
)

YearsAtCompany = st.sidebar.slider(
    "Years At Company",
    0,
    40,
    5
)

JobLevel = st.sidebar.selectbox(
    "Job Level",
    [1, 2, 3, 4, 5]
)

PercentSalaryHike = st.sidebar.slider(
    "Salary Hike %",
    10,
    30,
    15
)

NumCompaniesWorked = st.sidebar.slider(
    "Companies Worked",
    0,
    10,
    2
)

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Age",
        Age
    )

with col2:
    st.metric(
        "Income",
        f"₹{MonthlyIncome:,}"
    )

with col3:
    st.metric(
        "Experience",
        TotalWorkingYears
    )

with col4:
    st.metric(
        "Job Level",
        JobLevel
    )

st.markdown("---")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button(
    "🔮 Predict Employee Attrition",
    use_container_width=True
):

    input_df = pd.DataFrame({
        "Age":[Age],
        "MonthlyIncome":[MonthlyIncome],
        "DistanceFromHome":[DistanceFromHome],
        "TotalWorkingYears":[TotalWorkingYears],
        "YearsAtCompany":[YearsAtCompany],
        "JobLevel":[JobLevel],
        "PercentSalaryHike":[PercentSalaryHike],
        "NumCompaniesWorked":[NumCompaniesWorked]
    })

    probability = predict(input_df)

    probability = float(probability)

    risk_percent = round(
        probability * 100,
        2
    )

    st.markdown("## 📊 Prediction Result")

    # ---------------------------
    # Gauge Chart
    # ---------------------------

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            title={
                "text":"Attrition Risk (%)"
            },
            gauge={
                "axis":{
                    "range":[0,100]
                },
                "bar":{
                    "thickness":0.3
                },
                "steps":[
                    {"range":[0,40]},
                    {"range":[40,70]},
                    {"range":[70,100]}
                ]
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ---------------------------
    # Result Card
    # ---------------------------

    if probability > 0.5:

        st.error(
            f"""
            ⚠️ High Attrition Risk

            Probability: {risk_percent}%
            """
        )

    else:

        st.success(
            f"""
            ✅ Low Attrition Risk

            Probability: {risk_percent}%
            """
        )

    # ---------------------------
    # Insights
    # ---------------------------

    st.markdown("## 🧠 ANN Insights")

    insights = []

    if MonthlyIncome < 5000:
        insights.append(
            "Low income may contribute to employee dissatisfaction."
        )

    if DistanceFromHome > 15:
        insights.append(
            "Long commute distance may increase attrition risk."
        )

    if YearsAtCompany < 2:
        insights.append(
            "New employees often have higher turnover rates."
        )

    if NumCompaniesWorked > 5:
        insights.append(
            "Frequent job changes indicate higher mobility."
        )

    if len(insights) == 0:
        insights.append(
            "Employee profile appears relatively stable."
        )

    for item in insights:
        st.info(item)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>

    Built with ❤️ using
    TensorFlow, ANN, Plotly and Streamlit

    </center>
    """,
    unsafe_allow_html=True
)