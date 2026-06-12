"""
Customer Churn Prediction Dashboard
PyTorch + Streamlit
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from predict import predict


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "churn.csv"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "model.pth"
)

SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "scaler.pkl"
)

ENCODER_PATH = os.path.join(
    ARTIFACT_DIR,
    "label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_columns.pkl"
)

METRICS_PATH = os.path.join(
    ARTIFACT_DIR,
    "metrics.json"
)

CM_PATH = os.path.join(
    ARTIFACT_DIR,
    "confusion_matrix.npy"
)

ROC_PATH = os.path.join(
    ARTIFACT_DIR,
    "roc_data.npz"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b,
            #334155
        );
    }

    .main-title{
        font-size:45px;
        font-weight:700;
        color:white;
        text-align:center;
        margin-bottom:20px;
    }

    .sub-title{
        color:#CBD5E1;
        text-align:center;
        font-size:18px;
        margin-bottom:30px;
    }

    .metric-card{
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.25);
        text-align:center;
    }

    .section-title{
        color:white;
        font-size:24px;
        font-weight:bold;
        margin-top:20px;
        margin-bottom:10px;
    }

    .info-box{
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.25);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():

    if not os.path.exists(METRICS_PATH):
        return {}

    with open(
        METRICS_PATH,
        "r"
    ) as file:

        return json.load(file)


def artifacts_available():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        FEATURE_COLUMNS_PATH
    ]

    return all(
        os.path.exists(file)
        for file in required_files
    )


# =====================================================
# LOAD DATA
# =====================================================

df = load_dataset()

metrics = load_metrics()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🏦 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Prediction",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    ANN Customer Churn Prediction

    Built with:
    - PyTorch
    - Streamlit
    - Plotly
    """
)


# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="main-title">
        🏦 Customer Churn Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sub-title">
        End-to-End Customer Churn Prediction using PyTorch
        </div>
        """,
        unsafe_allow_html=True
    )

    if df is None:

        st.error(
            "Dataset not found in data/churn.csv"
        )

        st.stop()

    # ==========================================
    # KPI CARDS
    # ==========================================

    total_customers = len(df)

    churn_customers = int(
        df["Exited"].sum()
    )

    retention_rate = round(
        (1 - df["Exited"].mean()) * 100,
        2
    )

    feature_count = (
        df.shape[1]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "Churn Customers",
            f"{churn_customers:,}"
        )

    with col3:

        st.metric(
            "Retention Rate",
            f"{retention_rate}%"
        )

    with col4:

        st.metric(
            "Features",
            feature_count
        )

    st.markdown("---")

    # ==========================================
    # MODEL METRICS
    # ==========================================

    st.markdown(
        """
        <div class="section-title">
        🎯 Model Performance
        </div>
        """,
        unsafe_allow_html=True
    )

    if metrics:

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric(
                "Accuracy",
                f"{metrics.get('accuracy',0):.4f}"
            )

        with m2:
            st.metric(
                "Precision",
                f"{metrics.get('precision',0):.4f}"
            )

        with m3:
            st.metric(
                "Recall",
                f"{metrics.get('recall',0):.4f}"
            )

        with m4:
            st.metric(
                "F1 Score",
                f"{metrics.get('f1_score',0):.4f}"
            )

        with m5:
            st.metric(
                "ROC AUC",
                f"{metrics.get('roc_auc',0):.4f}"
            )

    else:

        st.warning(
            "Metrics not found. Run train.py first."
        )

    st.markdown("---")

    # ==========================================
    # DATASET PREVIEW
    # ==========================================

    st.markdown(
        """
        <div class="section-title">
        📄 Dataset Preview
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.markdown("---")

    # PART 2 STARTS HERE
    # ==========================================
    # CHURN DISTRIBUTION
    # ==========================================

    st.markdown(
        """
        <div class="section-title">
        📊 Customer Churn Distribution
        </div>
        """,
        unsafe_allow_html=True
    )

    try:

        churn_counts = (
            df["Exited"]
            .value_counts()
            .reset_index()
        )

        churn_counts.columns = [
            "Status",
            "Count"
        ]

        churn_counts["Status"] = (
            churn_counts["Status"]
            .map({
                0: "Stayed",
                1: "Exited"
            })
        )

        fig = px.pie(
            churn_counts,
            names="Status",
            values="Count",
            hole=0.55,
            title="Customer Churn Distribution"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Error loading churn chart: {e}"
        )

    st.markdown("---")

    # ==========================================
    # CORRELATION HEATMAP
    # ==========================================

    st.markdown(
        """
        <div class="section-title">
        🔥 Correlation Heatmap
        </div>
        """,
        unsafe_allow_html=True
    )

    try:

        numeric_df = df.select_dtypes(
            include=["number"]
        )

        corr = numeric_df.corr()

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1
            )
        )

        heatmap.update_layout(
            height=700
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Heatmap Error: {e}"
        )

    st.markdown("---")

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.markdown(
        """
        <div class="section-title">
        🎯 Top Features Affecting Churn
        </div>
        """,
        unsafe_allow_html=True
    )

    try:

        feature_importance = (
            corr["Exited"]
            .drop("Exited")
            .abs()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            x=feature_importance.values,
            y=feature_importance.index,
            orientation="h",
            title="Feature Importance (Correlation Based)"
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Feature Importance Error: {e}"
        )

    st.markdown("---")

    # ==========================================
    # GEOGRAPHY ANALYSIS
    # ==========================================

    if "Geography" in df.columns:

        st.markdown(
            """
            <div class="section-title">
            🌍 Churn Rate by Geography
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            geo = (
                df.groupby("Geography")
                ["Exited"]
                .mean()
                .reset_index()
            )

            geo["Exited"] = (
                geo["Exited"] * 100
            )

            fig = px.bar(
                geo,
                x="Geography",
                y="Exited",
                text="Exited",
                title="Churn Percentage by Country"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Geography Error: {e}"
            )

    st.markdown("---")

    # ==========================================
    # GENDER ANALYSIS
    # ==========================================

    if "Gender" in df.columns:

        st.markdown(
            """
            <div class="section-title">
            👨‍💼👩‍💼 Churn by Gender
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            gender_df = (
                df.groupby("Gender")
                ["Exited"]
                .mean()
                .reset_index()
            )

            gender_df["Exited"] = (
                gender_df["Exited"] * 100
            )

            fig = px.bar(
                gender_df,
                x="Gender",
                y="Exited",
                text="Exited"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Gender Analysis Error: {e}"
            )

    st.markdown("---")

    # ==========================================
    # AGE DISTRIBUTION
    # ==========================================

    if "Age" in df.columns:

        st.markdown(
            """
            <div class="section-title">
            🎂 Age Distribution
            </div>
            """,
            unsafe_allow_html=True
        )

        fig = px.histogram(
            df,
            x="Age",
            nbins=30,
            title="Customer Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================
    # BALANCE DISTRIBUTION
    # ==========================================

    if "Balance" in df.columns:

        st.markdown(
            """
            <div class="section-title">
            💰 Balance Distribution
            </div>
            """,
            unsafe_allow_html=True
        )

        fig = px.histogram(
            df,
            x="Balance",
            nbins=30,
            title="Account Balance Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================
    # ROC CURVE
    # ==========================================

    st.markdown(
        """
        <div class="section-title">
        📈 ROC Curve
        </div>
        """,
        unsafe_allow_html=True
    )

    try:

        if os.path.exists(ROC_PATH):

            roc_data = np.load(
                ROC_PATH
            )

            fpr = roc_data["fpr"]
            tpr = roc_data["tpr"]

            roc_fig = go.Figure()

            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name="ROC Curve"
                )
            )

            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random"
                )
            )

            roc_fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500
            )

            st.plotly_chart(
                roc_fig,
                use_container_width=True
            )

        else:

            st.warning(
                "ROC data not found. Run train.py first."
            )

    except Exception as e:

        st.error(
            f"ROC Curve Error: {e}"
        )

    st.markdown("---")

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    st.markdown(
        """
        <div class="section-title">
        🎯 Confusion Matrix
        </div>
        """,
        unsafe_allow_html=True
    )

    try:

        if os.path.exists(CM_PATH):

            cm = np.load(
                CM_PATH
            )

            cm_fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=[
                        "Predicted No",
                        "Predicted Yes"
                    ],
                    y=[
                        "Actual No",
                        "Actual Yes"
                    ]
                )
            )

            cm_fig.update_layout(
                height=500
            )

            st.plotly_chart(
                cm_fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Confusion Matrix not found."
            )

    except Exception as e:

        st.error(
            f"Confusion Matrix Error: {e}"
        )

    # PART 3 STARTS HERE

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.markdown(
        """
        <div class="main-title">
        🔮 Customer Churn Prediction
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sub-title">
        Enter customer details and predict churn probability
        </div>
        """,
        unsafe_allow_html=True
    )

    if not artifacts_available():

        st.error(
            """
            Required artifacts not found.

            Please run:

            python train.py

            before using predictions.
            """
        )

        st.stop()

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=650
            )

            geography = st.selectbox(
                "Geography",
                [
                    "France",
                    "Germany",
                    "Spain"
                ]
            )

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female"
                ]
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

            tenure = st.number_input(
                "Tenure",
                min_value=0,
                max_value=10,
                value=5
            )

        with col2:

            balance = st.number_input(
                "Balance",
                min_value=0.0,
                value=50000.0
            )

            products = st.number_input(
                "Number Of Products",
                min_value=1,
                max_value=4,
                value=2
            )

            card = st.selectbox(
                "Has Credit Card",
                [0, 1]
            )

            active_member = st.selectbox(
                "Is Active Member",
                [0, 1]
            )

            salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=60000.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn"
        )

    if submitted:

        try:

            sample = pd.DataFrame({

                "CreditScore": [credit_score],

                "Geography": [geography],

                "Gender": [gender],

                "Age": [age],

                "Tenure": [tenure],

                "Balance": [balance],

                "NumOfProducts": [products],

                "HasCrCard": [card],

                "IsActiveMember": [active_member],

                "EstimatedSalary": [salary]

            })

            probability, prediction = predict(
                sample
            )

            st.markdown("---")

            st.subheader(
                "Prediction Result"
            )

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={
                        "text":
                        "Churn Risk (%)"
                    },
                    gauge={
                        "axis": {
                            "range":
                            [0, 100]
                        }
                    }
                )
            )

            gauge.update_layout(
                height=400
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            if prediction == "Leave":

                st.error(
                    f"""
                    High Churn Risk

                    Prediction: {prediction}

                    Probability:
                    {probability:.2%}
                    """
                )

            else:

                st.success(
                    f"""
                    Customer Likely To Stay

                    Prediction: {prediction}

                    Probability:
                    {probability:.2%}
                    """
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.markdown(
        """
        <div class="main-title">
        ℹ️ About Project
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <h3>Project Overview</h3>

        This project predicts whether a customer
        is likely to leave the bank using a
        PyTorch Artificial Neural Network.

        The application includes:

        ✔ Data Analysis

        ✔ Feature Visualization

        ✔ ANN Training

        ✔ Churn Prediction

        ✔ Interactive Dashboard

        ✔ Streamlit Deployment

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 🛠 Technologies

            - Python

            - PyTorch

            - Streamlit

            - Plotly

            - NumPy

            - Pandas

            - Scikit-Learn

            - Joblib
            """
        )

    with col2:

        st.markdown(
            """
            ### ⚙ Workflow

            1. Data Collection

            2. Data Preprocessing

            3. Feature Engineering

            4. ANN Training

            5. Evaluation

            6. Prediction

            7. Deployment
            """
        )

    st.markdown("---")

    st.markdown(
        """
        ### 🧠 ANN Architecture

        Input Layer

        ↓

        Linear(128)

        ↓

        ReLU

        ↓

        Dropout(0.3)

        ↓

        Linear(64)

        ↓

        ReLU

        ↓

        Dropout(0.3)

        ↓

        Linear(32)

        ↓

        ReLU

        ↓

        Linear(1)

        ↓

        Sigmoid
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🚀 Deployment

        Compatible With:

        - Local Machine

        - GitHub

        - Streamlit Cloud

        - Linux Servers

        - Windows Systems
        """
    )

    st.success(
        """
        ANN Customer Churn Prediction
        Project Ready For Deployment
        """
    )

