import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_option_menu import option_menu

from utils.auto_fix import auto_fix_dataset
from utils.model_recommender import recommend_model
from utils.data_quality import dataset_health_score
from utils.error_detection import detect_data_issues
from utils.cleaning_suggestions import cleaning_suggestions
from utils.automl_pipeline import run_automl
from utils.dashboard import missing_values_chart, feature_distribution, model_performance_chart
from utils.dataset_report import generate_dataset_report
from utils.leakage_detector import detect_leakage
from utils.explainability import explain_model


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="DataSage AI", layout="wide")

st.title("DataSage AI")
st.caption("AI-Powered Dataset Quality & AutoML Platform")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    selected = option_menu(
        "DataSage AI",
        [
            "Dataset Overview",
            "EDA Report",
            "Data Quality",
            "ML Analysis",
            "AutoML"
        ],
        icons=[
            "table",
            "bar-chart",
            "shield-check",
            "cpu",
            "robot"
        ],
        default_index=0
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

    # --------------------------------------------------
    # ABOUT PROJECT
    # --------------------------------------------------

    with st.expander("About this project"):

        st.write(
        """
**DataSage AI** is an AI-powered platform designed to automatically analyze dataset quality and accelerate machine learning workflows.

It helps data scientists and analysts quickly understand datasets, detect quality issues, clean data automatically, and discover the best machine learning models.

### Key Features

• Automated Exploratory Data Analysis (EDA)  
• Dataset Health Scoring  
• Data Quality Issue Detection  
• AI Data Cleaning Suggestions  
• Automatic Dataset Repair  
• ML Model Recommendation  
• AutoML Best Model Finder  
• Data Leakage Detection  
• Feature Importance & Explainability  
• AI Generated Dataset Reports  

### Technology Stack

Python  
Streamlit  
Pandas  
Scikit-learn  
Plotly  
ydata-profiling
        """
        )


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        st.session_state.df = pd.read_csv(uploaded_file)

    else:
        st.session_state.df = pd.read_excel(uploaded_file)


# ALWAYS USE SESSION DATA
df = st.session_state.df


# --------------------------------------------------
# IF DATASET NOT UPLOADED
# --------------------------------------------------

if df is None:

    st.info("Upload a CSV dataset using the sidebar to start analysis.")
    st.stop()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

if selected == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# EDA REPORT
# --------------------------------------------------

if selected == "EDA Report":

    st.header("EDA Report")

    if st.button("Generate Report"):

        with st.spinner("Generating EDA Report..."):

            profile = ProfileReport(df, explorative=True)

            html = profile.to_html()

            st.download_button(
                "Download Full EDA Report",
                html,
                file_name="eda_report.html"
            )

            st.success("EDA Report generated. Download for full interactive view.")


# --------------------------------------------------
# DATA QUALITY
# --------------------------------------------------

if selected == "Data Quality":

    st.header("Data Quality Analysis")

    if st.button("Analyze Dataset Quality"):

        results = dataset_health_score(df)

        st.metric("Dataset Health Score", f"{results['health_score']} / 100")

        col1, col2 = st.columns(2)

        col1.metric("Missing Values", results["missing_values"])
        col2.metric("Duplicate Rows", results["duplicate_rows"])

        st.write("Missing %:", results["missing_percentage"])
        st.write("Duplicate %:", results["duplicate_percentage"])

    st.subheader("Dataset Dashboard")

    fig = missing_values_chart(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    fig2 = feature_distribution(df)
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("AI Dataset Report")

    if st.button("Generate AI Dataset Report"):

        report = generate_dataset_report(df)

        for line in report:
            st.write("•", line)

    st.subheader("Error Detection")

    if st.button("Detect Data Issues"):

        issues = detect_data_issues(df)

        if len(issues) == 0:
            st.success("No major issues detected.")

        else:
            for issue in issues:
                st.warning(issue)

    st.subheader("Cleaning Suggestions")

    if st.button("Generate Cleaning Suggestions"):

        suggestions = cleaning_suggestions(df)

        if len(suggestions) == 0:
            st.success("Dataset looks clean.")

        else:
            for s in suggestions:
                st.info(s)

    st.subheader("Auto Fix Dataset")

    if st.button("Automatically Clean Dataset"):

        clean_df, fix_report = auto_fix_dataset(df)

        st.success("Dataset cleaned successfully")

        for r in fix_report:
            st.write("✔", r)

        st.dataframe(clean_df.head(), use_container_width=True)


# --------------------------------------------------
# ML ANALYSIS
# --------------------------------------------------

if selected == "ML Analysis":

    st.header("Machine Learning Analysis")

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run Model Recommendation"):

        models = recommend_model(df, target)

        st.dataframe(models, use_container_width=True)

    st.subheader("Data Leakage Detector")

    if st.button("Check Data Leakage"):

        leaks = detect_leakage(df, target)

        if len(leaks) == 0:
            st.success("No leakage detected")

        else:
            for l in leaks:
                st.warning(l)

    st.subheader("Feature Importance")

    if st.button("Generate Feature Importance"):

        importance = explain_model(df, target)

        st.dataframe(importance, use_container_width=True)


# --------------------------------------------------
# AUTOML
# --------------------------------------------------

if selected == "AutoML":

    st.header("AutoML Best Model Finder")

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):

        best_model, best_score, results_df = run_automl(df, target)

        st.success(f"Best Model: {best_model}")

        st.metric("Best Accuracy", round(best_score, 3))

        st.dataframe(results_df, use_container_width=True)

        fig = model_performance_chart(results_df)

        if fig:
            st.plotly_chart(fig, use_container_width=True)