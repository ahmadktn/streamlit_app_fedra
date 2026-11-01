import streamlit as st
import sys
import os

# Ensure src directory is in python path
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

st.set_page_config(
    page_title="FEDRA 3.0: Nigerian Student Dropout & Fairness Suite",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Multi-Page Navigation with Material Symbols
nav = st.navigation(
    [
        st.Page(
            "app_pages/cohort_monitor.py",
            title="Cohort early warning",
            icon=":material/analytics:",
            default=True
        ),
        st.Page(
            "app_pages/eda_patterns.py",
            title="EDA & risk patterns",
            icon=":material/insights:"
        ),
        st.Page(
            "app_pages/student_simulator.py",
            title="Student risk simulator",
            icon=":material/person_search:"
        ),
        st.Page(
            "app_pages/model_benchmarks.py",
            title="Model benchmarks & lab",
            icon=":material/speed:"
        ),
        st.Page(
            "app_pages/fairness_audit.py",
            title="Fairness audit & mitigation",
            icon=":material/balance:"
        ),
    ],
    position="sidebar"
)

nav.run()
