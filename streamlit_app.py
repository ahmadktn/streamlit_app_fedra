import streamlit as st
import sys
import os

# Ensure src directory is in python path
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fedra_app.pipeline import (
    FACULTIES,
    FACULTY_DEPARTMENTS_MAP,
    ALL_DEPARTMENTS,
    get_faculty_for_department
)

st.set_page_config(
    page_title="FEDRA 3.0: Nigerian Student Dropout & Fairness Suite",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Department Scoping in Sidebar (No Emojis)
with st.sidebar:
    st.markdown("### Department Scope")
    st.caption("Isolate analysis to one academic department across all pages.")
    
    scope_faculty = st.selectbox(
        "Filter by Faculty",
        ["All Faculties"] + FACULTIES,
        index=0,
        key="sidebar_scope_faculty"
    )
    
    if scope_faculty != "All Faculties":
        dept_options = FACULTY_DEPARTMENTS_MAP.get(scope_faculty, ALL_DEPARTMENTS)
    else:
        dept_options = ALL_DEPARTMENTS
        
    full_scope_options = ["All Departments (Overview)"] + dept_options
    
    if "active_department" not in st.session_state:
        st.session_state["active_department"] = "Computer Science"
        
    current_dept = st.session_state["active_department"]
    current_idx = full_scope_options.index(current_dept) if current_dept in full_scope_options else 0
    
    selected_dept = st.selectbox(
        "Active Department",
        full_scope_options,
        index=current_idx,
        key="sidebar_active_department"
    )
    st.session_state["active_department"] = selected_dept
    
    if selected_dept != "All Departments (Overview)":
        parent_fac = get_faculty_for_department(selected_dept)
        st.info(f"Focused Program: {selected_dept}\n\nFaculty: {parent_fac}")
    else:
        st.caption("University overview mode. Select a specific department above to isolate its cohort.")
        
    st.divider()

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
            "app_pages/batch_prediction.py",
            title="Batch CSV prediction",
            icon=":material/upload_file:"
        ),
        st.Page(
            "app_pages/student_simulator.py",
            title="Student risk simulator",
            icon=":material/person_search:"
        ),
        st.Page(
            "app_pages/eda_patterns.py",
            title="EDA & risk patterns",
            icon=":material/insights:"
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
