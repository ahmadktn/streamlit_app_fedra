import sys
import os
from pathlib import Path
import importlib
import streamlit as st
import pandas as pd
import numpy as np

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import fedra_app.pipeline
importlib.reload(fedra_app.pipeline)

from fedra_app.pipeline import (
    train_suite_models,
    predict_batch,
    get_csv_template,
    get_sample_dataframe,
    FACULTIES,
    STATES,
    FACULTY_DEPARTMENTS_MAP,
    ALL_DEPARTMENTS,
    get_faculty_for_department
)
from fedra_app.visualizations import (
    plot_batch_risk_distribution,
    plot_batch_priority_donut,
    plot_batch_faculty_bars,
    plot_department_risk_bars
)

st.header("Batch student risk prediction & CSV upload", anchor=False)
st.caption("Upload institutional rosters or departmental cohorts to run FEDRA 3.0 multi-scale ensemble attrition predictions, isolate department-level risks, and generate actionable intervention recommendations.")

with st.spinner("Initializing FEDRA 3.0 ensemble pipeline..."):
    artifacts = train_suite_models()
    th_fedra = artifacts['eval']['FEDRA 3.0']['threshold']

# ==============================================================================
# 1. TEMPLATE DOWNLOAD & SCHEMA GUIDANCE (WITH DEPARTMENT CUSTOMIZATION)
# ==============================================================================
with st.container(border=True):
    st.subheader("📥 Departmental CSV Templates & Format Specification", anchor=False)
    st.markdown(
        "Download a pre-formatted template to ensure your institutional dataset matches the FEDRA 3.0 schema. "
        "Because different departments have different academic loads and risk factors, you can select your **specific department** "
        "to download a tailored blank header or sample data populated with realistic Nigerian student records."
    )
    
    col_t_dept, col_dl1, col_dl2, col_demo = st.columns([1.2, 1, 1, 1.2])
    
    with col_t_dept:
        dept_template_choice = st.selectbox(
            "Select Department for Template",
            ["All Departments (General)"] + ALL_DEPARTMENTS,
            index=0,
            help="Generates templates pre-aligned with this department's faculty and benchmarks."
        )
        target_dept_param = None if dept_template_choice == "All Departments (General)" else dept_template_choice
        
    with col_dl1:
        blank_csv = get_csv_template(sample=False, department=target_dept_param)
        dept_slug = dept_template_choice.lower().replace(" ", "_") if target_dept_param else "all_departments"
        st.download_button(
            label="📄 Blank template",
            data=blank_csv,
            file_name=f"fedra3_template_{dept_slug}.csv",
            mime="text/csv",
            help=f"CSV containing the canonical column headers customized for {dept_template_choice}.",
            use_container_width=True
        )
        
    with col_dl2:
        sample_csv = get_csv_template(sample=True, department=target_dept_param)
        st.download_button(
            label="📊 Sample data (10)",
            data=sample_csv,
            file_name=f"fedra3_sample_{dept_slug}.csv",
            mime="text/csv",
            help=f"CSV with 10 representative Nigerian student profiles for {dept_template_choice}.",
            use_container_width=True
        )
        
    with col_demo:
        demo_btn_lbl = f"⚡ Load demo ({target_dept_param[:14]}..)" if target_dept_param and len(target_dept_param) > 16 else f"⚡ Load demo ({target_dept_param or '10 students'})"
        if st.button(demo_btn_lbl, use_container_width=True, type="secondary"):
            st.session_state['batch_df_raw'] = get_sample_dataframe(department=target_dept_param)
            st.session_state['batch_source_name'] = f"Demo ({dept_template_choice})"
            if 'batch_predictions' in st.session_state:
                del st.session_state['batch_predictions']
            st.rerun()

    with st.expander("📋 View Expected CSV Column Schema & Guidelines", expanded=False):
        c_sch1, c_sch2 = st.columns(2)
        with c_sch1:
            st.markdown("""
            **Academic & Behavioral Indicators:**
            - `StudentID`: Unique student identifier (e.g. `STU-1001`)
            - `GPA`: Current estimated GPA on a 5.0 scale (e.g. `3.45`)
            - `PreviousSemesterGPA`: Prior term GPA on 5.0 scale (e.g. `3.20`)
            - `AttendanceRate`: Class attendance percentage `0`–`100` (e.g. `85.0` or `85%`)
            - `StudyHoursPerWeek`: Weekly independent study hours (e.g. `15`)
            - `FailedCourses`: Number of registered courses failed (e.g. `0`, `2`)
            - `CoursesCreditLoad`: Total credit unit load (e.g. `18`)
            - `MentoringSessionsAttended`: Peer/faculty mentoring sessions (e.g. `2`)
            - `CounsellingVisits`: Institutional guidance visits (e.g. `0`)
            - `LibraryVisitsPerMonth`: Monthly physical/digital library visits (e.g. `4`)
            """)
        with c_sch2:
            st.markdown("""
            **Socioeconomic, Demographic & Institutional Context:**
            - `TuitionPaid`: Tuition fee status (`Paid` or `Unpaid`, `1` or `0`)
            - `MonthlyAllowance`: Monthly living allowance in Naira (e.g. `45000` or `45k`)
            - `FundingSource`: `Self Funded`, `Government Loan`, `Partial Scholarship`, `Full Scholarship`, `Family Support`
            - `Housing`: `Off Campus`, `On Campus`, `Family Home`
            - `DistanceFromHome`: Commute distance from residence in km (e.g. `35`)
            - `Faculty`: Target faculty (e.g. `Faculty of Science`, `Faculty of Engineering`)
            - `Department`: Student major/department (e.g. `Computer Science`)
            - `YearOfStudy`: Academic year level (`1`, `2`, `3`, `4`, `5`)
            - `State`: State of origin (e.g. `Lagos`, `Kano`, `Plateau`)
            - `Gender`: `Male`, `Female`
            """)
        st.caption("ℹ️ *Note: Missing or unrecognized columns will automatically fall back to balanced median/mode baseline values so prediction never fails.*")

st.divider()

# ==============================================================================
# 2. FILE UPLOADER & DATA PREPARATION
# ==============================================================================
st.subheader("📤 Upload Student CSV for Prediction", anchor=False)

uploaded_file = st.file_uploader(
    "Choose a CSV file containing student records",
    type=["csv"],
    help="Upload your formatted CSV file to initiate batch scoring."
)

# Handle file upload
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in raw_df.columns:
            raw_df.drop(columns=['Unnamed: 0'], inplace=True)
            
        st.session_state['batch_df_raw'] = raw_df
        st.session_state['batch_source_name'] = uploaded_file.name
        # Clear previous predictions if file changed
        if st.session_state.get('last_uploaded_name') != uploaded_file.name:
            st.session_state['last_uploaded_name'] = uploaded_file.name
            if 'batch_predictions' in st.session_state:
                del st.session_state['batch_predictions']
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")

# Check if data is loaded in session
if 'batch_df_raw' in st.session_state:
    df_current = st.session_state['batch_df_raw']
    src_name = st.session_state.get('batch_source_name', 'Uploaded File')
    
    st.info(f"📁 Loaded dataset: **{src_name}** | **{len(df_current):,}** student records detected across **{len(df_current.columns)}** columns.")
    
    with st.expander("👁️ Preview Raw Uploaded Data (First 5 Rows)", expanded=False):
        st.dataframe(df_current.head(5), width='stretch')

    # Department Isolation Control
    depts_found = sorted([str(d) for d in df_current['Department'].dropna().unique() if str(d).strip() != '']) if 'Department' in df_current.columns else []
    
    with st.container(border=True):
        st.subheader("🎯 Department Isolation & Focus Analysis", anchor=False)
        st.markdown(
            "Different departments possess distinct grading curves, course credit loads, and dropout dynamics. "
            "To isolate departmental risk profiles and avoid confounding effects across disparate programs, "
            "select **one department at a time** below for dedicated analysis, triage, and targeted export."
        )
        col_dept_sel, col_dept_info = st.columns([1.5, 2])
        with col_dept_sel:
            if depts_found:
                dept_options = ["All Departments (Overview)"] + depts_found
                selected_dept_focus = st.selectbox(
                    "Active Department Focus",
                    dept_options,
                    index=0,
                    help="Isolate all downstream KPIs, risk distributions, and triage recommendations to this specific department."
                )
            else:
                selected_dept_focus = "All Departments (Overview)"
                st.caption("No 'Department' column detected in dataset; analyzing complete cohort.")
        with col_dept_info:
            if selected_dept_focus != "All Departments (Overview)":
                dept_n = int((df_current['Department'] == selected_dept_focus).sum())
                st.info(f"🎯 **Department Focus:** Isolating **{selected_dept_focus}** ({dept_n:,} students, {dept_n/len(df_current)*100:.1f}% of cohort).")
            else:
                st.info(f"🌐 **Cohort Overview:** Evaluating all {len(depts_found) if depts_found else 'available'} departments. Choose a specific department to isolate.")

    # Prediction Action Button
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_pred = st.button("🔮 Run FEDRA 3.0 Batch Prediction", type="primary", use_container_width=True)
        
    with col_info:
        st.caption(f"Ensemble Models: CatBoost (w=0.45), XGBoost (w=0.45), Logistic Anchor (w=0.10) | Calibrated Cutoff: **{th_fedra*100:.1f}%**")
        
    if run_pred or ('batch_predictions' in st.session_state):
        if run_pred or ('batch_predictions' not in st.session_state):
            with st.spinner("Transforming features and computing multi-branch predictions..."):
                try:
                    pred_results = predict_batch(df_current, artifacts=artifacts)
                    st.session_state['batch_predictions'] = pred_results
                except Exception as ex:
                    st.error(f"Prediction failed: {ex}")
                    st.stop()
                    
        full_res_df = st.session_state['batch_predictions']
        
        # Isolate to single department if selected
        if selected_dept_focus != "All Departments (Overview)" and 'Department' in full_res_df.columns:
            res_df = full_res_df[full_res_df['Department'] == selected_dept_focus].copy()
            dept_banner_text = f" (Isolated Department: {selected_dept_focus})"
        else:
            res_df = full_res_df.copy()
            dept_banner_text = ""
        
        st.divider()
        
        # ==============================================================================
        # 3. KPI SUMMARY CARDS
        # ==============================================================================
        total_n = len(res_df)
        at_risk_n = int(res_df['Predicted_At_Risk'].sum()) if total_n > 0 else 0
        at_risk_pct = (at_risk_n / total_n * 100) if total_n > 0 else 0.0
        critical_n = int((res_df['Intervention_Priority'] == 'Critical Priority').sum()) if total_n > 0 else 0
        critical_pct = (critical_n / total_n * 100) if total_n > 0 else 0.0
        avg_risk = float(res_df['Risk_Score_Pct'].mean()) if total_n > 0 else 0.0
        
        st.subheader(f"📊 Assessment Summary & Early-Warning KPIs{dept_banner_text}", anchor=False)
        
        with st.container(horizontal=True):
            st.metric("Total Records Evaluated", f"{total_n:,}", f"{selected_dept_focus if selected_dept_focus != 'All Departments (Overview)' else 'Full batch'}", border=True)
            st.metric(
                "Flagged At-Risk",
                f"{at_risk_n:,}",
                f"{at_risk_pct:.1f}% of cohort",
                delta_color="inverse",
                border=True
            )
            st.metric(
                "Critical Priority Alerts",
                f"{critical_n:,}",
                f"{critical_pct:.1f}% of cohort",
                delta_color="inverse",
                border=True
            )
            st.metric(
                "Cohort Mean Risk Score",
                f"{avg_risk:.1f}%",
                f"Cutoff: {th_fedra*100:.1f}%",
                delta_color="inverse",
                border=True
            )

        # ==============================================================================
        # 4. VISUAL DISTRIBUTION & BREAKDOWN CHARTS
        # ==============================================================================
        st.markdown(f"##### Risk Visualizations & Distributions{dept_banner_text}")
        col_c1, col_c2 = st.columns([1.3, 1])
        
        with col_c1:
            fig_dist = plot_batch_risk_distribution(res_df, threshold=th_fedra)
            st.plotly_chart(fig_dist, width='stretch')
            
        with col_c2:
            fig_donut = plot_batch_priority_donut(res_df)
            st.plotly_chart(fig_donut, width='stretch')

        # Department Comparative Ranking Bar Chart
        if 'Department' in full_res_df.columns and full_res_df['Department'].nunique() > 1:
            fig_dept_comp = plot_department_risk_bars(
                full_res_df,
                selected_dept=selected_dept_focus if selected_dept_focus != "All Departments (Overview)" else None,
                max_departments=12
            )
            if fig_dept_comp:
                st.plotly_chart(fig_dept_comp, width='stretch')
        elif 'Faculty' in res_df.columns and res_df['Faculty'].nunique() > 1:
            fig_fac = plot_batch_faculty_bars(res_df)
            if fig_fac:
                st.plotly_chart(fig_fac, width='stretch')

        st.divider()

        # ==============================================================================
        # 5. FILTERABLE STUDENT INTERVENTION QUEUE
        # ==============================================================================
        st.subheader(f"📋 Scored Student Intervention Queue ({total_n:,} students){dept_banner_text}", anchor=False)
        
        # Filter row
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1.5])
        with f_col1:
            prio_filter = st.multiselect(
                "Filter by Triage Priority",
                ["Critical Priority", "Moderate Risk", "On Track"],
                default=["Critical Priority", "Moderate Risk", "On Track"]
            )
        with f_col2:
            fac_options = ["All Faculties"] + sorted([str(x) for x in res_df['Faculty'].dropna().unique()]) if 'Faculty' in res_df.columns else ["All Faculties"]
            fac_filter = st.selectbox("Filter by Faculty", fac_options)
            
        with f_col3:
            search_query = st.text_input("🔍 Search StudentID, State, Major...", placeholder="Type to filter...")

        # Apply filters
        filtered_res = res_df.copy()
        if prio_filter:
            filtered_res = filtered_res[filtered_res['Intervention_Priority'].isin(prio_filter)]
        if fac_filter != "All Faculties" and 'Faculty' in filtered_res.columns:
            filtered_res = filtered_res[filtered_res['Faculty'] == fac_filter]
        if search_query:
            sq = search_query.strip().lower()
            mask = filtered_res.astype(str).apply(lambda row: sq in ' '.join(row).lower(), axis=1)
            filtered_res = filtered_res[mask]

        st.caption(f"Showing **{len(filtered_res):,}** of **{total_n:,}** student records.")

        display_queue_cols = [
            'StudentID', 'Faculty', 'Department', 'YearOfStudy', 'State', 'Gender',
            'GPA', 'PreviousSemesterGPA', 'GPA_Momentum', 'AttendanceRate', 'TuitionPaid',
            'Risk_Score_Pct', 'Intervention_Priority', 'Actionable_Recommendation'
        ]
        # Only select columns that exist in DataFrame
        queue_cols = [c for c in display_queue_cols if c in filtered_res.columns]

        st.dataframe(
            filtered_res[queue_cols],
            column_config={
                "Risk_Score_Pct": st.column_config.ProgressColumn(
                    "Dropout Risk",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "GPA": st.column_config.NumberColumn("Current GPA", format="%.2f"),
                "GPA_Momentum": st.column_config.NumberColumn("Δ Momentum", format="%+.2f"),
                "AttendanceRate": st.column_config.NumberColumn("Attendance", format="%.1f%%"),
                "YearOfStudy": st.column_config.TextColumn("Year"),
                "Intervention_Priority": st.column_config.TextColumn("Triage Tier"),
                "Actionable_Recommendation": st.column_config.TextColumn("Personalized Intervention Action")
            },
            width='stretch',
            hide_index=True
        )

        st.divider()

        # ==============================================================================
        # 6. DOWNLOAD SCORED PREDICTIONS (GLOBAL & DEPARTMENTAL)
        # ==============================================================================
        st.subheader("💾 Export Scored Predictions", anchor=False)
        st.markdown("Export enriched predictions with calibrated risk scores, probability tiers, and actionable intervention steps as CSV.")

        if selected_dept_focus != "All Departments (Overview)":
            c_exp1, c_exp2, c_exp3 = st.columns([1.2, 1, 1])
            with c_exp1:
                dept_csv = res_df.to_csv(index=False).encode('utf-8')
                dept_safe_name = selected_dept_focus.lower().replace(" ", "_")
                st.download_button(
                    label=f"🎯 Download {selected_dept_focus} Roster (CSV)",
                    data=dept_csv,
                    file_name=f"fedra3_predictions_{dept_safe_name}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            with c_exp2:
                at_risk_df = res_df[res_df['Predicted_At_Risk'] == 1]
                at_risk_csv = at_risk_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"🚨 Download At-Risk Only ({len(at_risk_df):,})",
                    data=at_risk_csv,
                    file_name=f"fedra3_predictions_{dept_safe_name}_at_risk.csv",
                    mime="text/csv",
                    disabled=(len(at_risk_df) == 0),
                    use_container_width=True
                )
            with c_exp3:
                full_csv = full_res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🌐 Download All Departments (CSV)",
                    data=full_csv,
                    file_name="fedra3_batch_predictions_all_departments.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                full_csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Complete Scored Batch (CSV)",
                    data=full_csv,
                    file_name="fedra3_batch_predictions_full.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            with c_exp2:
                at_risk_df = res_df[res_df['Predicted_At_Risk'] == 1]
                at_risk_csv = at_risk_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"🚨 Download Flagged At-Risk Only ({len(at_risk_df):,} Students)",
                    data=at_risk_csv,
                    file_name="fedra3_batch_predictions_at_risk_only.csv",
                    mime="text/csv",
                    disabled=(len(at_risk_df) == 0),
                    use_container_width=True
                )
else:
    st.info("👆 Please upload a student CSV file above, or click **⚡ Load demo sample data** to preview the batch prediction workflow.")

