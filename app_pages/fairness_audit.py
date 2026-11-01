import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fedra_app.pipeline import train_suite_models
from fedra_app.fairness import compute_fairness_metrics, calibrate_demographic_parity
from fedra_app.visualizations import plot_fairness_disparity_bars, plot_mitigation_impact

st.header("Fairness audit & bias mitigation studio", anchor=False)
st.caption("Audit algorithmic disparities across sensitive attributes and test Pareto-calibrated post-processing mitigations.")

artifacts = train_suite_models()
eval_dict = artifacts['eval']
test_data = artifacts['data']['test_data']
yte = test_data['y_test']

# Protected attributes in test set
sens_gender = test_data['sens_gender_test']
sens_funding = test_data['sens_funding_test']
sens_firstgen = test_data['sens_firstgen_test']

attr_choice = st.segmented_control(
    "Select protected demographic dimension",
    ["Gender (Male vs. Female)", "Funding (Self-Funded vs. Sponsored)", "Parental Ed (First-Gen vs. Non-Gen)"],
    default="Gender (Male vs. Female)"
)

if attr_choice == "Gender (Male vs. Female)":
    group_arr = sens_gender
    lbl_a, lbl_b = 'Male', 'Female'
elif attr_choice == "Funding (Self-Funded vs. Sponsored)":
    group_arr = sens_funding
    lbl_a, lbl_b = 'Self-Funded', 'Sponsored'
else:
    group_arr = sens_firstgen
    lbl_a, lbl_b = 'First-Gen', 'Non-First-Gen'

# Evaluate FEDRA 3.0 base disparity
p_test_fedra = eval_dict['FEDRA 3.0']['p_test']
th_fedra     = eval_dict['FEDRA 3.0']['threshold']
base_preds   = (p_test_fedra >= th_fedra).astype(int)

dpd, eod, sr_a, sr_b, tpr_a, tpr_b = compute_fairness_metrics(yte, base_preds, group_arr)

with st.container(horizontal=True):
    dpd_status = "Pass (<= 0.05)" if dpd <= 0.05 else "Disparate (> 0.05)"
    eod_status = "Pass (<= 0.05)" if eod <= 0.05 else "Disparate (> 0.05)"
    
    st.metric(
        "Demographic parity disparity (DPD)",
        f"{dpd:.4f}",
        dpd_status,
        delta_color="normal" if dpd <= 0.05 else "inverse",
        border=True
    )
    st.metric(
        "Equal opportunity disparity (EOD)",
        f"{eod:.4f}",
        eod_status,
        delta_color="normal" if eod <= 0.05 else "inverse",
        border=True
    )
    st.metric(f"{lbl_a} positive rate", f"{sr_a:.1%}", f"TPR: {tpr_a:.1%}", border=True)
    st.metric(f"{lbl_b} positive rate", f"{sr_b:.1%}", f"TPR: {tpr_b:.1%}", border=True)

# Subgroup Disparity Bar Chart
with st.container(border=True):
    st.markdown("##### Subgroup Demographic Parity & Equal Opportunity Comparison")
    st.caption(f"Compares positive selection rates (Predicted At-Risk) and True Positive Rates (TPR / Sensitivity) for {lbl_a} vs. {lbl_b}.")
    fig_disp = plot_fairness_disparity_bars(sr_a, sr_b, tpr_a, tpr_b, lbl_a, lbl_b)
    st.plotly_chart(fig_disp, use_container_width=True)

st.divider()

# Post-processing mitigation
with st.container(border=True):
    st.subheader("Post-Processing Equalized Selection Mitigation", anchor=False)
    st.caption("Calibrates group-specific decision thresholds to eliminate Demographic Parity and Equal Opportunity bias.")
    
    col_ctrl, col_after = st.columns([1, 1.3])
    with col_ctrl:
        target_rate = st.slider("Target positive selection rate across groups", 0.40, 0.80, 0.63, step=0.01)
        mitig_preds, th_a, th_b = calibrate_demographic_parity(p_test_fedra, group_arr, target_acceptance_rate=target_rate)
        
        dpd_m, eod_m, sr_a_m, sr_b_m, tpr_a_m, tpr_b_m = compute_fairness_metrics(yte, mitig_preds, group_arr)
        f1_base = eval_dict['FEDRA 3.0']['f1']
        f1_m = f1_score(yte, mitig_preds)

        st.metric("Mitigated DPD", f"{dpd_m:.4f}", f"{(dpd_m - dpd):+.4f} change", delta_color="inverse", border=True)
        st.metric("Mitigated EOD", f"{eod_m:.4f}", f"{(eod_m - eod):+.4f} change", delta_color="inverse", border=True)
        st.metric("Post-Mitigation F1-Score", f"{f1_m:.4f}", f"{(f1_m - f1_base):+.4f} delta", border=True)

    with col_after:
        fig_mit = plot_mitigation_impact(dpd, eod, f1_base, dpd_m, eod_m, f1_m)
        st.plotly_chart(fig_mit, use_container_width=True)

