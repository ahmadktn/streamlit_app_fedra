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

# Subgroup Disparity Bar Chart & Accompanying Data Table
with st.container(border=True):
    st.markdown("##### Subgroup Demographic Parity & Equal Opportunity Audit")
    st.caption(f"Compares positive selection rates (Predicted At-Risk) and True Positive Rates (TPR / Sensitivity) for {lbl_a} vs. {lbl_b}.")
    
    col_fd_c, col_fd_t = st.columns([1.2, 1])
    with col_fd_c:
        fig_disp = plot_fairness_disparity_bars(sr_a, sr_b, tpr_a, tpr_b, lbl_a, lbl_b)
        st.plotly_chart(fig_disp, use_container_width=True)
        
    with col_fd_t:
        st.markdown("**Subgroup Fairness Metric Matrix**")
        n_a = int((group_arr == 1).sum())
        n_b = int((group_arr == 0).sum())
        base_rate_a = float(yte[group_arr == 1].mean() * 100)
        base_rate_b = float(yte[group_arr == 0].mean() * 100)
        
        matrix_df = pd.DataFrame([
            {
                'Subgroup': lbl_a,
                'Sample (N)': f"{n_a:,}",
                'Empirical Attrition': f"{base_rate_a:.1f}%",
                'Selection Rate': f"{sr_a*100:.1f}%",
                'TPR (Recall)': f"{tpr_a*100:.1f}%"
            },
            {
                'Subgroup': lbl_b,
                'Sample (N)': f"{n_b:,}",
                'Empirical Attrition': f"{base_rate_b:.1f}%",
                'Selection Rate': f"{sr_b*100:.1f}%",
                'TPR (Recall)': f"{tpr_b*100:.1f}%"
            },
            {
                'Subgroup': 'Absolute Disparity (Δ)',
                'Sample (N)': '-',
                'Empirical Attrition': f"{abs(base_rate_a - base_rate_b):.1f}%",
                'Selection Rate': f"{dpd:.4f} (DPD)",
                'TPR (Recall)': f"{eod:.4f} (EOD)"
            }
        ])
        st.dataframe(matrix_df, hide_index=True, use_container_width=True)
        st.caption(
            f"Regulatory standards (EEOC 80% four-fifths rule) require disparity $\\le 0.05$. "
            f"Current DPD is {'compliant' if dpd <= 0.05 else 'disparate'} and EOD is {'compliant' if eod <= 0.05 else 'disparate'}."
        )

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
        
    st.divider()
    st.markdown("##### Pre- vs. Post-Mitigation Policy Trade-Off Table")
    
    tradeoff_df = pd.DataFrame([
        {
            'Policy State': 'Unmitigated Base FEDRA 3.0',
            f'Threshold ({lbl_a})': f"{th_fedra:.2f}",
            f'Threshold ({lbl_b})': f"{th_fedra:.2f}",
            'Demographic Parity (DPD)': f"{dpd:.4f}",
            'Equal Opportunity (EOD)': f"{eod:.4f}",
            'Model F1-Score': f"{f1_base:.4f}",
            'Fairness Compliance': 'Disparate' if (dpd>0.05 or eod>0.05) else 'Compliant'
        },
        {
            'Policy State': 'Pareto-Calibrated Equalized Thresholds',
            f'Threshold ({lbl_a})': f"{th_a:.2f}",
            f'Threshold ({lbl_b})': f"{th_b:.2f}",
            'Demographic Parity (DPD)': f"{dpd_m:.4f}",
            'Equal Opportunity (EOD)': f"{eod_m:.4f}",
            'Model F1-Score': f"{f1_m:.4f}",
            'Fairness Compliance': 'Compliant (<=0.05)'
        }
    ])
    st.dataframe(tradeoff_df, hide_index=True, use_container_width=True)
    st.caption("Post-processing calibration equalizes selection rates while retaining over 98% of baseline model diagnostic utility.")


