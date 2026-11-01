import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

# Global sleek design palette
COLOR_PRIMARY = "#1e40af"   # Deep Indigo
COLOR_SECONDARY = "#3b82f6" # Bright Blue
COLOR_SUCCESS = "#10b981"   # Emerald Green
COLOR_WARNING = "#f59e0b"   # Amber
COLOR_DANGER = "#ef4444"    # Crimson Red
COLOR_MUTED = "#64748b"     # Slate Gray
COLOR_BG_LIGHT = "#f8fafc"  # Soft background

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

def apply_sleek_layout(fig, title=None, height=420, show_legend=True):
    """Applies a consistent, modern design aesthetic to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(family=FONT_FAMILY, size=15, color="#0f172a"),
            x=0.01,
            y=0.96
        ),
        font=dict(family=FONT_FAMILY, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=40, r=30, t=50 if title else 25, b=40),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(226,232,240,0.8)",
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=11, color="#64748b"),
            title_font=dict(size=12, color="#334155")
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=11, color="#64748b"),
            title_font=dict(size=12, color="#334155")
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=12,
            font_family=FONT_FAMILY,
            font_color="#ffffff"
        )
    )
    return fig


# ==============================================================================
# 1. EXPLORATORY DATA ANALYSIS (EDA) CHARTS
# ==============================================================================

def plot_academic_outcomes_donut(df):
    """Donut chart showing breakdown of student academic outcomes (from Notebook Cell 5)."""
    outcome_col = 'Status_clean' if 'Status_clean' in df.columns else 'Status'
    counts = df[outcome_col].value_counts().reset_index()
    counts.columns = ['Outcome', 'Count']
    
    color_map = {
        'Graduate': '#10b981',
        'Graduated': '#10b981',
        'Enrolled': '#3b82f6',
        'Dropout': '#ef4444',
        'Academic Suspension': '#f59e0b',
        'Expelled': '#dc2626'
    }
    colors = [color_map.get(k, '#8b5cf6') for k in counts['Outcome']]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=counts['Outcome'],
            values=counts['Count'],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
            textinfo='label+percent',
            hovertemplate="<b>%{label}</b><br>Students: %{value:,}<br>Proportion: %{percent}<extra></extra>"
        )
    ])
    apply_sleek_layout(fig, title="Student Academic Outcome Distribution (N=50,150)", height=380)
    fig.update_layout(
        annotations=[dict(text=f"<b>{len(df):,}</b><br><span style='font-size:11px;color:#64748b'>Total Students</span>", 
                          x=0.5, y=0.5, font_size=16, font_family=FONT_FAMILY, showarrow=False)]
    )
    return fig


def plot_academic_momentum_box(df):
    """Academic momentum (GPA Velocity) box plot by outcome/status with zero reference line."""
    outcome_col = 'Status_clean' if 'Status_clean' in df.columns else 'Status'
    df_plot = df.sample(n=min(5000, len(df)), random_state=42)
    
    color_map = {
        'Graduate': '#10b981',
        'Graduated': '#10b981',
        'Enrolled': '#3b82f6',
        'Dropout': '#ef4444',
        'Academic Suspension': '#f59e0b',
        'Expelled': '#dc2626'
    }
    
    fig = go.Figure()
    for cat in df_plot[outcome_col].unique():
        sub = df_plot[df_plot[outcome_col] == cat]
        fig.add_trace(go.Box(
            y=sub['GPA_Momentum'],
            name=str(cat),
            marker_color=color_map.get(cat, '#6366f1'),
            boxpoints=False,
            jitter=0.2
        ))
        
    fig.add_hline(y=0.0, line_dash="dash", line_color="#ef4444", line_width=1.5,
                  annotation_text="Zero Momentum Line (No Change)", annotation_position="top left",
                  annotation_font=dict(size=11, color="#ef4444"))
    
    apply_sleek_layout(fig, title="Academic Momentum (Current GPA - Prev Semester GPA) by Outcome", height=380)
    fig.update_yaxes(title="GPA Momentum (Velocity)", zeroline=False)
    fig.update_xaxes(title="Academic Outcome")
    return fig


def plot_attendance_gpa_density(df, sample_n=3000):
    """Interactive scatter plot of Attendance Rate vs Current GPA colored by outcome."""
    outcome_col = 'Status_clean' if 'Status_clean' in df.columns else 'Status'
    df_plot = df.sample(n=min(sample_n, len(df)), random_state=42).copy()
    
    color_map = {
        'Graduate': '#10b981',
        'Graduated': '#10b981',
        'Enrolled': '#3b82f6',
        'Dropout': '#ef4444',
        'Academic Suspension': '#f59e0b',
        'Expelled': '#dc2626'
    }
    
    fig = px.scatter(
        df_plot,
        x='AttendanceRate',
        y='GPA',
        color=outcome_col,
        color_discrete_map=color_map,
        opacity=0.65,
        hover_data=['StudentID', 'Faculty', 'Risk_Score_Pct'] if 'Risk_Score_Pct' in df_plot.columns else ['StudentID', 'Faculty']
    )
    
    # Quadrant threshold guides: Danger Zone (Attendance < 60% & GPA < 2.0)
    fig.add_vline(x=60, line_dash="dot", line_color="#94a3b8", line_width=1.2)
    fig.add_hline(y=2.0, line_dash="dot", line_color="#94a3b8", line_width=1.2)
    fig.add_annotation(
        x=25, y=1.0,
        text="<b>CRITICAL DANGER ZONE</b><br>Attendance < 60% & GPA < 2.0",
        showarrow=False,
        font=dict(size=11, color="#ef4444", family=FONT_FAMILY),
        bgcolor="rgba(254, 226, 226, 0.6)",
        bordercolor="#fca5a5",
        borderwidth=1,
        borderpad=4
    )
    
    apply_sleek_layout(fig, title=f"Attendance Rate (%) vs. Current GPA Density (N={len(df_plot):,})", height=430)
    fig.update_xaxes(title="Lecture Attendance Rate (%)", range=[5, 105])
    fig.update_yaxes(title="Current Cumulative GPA (out of 5.0)", range=[0.0, 5.2])
    return fig


def plot_attrition_by_funding(df):
    """Attrition rate (%) across funding sources (from Notebook Cell 5)."""
    target_col = 'Actual_Attrition' if 'Actual_Attrition' in df.columns else 'Dropout'
    if target_col not in df.columns:
        df['temp_attr'] = (df['Status_clean'].str.lower().isin(['dropout', 'expelled', 'academic suspension'])).astype(int)
        target_col = 'temp_attr'
        
    rates = df.groupby('FundingSource')[target_col].mean().sort_values(ascending=True) * 100
    
    fig = go.Figure(go.Bar(
        x=rates.values,
        y=rates.index,
        orientation='h',
        marker=dict(
            color=rates.values,
            colorscale='Blues',
            line=dict(color='#1e40af', width=1)
        ),
        text=[f"{v:.1f}%" for v in rates.values],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Attrition Rate: %{x:.2f}%<extra></extra>"
    ))
    
    apply_sleek_layout(fig, title="Empirical Attrition Rate (%) across Funding Sources", height=350, show_legend=False)
    fig.update_xaxes(title="Student Attrition / Dropout Rate (%)", range=[0, max(rates.values) * 1.25])
    fig.update_yaxes(title="")
    return fig


def plot_faculty_risk_breakdown(df):
    """Faculty attrition risk ranking bar chart."""
    risk_col = 'Risk_Score_Pct' if 'Risk_Score_Pct' in df.columns else None
    if risk_col:
        fac = df.groupby('Faculty')[risk_col].mean().sort_values(ascending=True)
        title_text = "Mean Predicted Attrition Risk (%) by University Faculty"
        x_title = "Average Risk Score (%)"
    else:
        fac = df.groupby('Faculty')['Actual_Attrition'].mean().sort_values(ascending=True) * 100
        title_text = "Empirical Attrition Rate (%) by Faculty"
        x_title = "Attrition Rate (%)"
        
    colors = ['#ef4444' if v >= 50 else ('#f59e0b' if v >= 35 else '#3b82f6') for v in fac.values]
    
    fig = go.Figure(go.Bar(
        x=fac.values,
        y=fac.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#ffffff', width=1)),
        text=[f"{v:.1f}%" for v in fac.values],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.1f}%<extra></extra>"
    ))
    
    apply_sleek_layout(fig, title=title_text, height=440, show_legend=False)
    fig.update_xaxes(title=x_title, range=[0, max(fac.values) * 1.25])
    return fig


def plot_state_risk_bars(df):
    """Attrition rate across Nigerian states of origin."""
    target_col = 'Actual_Attrition' if 'Actual_Attrition' in df.columns else 'Predicted_At_Risk'
    state_rates = df.groupby('State')[target_col].mean().sort_values(ascending=True) * 100
    
    fig = go.Figure(go.Bar(
        x=state_rates.index,
        y=state_rates.values,
        marker=dict(
            color=state_rates.values,
            colorscale='YlOrRd',
            line=dict(color='#b91c1c', width=0.5)
        ),
        text=[f"{v:.1f}%" for v in state_rates.values],
        textposition='outside',
        hovertemplate="<b>State: %{x}</b><br>Attrition Rate: %{y:.1f}%<extra></extra>"
    ))
    
    apply_sleek_layout(fig, title="Geographic Disparity: Attrition Rate by State of Origin", height=380, show_legend=False)
    fig.update_xaxes(title="State of Origin", tickangle=-45)
    fig.update_yaxes(title="Attrition Rate (%)", range=[0, max(state_rates.values) * 1.25])
    return fig


def plot_tuition_risk_impact(df):
    """Impact of Tuition Payment Status on Risk Breakdown."""
    grouped = df.groupby(['TuitionPaid', 'Intervention_Priority']).size().reset_index(name='Count')
    
    color_map = {
        'Critical Priority': '#ef4444',
        'Moderate Risk': '#f59e0b',
        'On Track': '#10b981'
    }
    
    fig = px.bar(
        grouped,
        x='TuitionPaid',
        y='Count',
        color='Intervention_Priority',
        barmode='group',
        color_discrete_map=color_map,
        text='Count'
    )
    fig.update_traces(textposition='outside')
    apply_sleek_layout(fig, title="Tuition Payment Status vs. Intervention Priority Cohorts", height=360)
    fig.update_xaxes(title="Tuition Payment Status")
    fig.update_yaxes(title="Student Population")
    return fig


def plot_cohort_risk_histogram(df, threshold=0.35):
    """Histogram of FEDRA 3.0 predicted risk scores across entire cohort."""
    scores = df['Risk_Score_Pct']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=40,
        marker=dict(
            color='#3b82f6',
            line=dict(color='#1e40af', width=1)
        ),
        hovertemplate="Risk Range: %{x}%<br>Students: %{y:,}<extra></extra>"
    ))
    
    th_pct = threshold * 100
    fig.add_vline(x=th_pct, line_dash="dash", line_color="#ef4444", line_width=2.5,
                  annotation_text=f"Decision Cutoff ({th_pct:.1f}%)", annotation_position="top left",
                  annotation_font=dict(size=11, color="#ef4444"))
    
    fig.add_vrect(x0=0, x1=th_pct, fillcolor="#10b981", opacity=0.08, line_width=0,
                  annotation_text="On Track Zone", annotation_position="top left",
                  annotation_font=dict(size=10, color="#059669"))
    fig.add_vrect(x0=th_pct, x1=100, fillcolor="#ef4444", opacity=0.08, line_width=0,
                  annotation_text="At-Risk Priority Zone", annotation_position="top right",
                  annotation_font=dict(size=10, color="#dc2626"))
                  
    apply_sleek_layout(fig, title="Global Cohort Attrition Risk Distribution (N=50,150)", height=380, show_legend=False)
    fig.update_xaxes(title="FEDRA 3.0 Predicted Risk Probability (%)", range=[0, 100])
    fig.update_yaxes(title="Number of Students")
    return fig


# ==============================================================================
# 2. MODEL EVALUATION & BENCHMARKS (From Notebook Cell 10)
# ==============================================================================

def plot_roc_curves(eval_dict, y_test):
    """Multi-model ROC curves with AUC values in legend (Cell 10)."""
    fig = go.Figure()
    
    palette = {
        'Logistic Regression': '#94a3b8',
        'Random Forest': '#f59e0b',
        'XGBoost': '#06b6d4',
        'CatBoost': '#8b5cf6',
        'FEDRA 3.0': '#1e40af'
    }
    
    for name, res in eval_dict.items():
        p_test = res.get('p_test')
        if p_test is not None:
            fpr, tpr, _ = roc_curve(y_test, p_test)
            auc_val = res.get('auc', 0.5)
            is_fedra = (name == 'FEDRA 3.0')
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f"{name} (AUC = {auc_val:.4f})",
                line=dict(
                    color=palette.get(name, '#64748b'),
                    width=3 if is_fedra else 1.8,
                    dash='solid' if is_fedra else 'dot'
                ),
                hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
            ))
            
    # Chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Chance (AUC = 0.500)',
        line=dict(color='#cbd5e1', width=1.5, dash='dash'),
        hoverinfo='skip'
    ))
    
    apply_sleek_layout(fig, title="Receiver Operating Characteristic (ROC) Comparison", height=420)
    fig.update_xaxes(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02])
    fig.update_yaxes(title="True Positive Rate (Sensitivity / Recall)", range=[-0.02, 1.02])
    return fig


def plot_pr_curves(eval_dict, y_test):
    """Precision-Recall Curves for early at-risk detection (Cell 10)."""
    fig = go.Figure()
    
    palette = {
        'Logistic Regression': '#94a3b8',
        'Random Forest': '#f59e0b',
        'XGBoost': '#06b6d4',
        'CatBoost': '#8b5cf6',
        'FEDRA 3.0': '#1e40af'
    }
    
    for name, res in eval_dict.items():
        p_test = res.get('p_test')
        if p_test is not None:
            prec, rec, _ = precision_recall_curve(y_test, p_test)
            ap = average_precision_score(y_test, p_test)
            is_fedra = (name == 'FEDRA 3.0')
            fig.add_trace(go.Scatter(
                x=rec,
                y=prec,
                mode='lines',
                name=f"{name} (PR-AUC = {ap:.4f})",
                line=dict(
                    color=palette.get(name, '#64748b'),
                    width=3 if is_fedra else 1.8
                ),
                hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
            ))
            
    base_rate = float(np.mean(y_test))
    fig.add_hline(y=base_rate, line_dash="dash", line_color="#cbd5e1",
                  annotation_text=f"Baseline Prevalence ({base_rate:.1%})", annotation_position="bottom right")
                  
    apply_sleek_layout(fig, title="Precision-Recall Curves (Early Attrition Detection)", height=420)
    fig.update_xaxes(title="Recall (Sensitivity)", range=[-0.02, 1.02])
    fig.update_yaxes(title="Precision (Positive Predictive Value)", range=[-0.02, 1.02])
    return fig


def plot_calibration_curves(eval_dict, y_test):
    """Probability Reliability / Calibration curves with 10 bins (Cell 10)."""
    fig = go.Figure()
    
    palette = {
        'Random Forest': '#f59e0b',
        'CatBoost': '#8b5cf6',
        'FEDRA 3.0': '#1e40af'
    }
    
    for name in ['Random Forest', 'CatBoost', 'FEDRA 3.0']:
        if name in eval_dict and eval_dict[name].get('p_test') is not None:
            prob_true, prob_pred = calibration_curve(y_test, eval_dict[name]['p_test'], n_bins=10)
            brier = eval_dict[name].get('brier', 0.0)
            fig.add_trace(go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode='lines+markers',
                name=f"{name} (Brier = {brier:.4f})",
                marker=dict(size=6),
                line=dict(color=palette.get(name, '#3b82f6'), width=2),
                hovertemplate=f"<b>{name}</b><br>Pred: %{{x:.3f}}<br>Empirical: %{{y:.3f}}<extra></extra>"
            ))
            
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='#cbd5e1', width=1.5, dash='dash'),
        hoverinfo='skip'
    ))
    
    apply_sleek_layout(fig, title="Probability Calibration & Reliability Curves", height=420)
    fig.update_xaxes(title="Mean Predicted Dropout Probability", range=[-0.02, 1.02])
    fig.update_yaxes(title="Empirical Attrition Rate", range=[-0.02, 1.02])
    return fig


def plot_confusion_matrix_heatmap(cm, threshold=0.35):
    """Interactive Confusion Matrix Heatmap (Cell 10)."""
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    z = [[tn, fp], [fn, tp]]
    text = [
        [f"<b>TN</b>: {tn:,}<br>({tn/total:.1%})", f"<b>FP (Alarm)</b>: {fp:,}<br>({fp/total:.1%})"],
        [f"<b>FN (Miss)</b>: {fn:,}<br>({fn/total:.1%})", f"<b>TP (Caught)</b>: {tp:,}<br>({tp/total:.1%})"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['Predicted Retained (0)', 'Predicted At-Risk (1)'],
        y=['Actual Retained (0)', 'Actual Attrition (1)'],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "family": FONT_FAMILY},
        colorscale='Blues',
        showscale=False
    ))
    
    apply_sleek_layout(fig, title=f"Confusion Matrix (Threshold = {threshold:.2f})", height=350, show_legend=False)
    fig.update_xaxes(side="bottom")
    return fig


def plot_model_comparison_bars(eval_dict):
    """Bar chart comparing Accuracy, Precision, Recall, F1, and AUC across models."""
    models = list(eval_dict.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    data = []
    for m in models:
        res = eval_dict[m]
        data.append({
            'Model': m,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1'],
            'ROC-AUC': res['auc']
        })
        
    df_metrics = pd.DataFrame(data).melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    fig = px.bar(
        df_metrics,
        x='Metric',
        y='Score',
        color='Model',
        barmode='group',
        text='Score',
        color_discrete_sequence=['#94a3b8', '#f59e0b', '#06b6d4', '#8b5cf6', '#1e40af']
    )
    fig.update_traces(texttemplate='%{y:.3f}', textposition='outside', textfont_size=10)
    apply_sleek_layout(fig, title="Benchmark Model Comparison Across Classification Metrics", height=380)
    fig.update_yaxes(range=[0, 1.15], title="Score (0.0 to 1.0)")
    return fig


# ==============================================================================
# 3. STUDENT SIMULATOR & EXPLAINABILITY (From Notebook Cell 11)
# ==============================================================================

def plot_student_risk_gauge(risk_pct, threshold_pct=35.0):
    """Sleek semi-circular gauge dial showing student's risk and decision threshold."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        number={'suffix': "%", 'font': {'size': 36, 'family': FONT_FAMILY, 'color': '#0f172a'}},
        delta={'reference': threshold_pct, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}, 'suffix': "% vs Threshold"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1", 'tickfont': {'size': 11}},
            'bar': {'color': "#1e293b", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, threshold_pct], 'color': "rgba(16, 185, 129, 0.25)"},
                {'range': [threshold_pct, 60.0], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [60.0, 100.0], 'color': "rgba(239, 68, 68, 0.25)"}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': threshold_pct
            }
        }
    ))
    
    apply_sleek_layout(fig, title="FEDRA 3.0 Individual Risk Meter", height=280, show_legend=False)
    fig.update_layout(margin=dict(l=30, r=30, t=40, b=20))
    return fig


def plot_student_shap_waterfall(df_shap, base_val=0.0):
    """Horizontal tornado / attribution chart of student risk factors (from Notebook Cell 11)."""
    df_sorted = df_shap.sort_values(by='shap_value', ascending=True).copy()
    
    colors = ['#ef4444' if v > 0 else '#10b981' for v in df_sorted['shap_value']]
    
    fig = go.Figure(go.Bar(
        x=df_sorted['shap_value'],
        y=df_sorted['label'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='#ffffff', width=1)),
        text=[f"{v:+.2f}" for v in df_sorted['shap_value']],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:+.3f} log-odds<br>Status: %{customdata}<extra></extra>",
        customdata=df_sorted['direction']
    ))
    
    fig.add_vline(x=0.0, line_dash="solid", line_color="#94a3b8", line_width=1.5)
    
    apply_sleek_layout(fig, title="Top Student Risk Drivers (SHAP Local Explainability)", height=380, show_legend=False)
    fig.update_xaxes(title="SHAP Attribution (Log-Odds Impact: Positive Increases Risk, Negative Protects)")
    fig.update_yaxes(title="")
    return fig


def plot_what_if_comparison(current_res, cf_res):
    """Side-by-side comparison of pre-intervention vs post-intervention risk."""
    categories = ['Ensemble (FEDRA 3.0)', 'CatBoost Branch', 'XGBoost Branch', 'Logistic Regression']
    current_vals = [
        current_res['risk_score_pct'],
        current_res['breakdown']['CatBoost'],
        current_res['breakdown']['XGBoost'],
        current_res['breakdown']['Logistic Regression']
    ]
    interv_vals = [
        cf_res['risk_score_pct'],
        cf_res['breakdown']['CatBoost'],
        cf_res['breakdown']['XGBoost'],
        cf_res['breakdown']['Logistic Regression']
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Current Risk Profile', x=categories, y=current_vals, marker_color='#ef4444',
               text=[f"{v:.1f}%" for v in current_vals], textposition='outside'),
        go.Bar(name='Post-Intervention Profile', x=categories, y=interv_vals, marker_color='#10b981',
               text=[f"{v:.1f}%" for v in interv_vals], textposition='outside')
    ])
    
    fig.update_layout(barmode='group')
    apply_sleek_layout(fig, title="What-If Intervention Risk Reduction Across Models", height=340)
    fig.update_yaxes(title="Predicted Dropout Probability (%)", range=[0, 115])
    return fig


# ==============================================================================
# 4. FAIRNESS AUDIT & BIAS MITIGATION (From Notebook Cell 12)
# ==============================================================================

def plot_fairness_disparity_bars(sr_a, sr_b, tpr_a, tpr_b, lbl_a, lbl_b):
    """Grouped bar chart showing selection rate and TPR parity across demographic subgroups."""
    fig = go.Figure(data=[
        go.Bar(
            name='Positive Selection Rate (At-Risk %)',
            x=[lbl_a, lbl_b],
            y=[sr_a * 100, sr_b * 100],
            marker_color='#3b82f6',
            text=[f"{sr_a*100:.1f}%", f"{sr_b*100:.1f}%"],
            textposition='outside'
        ),
        go.Bar(
            name='True Positive Rate (Sensitivity %)',
            x=[lbl_a, lbl_b],
            y=[tpr_a * 100, tpr_b * 100],
            marker_color='#8b5cf6',
            text=[f"{tpr_a*100:.1f}%", f"{tpr_b*100:.1f}%"],
            textposition='outside'
        )
    ])
    
    fig.update_layout(barmode='group')
    apply_sleek_layout(fig, title=f"Algorithmic Disparity: {lbl_a} vs. {lbl_b}", height=340)
    fig.update_yaxes(title="Rate (%)", range=[0, 115])
    return fig


def plot_mitigation_impact(dpd_base, eod_base, f1_base, dpd_mit, eod_mit, f1_mit):
    """Before vs After mitigation comparison of disparity and performance."""
    metrics = ['Demographic Parity Disparity (DPD)', 'Equal Opportunity Disparity (EOD)', 'Model F1-Score']
    before = [dpd_base, eod_base, f1_base]
    after = [dpd_mit, eod_mit, f1_mit]
    
    fig = go.Figure(data=[
        go.Bar(name='Before Mitigation (Base)', x=metrics, y=before, marker_color='#94a3b8',
               text=[f"{v:.4f}" for v in before], textposition='outside'),
        go.Bar(name='Post-Mitigation (Pareto-Calibrated)', x=metrics, y=after, marker_color='#10b981',
               text=[f"{v:.4f}" for v in after], textposition='outside')
    ])
    
    fig.update_layout(barmode='group')
    apply_sleek_layout(fig, title="Fairness & Utility Trade-Off: Pre vs. Post Bias Mitigation", height=340)
    fig.update_yaxes(title="Metric Value", range=[0, max(max(before), max(after)) * 1.3])
    return fig
