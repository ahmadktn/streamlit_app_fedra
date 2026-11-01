import numpy as np
import pandas as pd

def compute_fairness_metrics(y_true, y_pred, group_arr, val_a=1, val_b=0):
    """
    Computes Demographic Parity Difference (DPD) and Equal Opportunity Difference (EOD).
    
    Parameters:
        y_true: True binary targets (0 = On-Track, 1 = At-Risk / Attrition).
        y_pred: Predicted binary labels (0 or 1).
        group_arr: Binary sensitive attribute array (e.g. 1 = Male, 0 = Female or 1 = Self-Funded, 0 = Sponsored).
        val_a: Label value for reference group.
        val_b: Label value for comparison group.
        
    Returns:
        tuple (dpd, eod, sr_a, sr_b, tpr_a, tpr_b)
    """
    mask_a = (group_arr == val_a)
    mask_b = (group_arr == val_b)

    sr_a = float(y_pred[mask_a].mean()) if mask_a.sum() > 0 else 0.0
    sr_b = float(y_pred[mask_b].mean()) if mask_b.sum() > 0 else 0.0
    dpd = abs(sr_a - sr_b)

    pos_a = mask_a & (y_true == 1)
    pos_b = mask_b & (y_true == 1)
    tpr_a = float(y_pred[pos_a].mean()) if pos_a.sum() > 0 else 0.0
    tpr_b = float(y_pred[pos_b].mean()) if pos_b.sum() > 0 else 0.0
    eod = abs(tpr_a - tpr_b)

    return dpd, eod, sr_a, sr_b, tpr_a, tpr_b

def calibrate_demographic_parity(probas, group_arr, target_acceptance_rate=0.60):
    """
    Post-processing Equalized Selection: calibrates group-specific decision cutoffs 
    such that each group receives equal selection rates, eliminating Demographic Parity bias.
    """
    th_a = float(np.percentile(probas[group_arr == 1], 100 * (1 - target_acceptance_rate)))
    th_b = float(np.percentile(probas[group_arr == 0], 100 * (1 - target_acceptance_rate)))

    preds = np.zeros(len(probas), dtype=int)
    preds[group_arr == 1] = (probas[group_arr == 1] >= th_a).astype(int)
    preds[group_arr == 0] = (probas[group_arr == 0] >= th_b).astype(int)

    return preds, th_a, th_b
