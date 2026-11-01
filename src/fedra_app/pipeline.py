import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

CACHE_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), "../../../shared_cache/fedra3_nigerian_cache.joblib"),
    os.path.join(os.path.dirname(__file__), "../../shared_cache/fedra3_nigerian_cache.joblib"),
    os.path.join(os.path.dirname(__file__), "../shared_cache/fedra3_nigerian_cache.joblib"),
    "apps/shared_cache/fedra3_nigerian_cache.joblib",
    "shared_cache/fedra3_nigerian_cache.joblib",
    "/home/ahmad/Downloads/fedra datasets/apps/shared_cache/fedra3_nigerian_cache.joblib"
]

DATA_CANDIDATE_PATHS = [
    "../../nigerian_student_dropout_dataset.csv",
    "../nigerian_student_dropout_dataset.csv",
    "nigerian_student_dropout_dataset.csv",
    os.path.join(os.path.dirname(__file__), "../../../../nigerian_student_dropout_dataset.csv"),
    "/home/ahmad/Downloads/fedra datasets/nigerian_student_dropout_dataset.csv"
]

FACULTIES = [
    'Faculty of Science', 'Faculty of Engineering', 'Faculty of Education',
    'Faculty of Social Sciences', 'Faculty of Management Sciences', 'Faculty of Agriculture',
    'Faculty of Arts', 'Faculty of Environmental Sciences', 'Faculty of Computing and IT',
    'Faculty of Medicine', 'Faculty of Law', 'Faculty of Pharmacy'
]

STATES = [
    'Plateau', 'Bauchi', 'Anambra', 'Oyo', 'Imo', 'Lagos', 'Adamawa', 'Delta',
    'Abuja', 'Kaduna', 'Kano', 'Enugu', 'Rivers', 'Edo', 'Benue', 'Kwara',
    'Ogun', 'Taraba', 'Borno', 'Sokoto'
]

def load_cached_bundle():
    for p in CACHE_CANDIDATE_PATHS:
        if os.path.exists(p):
            return joblib.load(p)
    return None

def train_suite_models():
    """Returns the FEDRA 3.0 model suite and cohort data (instant from cache)."""
    bundle = load_cached_bundle()
    if bundle is not None:
        if 'cohort_df' in bundle:
            cohort_df = bundle['cohort_df']
        else:
            csv_path = bundle.get('cohort_csv', 'apps/shared_cache/fedra3_cohort.csv')
            if not os.path.exists(csv_path):
                # Search relative paths
                for alt in [
                    os.path.join(os.path.dirname(__file__), "../../../shared_cache/fedra3_cohort.csv"),
                    os.path.join(os.path.dirname(__file__), "../../shared_cache/fedra3_cohort.csv"),
                    os.path.join(os.path.dirname(__file__), "../shared_cache/fedra3_cohort.csv"),
                    "/home/ahmad/Downloads/fedra datasets/apps/shared_cache/fedra3_cohort.csv"
                ]:
                    if os.path.exists(alt):
                        csv_path = alt
                        break
            cohort_df = pd.read_csv(csv_path)
        test_data = bundle['test_data']
        return {
            'models': bundle['models'],
            'eval': bundle['eval'],
            'meta_weights': bundle['meta_weights'],
            'full_probas': bundle['full_probas'],
            'data': {
                'df': cohort_df,
                'imputer': bundle['imputer'],
                'scaler': bundle['scaler'],
                'feature_cols': bundle['feature_cols'],
                'target_enc_maps': bundle['target_enc_maps'],
                'freq_enc_maps': bundle['freq_enc_maps'],
                'global_y_mean': bundle['global_y_mean'],
                'test_data': test_data
            }
        }
    raise FileNotFoundError("FEDRA 3.0 cache not found. Please verify apps/shared_cache/fedra3_nigerian_cache.joblib exists.")

def predict_single_student(student_dict, artifacts=None):
    """
    Transforms a single raw student dictionary into the engineered feature vector
    and produces FEDRA 3.0 probability and risk tier.
    """
    if artifacts is None:
        artifacts = train_suite_models()
    
    data = artifacts['data']
    models = artifacts['models']
    meta_weights = artifacts['meta_weights']
    th_fedra = artifacts['eval']['FEDRA 3.0']['threshold']
    imputer = data['imputer']
    scaler = data['scaler']
    feature_cols = data['feature_cols']
    target_enc_maps = data['target_enc_maps']
    freq_enc_maps = data['freq_enc_maps']
    global_y_mean = data['global_y_mean']

    gpa = float(student_dict.get('GPA', 3.0))
    prev_gpa = float(student_dict.get('PreviousSemesterGPA', 3.0))
    attendance = float(student_dict.get('AttendanceRate', 70.0))
    study_hrs = float(student_dict.get('StudyHoursPerWeek', 15.0))
    failed = float(student_dict.get('FailedCourses', 0.0))
    credit_load = float(student_dict.get('CoursesCreditLoad', 18.0))
    allowance = float(student_dict.get('MonthlyAllowance', 40000.0))
    dependants = float(student_dict.get('Dependants', 1.0))
    distance = float(student_dict.get('DistanceFromHome', 150.0))
    mentoring = float(student_dict.get('MentoringSessionsAttended', 2.0))
    counselling = float(student_dict.get('CounsellingVisits', 0.0))
    tuition_paid = 1.0 if str(student_dict.get('TuitionPaid', 'Paid')).lower() in ['1', 'paid', 'yes'] else 0.0
    funding = str(student_dict.get('FundingSource', 'Self Funded'))
    housing = str(student_dict.get('Housing', 'Off Campus'))
    parent_ed = str(student_dict.get('ParentalEducationLevel', 'Secondary'))
    faculty = str(student_dict.get('Faculty', 'Faculty of Science'))
    dept = str(student_dict.get('Department', 'Computer Science'))
    state = str(student_dict.get('State', 'Lagos'))
    gender = str(student_dict.get('Gender', 'Male'))
    entry_mode = str(student_dict.get('EntryMode', 'UTME'))

    # Feature Engineering
    gpa_momentum = gpa - prev_gpa
    failure_load_ratio = failed / (credit_load + 1e-5)
    academic_standing_index = (gpa * (attendance / 100.0)) / (failed + 1.0)
    allowance_per_dependant = allowance / (dependants + 1.0)
    financial_precarity = 1.0 if (tuition_paid == 0.0 and funding == 'Self Funded') else 0.0
    commute_strain = distance * (1.0 if housing == 'Off Campus' else 0.0)
    study_effort_score = (attendance / 100.0) * study_hrs
    support_utilization_ratio = (mentoring + counselling) / (failed + 1.0)
    first_gen_hardship = 1.0 if (parent_ed == 'Primary' and funding == 'Self Funded') else 0.0

    raw_row = {
        'YearOfStudy': float(student_dict.get('YearOfStudy', 2)),
        'Dependants': dependants,
        'PreviousSemesterGPA': prev_gpa,
        'CoursesCreditLoad': credit_load,
        'FailedCourses': failed,
        'TuitionPaid': tuition_paid,
        'PartTimeJob': float(student_dict.get('PartTimeJob', 0)),
        'LibraryVisitsPerMonth': float(student_dict.get('LibraryVisitsPerMonth', 4)),
        'ExtracurricularActivities': float(student_dict.get('ExtracurricularActivities', 1)),
        'MentoringSessionsAttended': mentoring,
        'CounsellingVisits': counselling,
        'Age_num': float(student_dict.get('Age', 22)),
        'GPA_num': gpa,
        'AttendanceRate_num': attendance,
        'StudyHoursPerWeek_num': study_hrs,
        'MonthlyAllowance_num': allowance,
        'DistanceFromHome_num': distance,
        'gpa_momentum': gpa_momentum,
        'failure_load_ratio': failure_load_ratio,
        'academic_standing_index': academic_standing_index,
        'allowance_per_dependant': allowance_per_dependant,
        'financial_precarity': financial_precarity,
        'commute_strain': commute_strain,
        'study_effort_score': study_effort_score,
        'support_utilization_ratio': support_utilization_ratio,
        'first_gen_hardship': first_gen_hardship,
        'Gender_clean_Male': 1.0 if gender.lower() == 'male' else 0.0,
        'EntryMode_clean_Direct Entry': 1.0 if entry_mode == 'Direct Entry' else 0.0,
        'EntryMode_clean_Remedial': 1.0 if entry_mode == 'Remedial' else 0.0,
        'EntryMode_clean_Scholarship': 1.0 if entry_mode == 'Scholarship' else 0.0,
        'Housing_clean_Off Campus': 1.0 if housing == 'Off Campus' else 0.0,
        'Housing_clean_On Campus': 1.0 if housing == 'On Campus' else 0.0,
        'FundingSource_clean_Full Scholarship': 1.0 if funding == 'Full Scholarship' else 0.0,
        'FundingSource_clean_Government Loan': 1.0 if funding == 'Government Loan' else 0.0,
        'FundingSource_clean_Partial Scholarship': 1.0 if funding == 'Partial Scholarship' else 0.0,
        'FundingSource_clean_Self Funded': 1.0 if funding == 'Self Funded' else 0.0,
        'ParentalEducation_clean_Secondary': 1.0 if parent_ed == 'Secondary' else 0.0,
        'ParentalEducation_clean_Tertiary': 1.0 if parent_ed == 'Tertiary' else 0.0,
        'Faculty_target_enc': target_enc_maps.get('Faculty', {}).get(faculty, global_y_mean),
        'Department_target_enc': target_enc_maps.get('Department', {}).get(dept, global_y_mean),
        'State_target_enc': target_enc_maps.get('State', {}).get(state, global_y_mean),
        'Faculty_freq_enc': freq_enc_maps.get('Faculty', {}).get(faculty, 0.05),
        'Department_freq_enc': freq_enc_maps.get('Department', {}).get(dept, 0.01),
        'State_freq_enc': freq_enc_maps.get('State', {}).get(state, 0.05)
    }

    # Reorder according to trained feature list
    vec = [raw_row.get(col, 0.0) for col in feature_cols]
    vec_df = pd.DataFrame([vec], columns=feature_cols)
    vec_imp = imputer.transform(vec_df)
    vec_sc = scaler.transform(vec_imp)

    p_cat = float(models['CatBoost'].predict_proba(vec_sc)[:, 1][0])
    p_xgb = float(models['XGBoost'].predict_proba(vec_sc)[:, 1][0])
    p_lr  = float(models['Logistic Regression'].predict_proba(vec_sc)[:, 1][0])

    fedra_score = float(meta_weights[0] * p_cat + meta_weights[1] * p_xgb + meta_weights[2] * p_lr)
    is_at_risk = bool(fedra_score >= th_fedra)

    priority = "Critical Priority" if fedra_score >= 0.60 else ("Moderate Risk" if is_at_risk else "On Track")

    return {
        'risk_score': fedra_score,
        'risk_score_pct': round(fedra_score * 100, 1),
        'is_at_risk': is_at_risk,
        'priority': priority,
        'threshold': th_fedra,
        'breakdown': {
            'CatBoost': round(p_cat * 100, 1),
            'XGBoost': round(p_xgb * 100, 1),
            'Logistic Regression': round(p_lr * 100, 1)
        },
        'vec_sc': vec_sc,
        'raw_row': raw_row
    }

def compute_student_shap(student_dict, artifacts=None, max_features=10):
    """
    Computes local SHAP attribution values for a single student profile using CatBoost TreeSHAP.
    Returns sorted dataframe of risk drivers (positive = increases risk, negative = protective).
    """
    if artifacts is None:
        artifacts = train_suite_models()
    
    from catboost import Pool
    res = predict_single_student(student_dict, artifacts=artifacts)
    vec_sc = res['vec_sc']
    raw_row = res['raw_row']
    feature_cols = artifacts['data']['feature_cols']
    cat = artifacts['models']['CatBoost']
    
    pool = Pool(vec_sc, feature_names=feature_cols)
    shap_vals = cat.get_feature_importance(pool, type='ShapValues')[0]
    contribs = shap_vals[:-1]
    base_val = float(shap_vals[-1])
    
    # Feature display name mapping for student counseling clarity
    friendly_names = {
        'GPA_num': 'Current GPA',
        'PreviousSemesterGPA': 'Previous Semester GPA',
        'gpa_momentum': 'GPA Velocity / Momentum',
        'AttendanceRate_num': 'Lecture Attendance Rate',
        'FailedCourses': 'Failed Courses Count',
        'academic_standing_index': 'Academic Standing Index',
        'failure_load_ratio': 'Failed Courses to Credit Ratio',
        'TuitionPaid': 'Tuition Payment Status',
        'financial_precarity': 'Financial Precarity Index',
        'StudyHoursPerWeek_num': 'Weekly Study Hours',
        'study_effort_score': 'Study Effort Combined Score',
        'DistanceFromHome_num': 'Commute Distance from Home',
        'commute_strain': 'Off-Campus Commute Strain',
        'MonthlyAllowance_num': 'Monthly Allowance',
        'allowance_per_dependant': 'Allowance per Dependant',
        'MentoringSessionsAttended': 'Mentoring Sessions',
        'support_utilization_ratio': 'Support Utilization Ratio',
        'CounsellingVisits': 'Counselling Visits',
        'first_gen_hardship': 'First-Gen Financial Hardship',
        'CoursesCreditLoad': 'Registered Credit Load',
        'YearOfStudy': 'Year of Study',
        'Dependants': 'Number of Dependants',
        'PartTimeJob': 'Has Part-Time Job',
        'LibraryVisitsPerMonth': 'Library Visits per Month',
        'ExtracurricularActivities': 'Extracurricular Activities',
        'Age_num': 'Student Age'
    }
    
    records = []
    for i, col in enumerate(feature_cols):
        val = contribs[i]
        feat_raw_val = raw_row.get(col, None)
        lbl = friendly_names.get(col, col.replace('_clean_', ': ').replace('_target_enc', ' Risk Tier').replace('_freq_enc', ' Prevalence'))
        records.append({
            'feature': col,
            'label': lbl,
            'shap_value': float(val),
            'abs_shap': abs(float(val)),
            'direction': 'Increases Risk' if val > 0 else 'Protective / Lowers Risk',
            'raw_value': feat_raw_val
        })
        
    df_shap = pd.DataFrame(records)
    df_shap = df_shap.sort_values(by='abs_shap', ascending=False).head(max_features)
    return df_shap, base_val, res
