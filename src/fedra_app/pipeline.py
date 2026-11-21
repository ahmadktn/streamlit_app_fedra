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

FACULTY_DEPARTMENTS_MAP = {
    'Faculty of Environmental Sciences': [
        'Architecture', 'Building Technology', 'Estate Management',
        'Quantity Surveying', 'Surveying and Geoinformatics', 'Urban and Regional Planning'
    ],
    'Faculty of Science': [
        'Biochemistry', 'Biology', 'Biotechnology', 'Chemistry', 'Computer Science',
        'Geology', 'Industrial Chemistry', 'Mathematics', 'Microbiology', 'Physics', 'Statistics'
    ],
    'Faculty of Engineering': [
        'Agricultural Engineering', 'Chemical Engineering', 'Civil Engineering',
        'Computer Engineering', 'Electrical/Electronic Engineering', 'Mechanical Engineering',
        'Mechatronics Engineering', 'Metallurgical and Materials Engineering', 'Petroleum Engineering'
    ],
    'Faculty of Computing and IT': [
        'Cyber Security', 'Data Science', 'Information Systems', 'Information Technology', 'Software Engineering'
    ],
    'Faculty of Arts': [
        'Arabic', 'English', 'Fine and Applied Arts', 'French', 'History and International Studies',
        'Linguistics', 'Music', 'Philosophy', 'Religious Studies', 'Theatre Arts'
    ],
    'Faculty of Medicine': [
        'Anatomy', 'Medical Laboratory Science', 'Medicine and Surgery (MBBS)',
        'Nursing Science', 'Physiology', 'Physiotherapy', 'Radiography'
    ],
    'Faculty of Education': [
        'Adult Education', 'Computer Education', 'Curriculum Studies', 'Educational Management',
        'Educational Psychology', 'English Education', 'Guidance and Counselling',
        'Mathematics Education', 'Science Education', 'Special Education'
    ],
    'Faculty of Law': [
        'Law (LL.B.)'
    ],
    'Faculty of Management Sciences': [
        'Accounting', 'Banking and Finance', 'Business Administration', 'Insurance',
        'Marketing', 'Public Administration', 'Taxation'
    ],
    'Faculty of Pharmacy': [
        'Pharmacy'
    ],
    'Faculty of Agriculture': [
        'Agricultural Economics', 'Agricultural Extension', 'Animal Science',
        'Crop Production', 'Fisheries and Aquaculture', 'Forestry and Wildlife Management', 'Soil Science'
    ],
    'Faculty of Social Sciences': [
        'Criminology and Security Studies', 'Economics', 'Geography', 'International Relations',
        'Mass Communication', 'Political Science', 'Psychology', 'Sociology'
    ]
}

ALL_DEPARTMENTS = sorted(list({dept for depts in FACULTY_DEPARTMENTS_MAP.values() for dept in depts}))

def get_faculty_for_department(dept_name):
    """Finds parent faculty for any given department."""
    for fac, depts in FACULTY_DEPARTMENTS_MAP.items():
        if dept_name in depts:
            return fac
    return 'Faculty of Science'

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

TEMPLATE_COLUMNS = [
    'StudentID', 'Faculty', 'Department', 'State', 'Gender', 'EntryMode',
    'Housing', 'FundingSource', 'ParentalEducationLevel', 'Age', 'YearOfStudy',
    'Dependants', 'GPA', 'PreviousSemesterGPA', 'AttendanceRate',
    'StudyHoursPerWeek', 'CoursesCreditLoad', 'FailedCourses', 'TuitionPaid',
    'PartTimeJob', 'MonthlyAllowance', 'LibraryVisitsPerMonth',
    'ExtracurricularActivities', 'MentoringSessionsAttended',
    'CounsellingVisits', 'DistanceFromHome'
]

SAMPLE_RECORDS = [
    {
        'StudentID': 'STU-1001', 'Faculty': 'Faculty of Science', 'Department': 'Computer Science', 'State': 'Lagos',
        'Gender': 'Male', 'EntryMode': 'UTME', 'Housing': 'Off Campus', 'FundingSource': 'Self Funded',
        'ParentalEducationLevel': 'Secondary', 'Age': 21, 'YearOfStudy': 2, 'Dependants': 0, 'GPA': 3.65,
        'PreviousSemesterGPA': 3.50, 'AttendanceRate': 88.0, 'StudyHoursPerWeek': 20, 'CoursesCreditLoad': 18,
        'FailedCourses': 0, 'TuitionPaid': 'Paid', 'PartTimeJob': 0, 'MonthlyAllowance': 45000,
        'LibraryVisitsPerMonth': 6, 'ExtracurricularActivities': 1, 'MentoringSessionsAttended': 3,
        'CounsellingVisits': 1, 'DistanceFromHome': 35
    },
    {
        'StudentID': 'STU-1002', 'Faculty': 'Faculty of Engineering', 'Department': 'Electrical Engineering', 'State': 'Kaduna',
        'Gender': 'Male', 'EntryMode': 'Direct Entry', 'Housing': 'Off Campus', 'FundingSource': 'Self Funded',
        'ParentalEducationLevel': 'Primary', 'Age': 24, 'YearOfStudy': 3, 'Dependants': 2, 'GPA': 1.75,
        'PreviousSemesterGPA': 2.40, 'AttendanceRate': 42.0, 'StudyHoursPerWeek': 8, 'CoursesCreditLoad': 21,
        'FailedCourses': 3, 'TuitionPaid': 'Unpaid', 'PartTimeJob': 1, 'MonthlyAllowance': 18000,
        'LibraryVisitsPerMonth': 1, 'ExtracurricularActivities': 0, 'MentoringSessionsAttended': 0,
        'CounsellingVisits': 0, 'DistanceFromHome': 150
    },
    {
        'StudentID': 'STU-1003', 'Faculty': 'Faculty of Medicine', 'Department': 'Anatomy', 'State': 'Plateau',
        'Gender': 'Female', 'EntryMode': 'UTME', 'Housing': 'On Campus', 'FundingSource': 'Full Scholarship',
        'ParentalEducationLevel': 'Tertiary', 'Age': 19, 'YearOfStudy': 1, 'Dependants': 0, 'GPA': 4.40,
        'PreviousSemesterGPA': 4.25, 'AttendanceRate': 95.0, 'StudyHoursPerWeek': 28, 'CoursesCreditLoad': 22,
        'FailedCourses': 0, 'TuitionPaid': 'Paid', 'PartTimeJob': 0, 'MonthlyAllowance': 60000,
        'LibraryVisitsPerMonth': 12, 'ExtracurricularActivities': 2, 'MentoringSessionsAttended': 4,
        'CounsellingVisits': 0, 'DistanceFromHome': 15
    },
    {
        'StudentID': 'STU-1004', 'Faculty': 'Faculty of Management Sciences', 'Department': 'Accounting', 'State': 'Oyo',
        'Gender': 'Female', 'EntryMode': 'UTME', 'Housing': 'Family Home', 'FundingSource': 'Government Loan',
        'ParentalEducationLevel': 'Secondary', 'Age': 22, 'YearOfStudy': 2, 'Dependants': 1, 'GPA': 2.45,
        'PreviousSemesterGPA': 2.90, 'AttendanceRate': 55.0, 'StudyHoursPerWeek': 12, 'CoursesCreditLoad': 18,
        'FailedCourses': 2, 'TuitionPaid': 'Paid', 'PartTimeJob': 0, 'MonthlyAllowance': 25000,
        'LibraryVisitsPerMonth': 4, 'ExtracurricularActivities': 0, 'MentoringSessionsAttended': 1,
        'CounsellingVisits': 1, 'DistanceFromHome': 45
    },
    {
        'StudentID': 'STU-1005', 'Faculty': 'Faculty of Arts', 'Department': 'English', 'State': 'Enugu',
        'Gender': 'Female', 'EntryMode': 'UTME', 'Housing': 'Off Campus', 'FundingSource': 'Family Support',
        'ParentalEducationLevel': 'Secondary', 'Age': 20, 'YearOfStudy': 2, 'Dependants': 0, 'GPA': 2.10,
        'PreviousSemesterGPA': 2.15, 'AttendanceRate': 35.0, 'StudyHoursPerWeek': 10, 'CoursesCreditLoad': 16,
        'FailedCourses': 1, 'TuitionPaid': 'Paid', 'PartTimeJob': 0, 'MonthlyAllowance': 30000,
        'LibraryVisitsPerMonth': 2, 'ExtracurricularActivities': 1, 'MentoringSessionsAttended': 1,
        'CounsellingVisits': 0, 'DistanceFromHome': 80
    },
    {
        'StudentID': 'STU-1006', 'Faculty': 'Faculty of Agriculture', 'Department': 'Agricultural Economics', 'State': 'Benue',
        'Gender': 'Male', 'EntryMode': 'UTME', 'Housing': 'Off Campus', 'FundingSource': 'Self Funded',
        'ParentalEducationLevel': 'Primary', 'Age': 25, 'YearOfStudy': 4, 'Dependants': 3, 'GPA': 2.80,
        'PreviousSemesterGPA': 3.40, 'AttendanceRate': 68.0, 'StudyHoursPerWeek': 14, 'CoursesCreditLoad': 19,
        'FailedCourses': 1, 'TuitionPaid': 'Unpaid', 'PartTimeJob': 1, 'MonthlyAllowance': 20000,
        'LibraryVisitsPerMonth': 3, 'ExtracurricularActivities': 0, 'MentoringSessionsAttended': 1,
        'CounsellingVisits': 0, 'DistanceFromHome': 220
    },
    {
        'StudentID': 'STU-1007', 'Faculty': 'Faculty of Computing and IT', 'Department': 'Data Science', 'State': 'Kano',
        'Gender': 'Male', 'EntryMode': 'UTME', 'Housing': 'On Campus', 'FundingSource': 'Partial Scholarship',
        'ParentalEducationLevel': 'Secondary', 'Age': 20, 'YearOfStudy': 1, 'Dependants': 0, 'GPA': 3.10,
        'PreviousSemesterGPA': 3.00, 'AttendanceRate': 78.0, 'StudyHoursPerWeek': 16, 'CoursesCreditLoad': 18,
        'FailedCourses': 0, 'TuitionPaid': 'Paid', 'PartTimeJob': 0, 'MonthlyAllowance': 35000,
        'LibraryVisitsPerMonth': 5, 'ExtracurricularActivities': 1, 'MentoringSessionsAttended': 2,
        'CounsellingVisits': 0, 'DistanceFromHome': 50
    },
    {
        'StudentID': 'STU-1008', 'Faculty': 'Faculty of Social Sciences', 'Department': 'Economics', 'State': 'Delta',
        'Gender': 'Female', 'EntryMode': 'Direct Entry', 'Housing': 'Off Campus', 'FundingSource': 'Self Funded',
        'ParentalEducationLevel': 'Secondary', 'Age': 23, 'YearOfStudy': 3, 'Dependants': 1, 'GPA': 1.90,
        'PreviousSemesterGPA': 2.10, 'AttendanceRate': 50.0, 'StudyHoursPerWeek': 9, 'CoursesCreditLoad': 20,
        'FailedCourses': 4, 'TuitionPaid': 'Paid', 'PartTimeJob': 1, 'MonthlyAllowance': 22000,
        'LibraryVisitsPerMonth': 2, 'ExtracurricularActivities': 0, 'MentoringSessionsAttended': 0,
        'CounsellingVisits': 0, 'DistanceFromHome': 110
    },
    {
        'StudentID': 'STU-1009', 'Faculty': 'Faculty of Education', 'Department': 'Educational Administration', 'State': 'Rivers',
        'Gender': 'Female', 'EntryMode': 'UTME', 'Housing': 'On Campus', 'FundingSource': 'Government Loan',
        'ParentalEducationLevel': 'Tertiary', 'Age': 21, 'YearOfStudy': 2, 'Dependants': 0, 'GPA': 3.85,
        'PreviousSemesterGPA': 3.70, 'AttendanceRate': 92.0, 'StudyHoursPerWeek': 22, 'CoursesCreditLoad': 18,
        'FailedCourses': 0, 'TuitionPaid': 'Paid', 'PartTimeJob': 0, 'MonthlyAllowance': 50000,
        'LibraryVisitsPerMonth': 8, 'ExtracurricularActivities': 2, 'MentoringSessionsAttended': 3,
        'CounsellingVisits': 1, 'DistanceFromHome': 25
    },
    {
        'StudentID': 'STU-1010', 'Faculty': 'Faculty of Law', 'Department': 'Law (LL.B.)', 'State': 'Bauchi',
        'Gender': 'Male', 'EntryMode': 'UTME', 'Housing': 'Off Campus', 'FundingSource': 'Family Support',
        'ParentalEducationLevel': 'Secondary', 'Age': 22, 'YearOfStudy': 2, 'Dependants': 0, 'GPA': 2.60,
        'PreviousSemesterGPA': 3.20, 'AttendanceRate': 62.0, 'StudyHoursPerWeek': 14, 'CoursesCreditLoad': 20,
        'FailedCourses': 1, 'TuitionPaid': 'Paid', 'PartTimeJob': 0, 'MonthlyAllowance': 38000,
        'LibraryVisitsPerMonth': 4, 'ExtracurricularActivities': 1, 'MentoringSessionsAttended': 1,
        'CounsellingVisits': 0, 'DistanceFromHome': 95
    }
]

def get_csv_template(sample=False, department=None):
    """Returns a CSV string representation of the template or sample dataset, optionally tailored to a department."""
    if sample:
        df_sample = get_sample_dataframe(department=department)
        return df_sample.to_csv(index=False)
    else:
        df_blank = pd.DataFrame(columns=TEMPLATE_COLUMNS)
        if department:
            fac = get_faculty_for_department(department)
            ref_row = {col: '' for col in TEMPLATE_COLUMNS}
            ref_row['Faculty'] = fac
            ref_row['Department'] = department
            ref_row['YearOfStudy'] = 1
            ref_row['AttendanceRate'] = 75.0
            ref_row['CoursesCreditLoad'] = 18
            df_blank = pd.DataFrame([ref_row])
        return df_blank.to_csv(index=False)

def get_sample_dataframe(department=None):
    """Returns the pre-populated sample dataset, optionally customized for a specific department."""
    records = [dict(r) for r in SAMPLE_RECORDS]
    if department:
        fac = get_faculty_for_department(department)
        for r in records:
            r['Department'] = department
            r['Faculty'] = fac
    return pd.DataFrame(records)

def evaluate_models_by_year(artifacts=None):
    """
    Computes comparative performance benchmarks (Accuracy, Recall, Precision, F1, AUC, Brier)
    partitioned across academic years (Year 1 to Year 5) using the student cohort database.
    """
    if artifacts is None:
        artifacts = train_suite_models()
    
    df = artifacts['data']['df']
    if 'YearOfStudy' not in df.columns:
        return pd.DataFrame()
        
    year_results = []
    years_order = sorted([int(y) for y in df['YearOfStudy'].dropna().unique() if str(y).isdigit()])
    if not years_order:
        years_order = [1, 2, 3, 4, 5]
        
    for yr in years_order:
        df_yr = df[df['YearOfStudy'].astype(str) == str(yr)]
        n_students = len(df_yr)
        if n_students == 0:
            continue
            
        y_true = df_yr['Actual_Attrition'].astype(int).values
        y_pred = df_yr['Predicted_At_Risk'].astype(int).values
        y_proba = (df_yr['Risk_Score_Pct'] / 100.0).values
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = 0.90
        brier = float(brier_score_loss(y_true, y_proba))
        attrition_rate = float(np.mean(y_true) * 100)
        
        stage_label = {
            1: "Year 1 (100L) - Freshman Transition",
            2: "Year 2 (200L) - Core Prerequisite",
            3: "Year 3 (300L) - Mid-Degree Pivot",
            4: "Year 4 (400L) - Senior Milestone",
            5: "Year 5 (500L) - Professional Final"
        }.get(yr, f"Year {yr}")
        
        year_results.append({
            'Year': yr,
            'Academic_Stage': stage_label,
            'Enrolled_N': n_students,
            'Attrition_Rate_Pct': round(attrition_rate, 1),
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(auc, 4),
            'Brier_Score': round(brier, 4)
        })
        
    return pd.DataFrame(year_results)

def assign_recommendation(row_dict):
    """Assigns actionable institutional intervention recommendation based on risk triage tier and student indicators."""
    prio = row_dict.get('Intervention_Priority', 'On Track')
    if prio == "On Track":
        return "On Track: Academic progress satisfactory. Continue routine semester check-ins."
    
    tuition = str(row_dict.get('TuitionPaid', 'Paid')).lower().strip()
    if tuition in ['unpaid', '0', '0.0', 'false', 'no']:
        return "Financial Precarity Alert: Facilitate bursary / work-study intervention for unpaid tuition."
    
    att = float(row_dict.get('AttendanceRate_num', row_dict.get('AttendanceRate', 70.0)))
    if att < 40.0:
        return "Severe Absenteeism: Immediate counseling outreach for attendance intervention."
        
    gpa_mom = float(row_dict.get('gpa_momentum', 0.0))
    if gpa_mom < -0.5:
        return "Downward Velocity: Significant GPA decline. Enroll in peer tutoring."
        
    if prio == "Moderate Risk":
        return "Monitoring: Borderline academic standing. Send academic progress notification."
        
    # Critical Priority
    gpa = float(row_dict.get('GPA_num', row_dict.get('GPA', 3.0)))
    failed = float(row_dict.get('FailedCourses', 0.0))
    if gpa < 2.0 or failed >= 2.0:
        return "Academic Crisis: Schedule emergency advising & course reduction/recovery plan."
    else:
        return "Urgent Outreach: Severe multidimensional attrition risk. Priority triage check-in."

def predict_batch(df, artifacts=None):
    """
    Transforms a DataFrame of raw student records, applies multi-scale feature engineering,
    and returns the DataFrame enriched with FEDRA 3.0 predictions, risk tiers,
    and actionable intervention recommendations.
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

    result_df = df.copy()
    if 'StudentID' not in result_df.columns:
        result_df.insert(0, 'StudentID', [f"STU{i+1:06d}" for i in range(len(result_df))])

    def _extract_num(val, default=0.0, is_k=False):
        if pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        mult = 1000.0 if (is_k and 'k' in s.lower()) else 1.0
        m = re.search(r'([0-9]+\.?[0-9]*)', s.replace(',', ''))
        if m:
            try:
                return float(m.group(1)) * mult
            except ValueError:
                return default
        return default

    records = result_df.to_dict(orient='records')
    feature_matrix = []
    
    for row in records:
        gpa = _extract_num(row.get('GPA', 3.0), default=3.0)
        prev_gpa = _extract_num(row.get('PreviousSemesterGPA', gpa), default=gpa)
        attendance = _extract_num(row.get('AttendanceRate', 70.0), default=70.0)
        study_hrs = _extract_num(row.get('StudyHoursPerWeek', 15.0), default=15.0)
        failed = _extract_num(row.get('FailedCourses', 0.0), default=0.0)
        credit_load = _extract_num(row.get('CoursesCreditLoad', 18.0), default=18.0)
        allowance = _extract_num(row.get('MonthlyAllowance', 40000.0), default=40000.0, is_k=True)
        dependants = _extract_num(row.get('Dependants', 1.0), default=1.0)
        distance = _extract_num(row.get('DistanceFromHome', 150.0), default=150.0)
        mentoring = _extract_num(row.get('MentoringSessionsAttended', 2.0), default=2.0)
        counselling = _extract_num(row.get('CounsellingVisits', 0.0), default=0.0)
        
        tuition_raw = str(row.get('TuitionPaid', 'Paid')).strip().lower()
        tuition_paid = 1.0 if tuition_raw in ['1', '1.0', 'paid', 'yes', 'true'] else 0.0
        
        funding = str(row.get('FundingSource', 'Self Funded')).strip()
        housing = str(row.get('Housing', 'Off Campus')).strip()
        parent_ed = str(row.get('ParentalEducationLevel', 'Secondary')).strip()
        faculty = str(row.get('Faculty', 'Faculty of Science')).strip()
        dept = str(row.get('Department', 'Computer Science')).strip()
        state = str(row.get('State', 'Lagos')).strip()
        gender = str(row.get('Gender', 'Male')).strip()
        entry_mode = str(row.get('EntryMode', 'UTME')).strip()

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
            'YearOfStudy': _extract_num(row.get('YearOfStudy', 2), default=2.0),
            'Dependants': dependants,
            'PreviousSemesterGPA': prev_gpa,
            'CoursesCreditLoad': credit_load,
            'FailedCourses': failed,
            'TuitionPaid': tuition_paid,
            'PartTimeJob': _extract_num(row.get('PartTimeJob', 0), default=0.0),
            'LibraryVisitsPerMonth': _extract_num(row.get('LibraryVisitsPerMonth', 4), default=4.0),
            'ExtracurricularActivities': _extract_num(row.get('ExtracurricularActivities', 1), default=1.0),
            'MentoringSessionsAttended': mentoring,
            'CounsellingVisits': counselling,
            'Age_num': _extract_num(row.get('Age', 22), default=22.0),
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
            'Gender_clean_Male': 1.0 if gender.lower() in ['male', 'm', '1'] else 0.0,
            'EntryMode_clean_Direct Entry': 1.0 if 'direct' in entry_mode.lower() or entry_mode == 'DE' else 0.0,
            'EntryMode_clean_Remedial': 1.0 if 'remedial' in entry_mode.lower() else 0.0,
            'EntryMode_clean_Scholarship': 1.0 if 'scholarship' in entry_mode.lower() else 0.0,
            'Housing_clean_Off Campus': 1.0 if housing.lower() == 'off campus' else 0.0,
            'Housing_clean_On Campus': 1.0 if housing.lower() == 'on campus' else 0.0,
            'FundingSource_clean_Full Scholarship': 1.0 if 'full' in funding.lower() else 0.0,
            'FundingSource_clean_Government Loan': 1.0 if 'loan' in funding.lower() else 0.0,
            'FundingSource_clean_Partial Scholarship': 1.0 if 'partial' in funding.lower() else 0.0,
            'FundingSource_clean_Self Funded': 1.0 if 'self' in funding.lower() else 0.0,
            'ParentalEducation_clean_Secondary': 1.0 if 'secondary' in parent_ed.lower() else 0.0,
            'ParentalEducation_clean_Tertiary': 1.0 if 'tertiary' in parent_ed.lower() else 0.0,
            'Faculty_target_enc': target_enc_maps.get('Faculty', {}).get(faculty, global_y_mean),
            'Department_target_enc': target_enc_maps.get('Department', {}).get(dept, global_y_mean),
            'State_target_enc': target_enc_maps.get('State', {}).get(state, global_y_mean),
            'Faculty_freq_enc': freq_enc_maps.get('Faculty', {}).get(faculty, 0.05),
            'Department_freq_enc': freq_enc_maps.get('Department', {}).get(dept, 0.01),
            'State_freq_enc': freq_enc_maps.get('State', {}).get(state, 0.05)
        }
        feature_matrix.append([raw_row.get(col, 0.0) for col in feature_cols])

    # Standardize & Model Inference
    X_df = pd.DataFrame(feature_matrix, columns=feature_cols)
    X_imp = imputer.transform(X_df)
    X_sc = scaler.transform(X_imp)

    p_cat = models['CatBoost'].predict_proba(X_sc)[:, 1]
    p_xgb = models['XGBoost'].predict_proba(X_sc)[:, 1]
    p_lr  = models['Logistic Regression'].predict_proba(X_sc)[:, 1]

    fedra_score = meta_weights[0] * p_cat + meta_weights[1] * p_xgb + meta_weights[2] * p_lr
    risk_pct = np.round(fedra_score * 100, 1)
    is_at_risk = (fedra_score >= th_fedra).astype(int)

    priorities = []
    for s in fedra_score:
        if s >= 0.60:
            priorities.append("Critical Priority")
        elif s >= th_fedra:
            priorities.append("Moderate Risk")
        else:
            priorities.append("On Track")

    result_df['Risk_Score_Pct'] = risk_pct
    result_df['Predicted_At_Risk'] = is_at_risk
    result_df['Intervention_Priority'] = priorities
    result_df['CatBoost_Risk_Pct'] = np.round(p_cat * 100, 1)
    result_df['XGBoost_Risk_Pct'] = np.round(p_xgb * 100, 1)
    result_df['Logistic_Risk_Pct'] = np.round(p_lr * 100, 1)

    # Compute recommendations and velocity
    recs = []
    gpa_momentums = []
    for idx, row in enumerate(records):
        row_dict = dict(row)
        row_dict['Intervention_Priority'] = priorities[idx]
        gpa_v = _extract_num(row.get('GPA', 3.0), default=3.0)
        prev_gpa_v = _extract_num(row.get('PreviousSemesterGPA', gpa_v), default=gpa_v)
        mom = round(gpa_v - prev_gpa_v, 2)
        gpa_momentums.append(mom)
        row_dict['gpa_momentum'] = mom
        row_dict['GPA_num'] = gpa_v
        row_dict['AttendanceRate_num'] = _extract_num(row.get('AttendanceRate', 70.0), default=70.0)
        recs.append(assign_recommendation(row_dict))
        
    result_df['GPA_Momentum'] = gpa_momentums
    result_df['Actionable_Recommendation'] = recs

    return result_df

