import os
import pickle
import pandas as pd


file_folder = os.path.join(os.path.abspath(''), '..', 'data')
item_dict = pd.read_csv(os.path.join(file_folder, 'item_dict'), header=0, index_col=[0])

# HR_ref_itemids = [223764, 223765, 224647]
# BPs_ref_itemids = [223766, 224646, 224645]
# BPd_ref_itemids = [226092, 226094, 226096]
# ref_itemids = HR_ref_itemids + BPs_ref_itemids + BPd_ref_itemids
#
# col_static = ['age', 'admission_weight', 'admission_type', 'admission_location']
# col_refs = ['Orthostatic HR lying', 'Orthostatic HR sitting', 'Orthostatic HR standing'] +\
#            ['Orthostatic BPs lying', 'Orthostatic BPs sitting', 'Orthostatic BPs standing'] +\
#            ['Orthostatic BPd lying', 'Orthostatic BPd sitting', 'Orthostatic BPd standing']
# col_vital = ['HR', 'NBPs', 'NBPd', 'NBPm', 'Temperature F', 'RR', 'SpO2']
# col_gp_encoded = [
#     'HR_mean', 'NBPs_mean', 'NBPd_mean', 'NBPm_mean', 'Temperature F_mean', 'RR_mean', 'SpO2_mean',
#     'HR_var', 'NBPs_var', 'NBPd_var', 'NBPm_var', 'Temperature F_var', 'RR_var', 'SpO2_var',
#     'HR_encoded1', 'HR_encoded2', 'HR_encoded3', 'HR_encoded4',
#     'NBPs_encoded1', 'NBPs_encoded2', 'NBPs_encoded3', 'NBPs_encoded4',
#     'NBPd_encoded1', 'NBPd_encoded2', 'NBPd_encoded3', 'NBPd_encoded4',
#     'NBPm_encoded1', 'NBPm_encoded2', 'NBPm_encoded3', 'NBPm_encoded4',
#     'Temperature F_encoded1', 'Temperature F_encoded2', 'Temperature F_encoded3', 'Temperature F_encoded4',
#     'RR_encoded1', 'RR_encoded2', 'RR_encoded3', 'RR_encoded4',
#     'SpO2_encoded1', 'SpO2_encoded2', 'SpO2_encoded3', 'SpO2_encoded4'
# ]

data_path = os.path.join(os.path.abspath(''), '..', 'data_all')
data_raw_path = os.path.join(data_path, 'raw')
data_processed_path = os.path.join(data_path, 'preprocessed')

adm_cohort = pickle.load(open(os.path.join(data_path, 'admission_cohort_48h.pkl'), 'rb'))

selected_chart_items = pickle.load(
    open(os.path.join(data_path, 'selected_chart_items.pkl'), 'rb')
)
selected_vital_iids = selected_chart_items['vital']
selected_lab_iids = selected_chart_items['lab']
del selected_chart_items

selected_numeric_iid = item_dict.loc[
    (item_dict['itemid'].isin(selected_vital_iids + selected_lab_iids)) &
    (item_dict['param_type'].isin(['Numeric', 'Numeric with tag'])),
    'itemid'
].to_list()
selected_text_iid = item_dict.loc[
    (item_dict['itemid'].isin(selected_vital_iids + selected_lab_iids)) &
    (item_dict['param_type']=='Text'),
    'itemid'
].to_list()

info_text_vals = pickle.load(
    open(os.path.join(data_path, 'text_info_onehotkey.pkl'), 'rb')
)
item_text_vals = pickle.load(
    open(os.path.join(data_path, 'text_chart_onehotkey.pkl'), 'rb')
)
item_text_vals = {
    item_dict[item_dict['itemid']==iid]['abbreviation'].item(): item_text_vals[iid]
    for iid in item_text_vals
}

norm_param_info = pickle.load(open(os.path.join(data_path, 'norm_param_info.pkl'), 'rb'))
norm_param_numeric = pickle.load(open(os.path.join(data_path, 'norm_param_chart.pkl'), 'rb'))
norm_param_numeric = {
    item_dict[item_dict['itemid']==iid]['abbreviation'].item(): norm_param_numeric[iid]
    for iid in norm_param_numeric
}
zero_imp_info = {
    col: - norm_param_info[col]['mean'] / norm_param_info[col]['std']
    for col in norm_param_info
}
zero_imp_numeric = {
    col: - norm_param_numeric[col]['mean'] / norm_param_numeric[col]['std']
    for col in norm_param_numeric
}


col_vital = [item_dict[item_dict['itemid']==iid]['abbreviation'].item() for iid in selected_vital_iids]
col_lab = [item_dict[item_dict['itemid']==iid]['abbreviation'].item() for iid in selected_lab_iids]
col_chart = col_vital + col_lab
col_info_numeric = ['age', 'admission_weight']
col_info_text = ['gender', 'admission_type']
col_info = col_info_numeric + col_info_text
col_chart_text = [item_dict[item_dict['itemid']==iid]['abbreviation'].item() for iid in selected_text_iid]
col_chart_numeric = [item_dict[item_dict['itemid']==iid]['abbreviation'].item() for iid in selected_numeric_iid]
col_numeric = col_info_numeric + col_chart_numeric
col_text = col_info_text + col_chart_text
col_numeric_encoded = [
    'HR_mean',
    'NBPs_mean',
    'NBPd_mean',
    'NBPm_mean',
    'RR_mean',
    'PO2 (Arterial)_mean',
    'Hemoglobin_mean',
    'PCO2 (Arterial)_mean',
    'SpO2_mean',
    'Hematocrit (serum)_mean',
    'WBC_mean',
    'AST_mean',
    'Chloride (serum)_mean',
    'Creatinine (serum)_mean',
    'Glucose (serum)_mean',
    'Magnesium_mean',
    'ALT_mean',
    'Sodium (serum)_mean',
    'Temperature F_mean',
    'PH (Arterial)_mean',
    'O2 Flow_mean',
    'FiO2_mean',
    'Arterial Base Excess_mean',
    'Alkaline Phosphate_mean',
    'BUN_mean',
    'Calcium non-ionized_mean',
    'Glucose FS (range 70 -100)_mean',
    'Lactic Acid_mean',
    'Phosphorous_mean',
    'Total Bilirubin_mean',
    'TCO2 (calc) Arterial_mean',
    'Anion gap_mean',
    'Potassium (serum)_mean',
    'HCO3 (serum)_mean',
    'Platelet Count_mean',
    'PT_mean',
    'PTT_mean',
    'INR_mean',
    'HR_var',
    'NBPs_var',
    'NBPd_var',
    'NBPm_var',
    'RR_var',
    'PO2 (Arterial)_var',
    'Hemoglobin_var',
    'PCO2 (Arterial)_var',
    'SpO2_var',
    'Hematocrit (serum)_var',
    'WBC_var',
    'AST_var',
    'Chloride (serum)_var',
    'Creatinine (serum)_var',
    'Glucose (serum)_var',
    'Magnesium_var',
    'ALT_var',
    'Sodium (serum)_var',
    'Temperature F_var',
    'PH (Arterial)_var',
    'O2 Flow_var',
    'FiO2_var',
    'Arterial Base Excess_var',
    'Alkaline Phosphate_var',
    'BUN_var',
    'Calcium non-ionized_var',
    'Glucose FS (range 70 -100)_var',
    'Lactic Acid_var',
    'Phosphorous_var',
    'Total Bilirubin_var',
    'TCO2 (calc) Arterial_var',
    'Anion gap_var',
    'Potassium (serum)_var',
    'HCO3 (serum)_var',
    'Platelet Count_var',
    'PT_var',
    'PTT_var',
    'INR_var',
    'HR_encoded1',
    'HR_encoded2',
    'HR_encoded3',
    'HR_encoded4',
    'NBPs_encoded1',
    'NBPs_encoded2',
    'NBPs_encoded3',
    'NBPs_encoded4',
    'NBPd_encoded1',
    'NBPd_encoded2',
    'NBPd_encoded3',
    'NBPd_encoded4',
    'NBPm_encoded1',
    'NBPm_encoded2',
    'NBPm_encoded3',
    'NBPm_encoded4',
    'RR_encoded1',
    'RR_encoded2',
    'RR_encoded3',
    'RR_encoded4',
    'PO2 (Arterial)_encoded1',
    'PO2 (Arterial)_encoded2',
    'PO2 (Arterial)_encoded3',
    'PO2 (Arterial)_encoded4',
    'Hemoglobin_encoded1',
    'Hemoglobin_encoded2',
    'Hemoglobin_encoded3',
    'Hemoglobin_encoded4',
    'PCO2 (Arterial)_encoded1',
    'PCO2 (Arterial)_encoded2',
    'PCO2 (Arterial)_encoded3',
    'PCO2 (Arterial)_encoded4',
    'SpO2_encoded1',
    'SpO2_encoded2',
    'SpO2_encoded3',
    'SpO2_encoded4',
    'Hematocrit (serum)_encoded1',
    'Hematocrit (serum)_encoded2',
    'Hematocrit (serum)_encoded3',
    'Hematocrit (serum)_encoded4',
    'WBC_encoded1',
    'WBC_encoded2',
    'WBC_encoded3',
    'WBC_encoded4',
    'AST_encoded1',
    'AST_encoded2',
    'AST_encoded3',
    'AST_encoded4',
    'Chloride (serum)_encoded1',
    'Chloride (serum)_encoded2',
    'Chloride (serum)_encoded3',
    'Chloride (serum)_encoded4',
    'Creatinine (serum)_encoded1',
    'Creatinine (serum)_encoded2',
    'Creatinine (serum)_encoded3',
    'Creatinine (serum)_encoded4',
    'Glucose (serum)_encoded1',
    'Glucose (serum)_encoded2',
    'Glucose (serum)_encoded3',
    'Glucose (serum)_encoded4',
    'Magnesium_encoded1',
    'Magnesium_encoded2',
    'Magnesium_encoded3',
    'Magnesium_encoded4',
    'ALT_encoded1',
    'ALT_encoded2',
    'ALT_encoded3',
    'ALT_encoded4',
    'Sodium (serum)_encoded1',
    'Sodium (serum)_encoded2',
    'Sodium (serum)_encoded3',
    'Sodium (serum)_encoded4',
    'Temperature F_encoded1',
    'Temperature F_encoded2',
    'Temperature F_encoded3',
    'Temperature F_encoded4',
    'PH (Arterial)_encoded1',
    'PH (Arterial)_encoded2',
    'PH (Arterial)_encoded3',
    'PH (Arterial)_encoded4',
    'O2 Flow_encoded1',
    'O2 Flow_encoded2',
    'O2 Flow_encoded3',
    'O2 Flow_encoded4',
    'FiO2_encoded1',
    'FiO2_encoded2',
    'FiO2_encoded3',
    'FiO2_encoded4',
    'Arterial Base Excess_encoded1',
    'Arterial Base Excess_encoded2',
    'Arterial Base Excess_encoded3',
    'Arterial Base Excess_encoded4',
    'Alkaline Phosphate_encoded1',
    'Alkaline Phosphate_encoded2',
    'Alkaline Phosphate_encoded3',
    'Alkaline Phosphate_encoded4',
    'BUN_encoded1',
    'BUN_encoded2',
    'BUN_encoded3',
    'BUN_encoded4',
    'Calcium non-ionized_encoded1',
    'Calcium non-ionized_encoded2',
    'Calcium non-ionized_encoded3',
    'Calcium non-ionized_encoded4',
    'Glucose FS (range 70 -100)_encoded1',
    'Glucose FS (range 70 -100)_encoded2',
    'Glucose FS (range 70 -100)_encoded3',
    'Glucose FS (range 70 -100)_encoded4',
    'Lactic Acid_encoded1',
    'Lactic Acid_encoded2',
    'Lactic Acid_encoded3',
    'Lactic Acid_encoded4',
    'Phosphorous_encoded1',
    'Phosphorous_encoded2',
    'Phosphorous_encoded3',
    'Phosphorous_encoded4',
    'Total Bilirubin_encoded1',
    'Total Bilirubin_encoded2',
    'Total Bilirubin_encoded3',
    'Total Bilirubin_encoded4',
    'TCO2 (calc) Arterial_encoded1',
    'TCO2 (calc) Arterial_encoded2',
    'TCO2 (calc) Arterial_encoded3',
    'TCO2 (calc) Arterial_encoded4',
    'Anion gap_encoded1',
    'Anion gap_encoded2',
    'Anion gap_encoded3',
    'Anion gap_encoded4',
    'Potassium (serum)_encoded1',
    'Potassium (serum)_encoded2',
    'Potassium (serum)_encoded3',
    'Potassium (serum)_encoded4',
    'HCO3 (serum)_encoded1',
    'HCO3 (serum)_encoded2',
    'HCO3 (serum)_encoded3',
    'HCO3 (serum)_encoded4',
    'Platelet Count_encoded1',
    'Platelet Count_encoded2',
    'Platelet Count_encoded3',
    'Platelet Count_encoded4',
    'PT_encoded1',
    'PT_encoded2',
    'PT_encoded3',
    'PT_encoded4',
    'PTT_encoded1',
    'PTT_encoded2',
    'PTT_encoded3',
    'PTT_encoded4',
    'INR_encoded1',
    'INR_encoded2',
    'INR_encoded3',
    'INR_encoded4'
]
