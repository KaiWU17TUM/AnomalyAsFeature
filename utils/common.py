import os
import pandas as pd


file_folder = os.path.join(os.path.abspath(''), '..', 'data')
item_dict = pd.read_csv(os.path.join(file_folder, 'item_dict'), header=0, index_col=[0])

HR_ref_itemids = [223764, 223765, 224647]
BPs_ref_itemids = [223766, 224646, 224645]
BPd_ref_itemids = [226092, 226094, 226096]
ref_itemids = HR_ref_itemids + BPs_ref_itemids + BPd_ref_itemids

col_static = ['age', 'admission_weight', 'admission_type', 'admission_location']
col_refs = ['Orthostatic HR lying', 'Orthostatic HR sitting', 'Orthostatic HR standing'] +\
           ['Orthostatic BPs lying', 'Orthostatic BPs sitting', 'Orthostatic BPs standing'] +\
           ['Orthostatic BPd lying', 'Orthostatic BPd sitting', 'Orthostatic BPd standing']
col_vital = ['HR', 'NBPs', 'NBPd', 'NBPm', 'Temperature F', 'RR', 'SpO2']
col_gp_encoded = [
    'HR_mean', 'NBPs_mean', 'NBPd_mean', 'NBPm_mean', 'Temperature F_mean', 'RR_mean', 'SpO2_mean',
    'HR_var', 'NBPs_var', 'NBPd_var', 'NBPm_var', 'Temperature F_var', 'RR_var', 'SpO2_var',
    'HR_encoded1', 'HR_encoded2', 'HR_encoded3', 'HR_encoded4',
    'NBPs_encoded1', 'NBPs_encoded2', 'NBPs_encoded3', 'NBPs_encoded4',
    'NBPd_encoded1', 'NBPd_encoded2', 'NBPd_encoded3', 'NBPd_encoded4',
    'NBPm_encoded1', 'NBPm_encoded2', 'NBPm_encoded3', 'NBPm_encoded4',
    'Temperature F_encoded1', 'Temperature F_encoded2', 'Temperature F_encoded3', 'Temperature F_encoded4',
    'RR_encoded1', 'RR_encoded2', 'RR_encoded3', 'RR_encoded4',
    'SpO2_encoded1', 'SpO2_encoded2', 'SpO2_encoded3', 'SpO2_encoded4'
]