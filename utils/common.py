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
