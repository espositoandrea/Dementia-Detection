import argparse
import pandas as pd
import re
import yaml
import numpy as np

parser = argparse.ArgumentParser("Label the OASIS3 PET dataset")
parser.add_argument('clinical_data')
parser.add_argument('pup_data')
parser.add_argument('-o', '--output', type=argparse.FileType('w'), default='-')
args = parser.parse_args()

pup_data = pd.read_csv(args.pup_data)
clinical_data = pd.read_csv(args.clinical_data)

# By default, the CSVs do not contain some information explicitly
# (although the columns have been defined). Thus, here we set them
# to usable values.

pup_data['Date'] = (pup_data['PUP_PUPTIMECOURSEDATA ID']
                        .apply(lambda s: int(s.split('_')[3][1:])))
pup_data['Subject'] = (pup_data['PUP_PUPTIMECOURSEDATA ID']
                           .apply(lambda s: s.split('_')[0]))

clinical_data['Date'] = (clinical_data['ADRC_ADRCCLINICALDATA ID']
                             .apply(lambda s: int(s.split('_')[2][1:])))


# We can now associate a label to each diagnosis
clinical_data.replace({'.': None, np.nan: None}, inplace=True)

labels = set()
for i in range(1, 6):
    labels = labels.union(clinical_data[f'dx{i}'].unique())
simplified = {str(k): re.match('^(AD dem|Vasc.*? dem|Frontotemporal dem|other mental retarAD demion|(Active )?DLBD|Active PSNP|Dementia)', str(k), re.I) is not None for k in labels}
simplified = {k: v or re.match('^uncertain.*?dem', k) is not None for k, v in simplified.items()}
# We classify uncertain cases as sick

# The final diagnosis is the OR of all the diagnosis
clinical_data['dementia'] = clinical_data[[f'dx{i}' for i in range(1,6)]].replace(simplified).any(axis=1)

# We try and fix false negatives and false positives
clinical_data = clinical_data.sort_values(by=['Subject', 'Date']).reset_index(drop=True)
current_subj = None
for index, row in clinical_data.iterrows():
    if current_subj != row['Subject']:
        current_subj = row['Subject']
        prec = (None, None) # i-2, i-1
        succ = (clinical_data.at[index + 1, 'dementia'] if index + 1 in clinical_data.index and clinical_data.at[index + 1, 'Subject'] == current_subj else None, clinical_data.at[index + 2, 'dementia'] if index + 2 in clinical_data.index and clinical_data.at[index + 2, 'Subject'] == current_subj else None)
    
    if row['dementia'] == True:
        new = prec[1] or any(succ)
    else:
        new = any(prec) and any(succ)
    clinical_data.at[index, 'dementia'] = new
    
    prec = (prec[1], new) # i-1, i
    succ = (succ[1], clinical_data.at[index + 3, 'dementia'] if index + 3 in clinical_data.index and clinical_data.at[index + 3, 'Subject'] == current_subj else None)

# We associated each PET scan to the nearest label in time
labeled_dataset = pup_data.copy()
labeled_dataset.drop(columns=['procType', 'model', 'templateType', 'FSId', 'MRId', 'mocoError', 'regError', 'Centil_fBP_TOT_CORTMEAN', 'Centil_fSUVR_TOT_CORTMEAN', 'Centil_fBP_rsf_TOT_CORTMEAN', 'Centil_fSUVR_rsf_TOT_CORTMEAN'], inplace=True)
labeled_dataset['Label'] = None
for index, row in labeled_dataset.iterrows():
    rows_of_subject = clinical_data.loc[clinical_data['Subject'] == row['Subject']][['Date', 'dementia']]
    rows_of_subject['Date'] = rows_of_subject['Date'].apply(lambda x: abs(x - row['Date']))
    labeled_dataset.at[index,'Label'] = (rows_of_subject.groupby(by='Date', as_index=False) 
                                                        .any()
                                                        .sort_values(by='Date')
                                                        .loc[0]['dementia'])
labeled_dataset.to_csv(args.output)