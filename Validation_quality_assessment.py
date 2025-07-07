## This code validates the automated quality assessement on resting state EEGs compared to expert judgement. 
# A spearmans rank correlation is performed and a scatter plot is made for visualisation.
# Scores of expert judgement can be given in an Excel file based on participant number and date of EEG recording and the 
# automated quality scores can be taken from the Quality_results file from the Validation_preprocessing.py file or 
# calculated separately with the assess_eeg_quality funtion in the Functions_quality_assessment file.

#%%
# import required toolboxes
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

#%%
# input 
database_excel_expert_judgement = pd.read_excel("EEG_visual_assessment.xlsx")
database_excel_automated_quality_assessment = pd.read_excel("Quality_results.xlsx", sheet_name = 'base_filters')

#%%
# prepare data for validation

expert_judgement = pd.DataFrame({
    'Number': database_excel_expert_judgement['Participant'],
    'Date': database_excel_expert_judgement['Date_of_Exam'], 
    'N-Score': database_excel_expert_judgement['Quality']
})
expert_judgement['Date'] = pd.to_datetime(expert_judgement['Date'], format='%d%m%Y')
expert_judgement['Number'] = expert_judgement['Number'].astype(str)
expert_judgement['N-Score'] = expert_judgement['N-Score'].astype(str).str[0].astype(int)


df = pd.DataFrame(columns=['Number', 'Date'])
df[['Number', 'Date']] = database_excel_automated_quality_assessment['filename'].str.split('_', expand=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d%m%Y')

automated_quality_assessment = pd.DataFrame({
    'Date': df['Date'],
    'Number': df['Number'],
    'Q-Score': database_excel_automated_quality_assessment['quality_ratio']
})
automated_quality_assessment['Number'] = automated_quality_assessment['Number'].astype(str)

df = pd.merge(expert_judgement, automated_quality_assessment, on=['Number', 'Date'], how='inner')

#%% 
# #visualisation
%matplotlib Tk

plt.figure(figsize = (7,7))
sns.stripplot(x='N-Score', y='Q-Score', data=df, order=[0, 1, 2, 3], jitter = True, alpha =0.7)
plt.xticks(ticks=[0, 1, 2, 3], labels=["0: Unusable", "1: Significant Artifacts", "2: Moderate Artifacts", "3: Few Artifacts"])
plt.title("Scatter plot of automated quality score against expert judgement")
plt.tight_layout()
plt.show()


#%% 
# statistical analysis

rho, p_value = spearmanr(df['N-Score'], df['Q-Score'])
print(f"Spearman's rho: {rho:.3f}, p-value: {p_value:.5f}")
