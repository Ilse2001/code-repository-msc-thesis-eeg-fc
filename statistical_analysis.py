## Code for statistical analysis of relation between functional connectivity and visual processing speed
# 1. paired t-tests for comparison eyes open and closed functional connectivity (Berger effect)
# 2. correlations functional connectivity and network metrics
# 3. correlation iAPF 

# all EEG-based measurements for the correlations will be extracted from de functional connectivity results file 
# created by the FC-analysis.py script. The participants to include can be given by a list (the correct results files 
# will be searched for in the assigned folder) or all the results files in the assigned folder can be used. The age, PSI scores (and if applicable iAPF assessments) should also be 
# given in a list with the same order as the participants numbers. 

#%%
# import required toolboxes
import os
import numpy as np
from scipy.stats import ttest_rel, spearmanr
import matplotlib.pyplot as plt

%matplotlib Tk

#%%
# input 
FC_result_folder = ('FC_results\pipeline') #fill in the name of the folder with the functional connectivity result files
participants = ['108076195', '108076186', '116014859'] # give a list of participant numbers to include (currently only applied to correlations)
age_categories = {'1': 'red', '2': 'green', '3': 'blue'} # age categories plots
ages = np.array(['1', '2', '3']) # ages of the participants in same order as numbers
PSI_scores = [21, 22, 23] # PSI_scores of the participant in same order as numbers
clinical_iapf_assessment = [8, 9, 10] # add the clinical assessment of iapf if required (also automated assessment available in FC-results file)

#%% 
# 1. paired t-tests for comparison eyes open and closed functional connectivity (Berger effect)
all_files_from_folder = os.listdir(FC_result_folder)
all_FC_result_files = [x for x in all_files_from_folder if x.endswith("functional_connectivity_results.npz")]

a_occ_open = []
a_occ_closed = []
a_ih_open = []
a_ih_closed = []

for file in all_FC_result_files: 
    path = os.path.join(FC_result_folder, file)
    data = np.load(path, allow_pickle=True)
    a_open = data['lobe_connectivity'].item()['alpha_open']
    a_occ_open.append((np.sum(a_open[3,0:3]) + np.sum(a_open[4:6,3]))/5)
    a_closed = data['lobe_connectivity'].item()['alpha_closed']
    a_occ_closed.append((np.sum(a_closed[3,0:3]) + np.sum(a_closed[4:6,3]))/5)
    a_ih_open.append(data['hemisphere_connectivity'].item()['alpha_open']['between'])
    a_ih_closed.append(data['hemisphere_connectivity'].item()['alpha_closed']['between'])

t_stat_occ, p_val_occ = ttest_rel(a_occ_open, a_occ_closed)
t_stat_hem, p_val_hem = ttest_rel(a_ih_open, a_ih_closed)

print("\nResults paired t-tests:")
print(f"\nOccipital connectivity difference between eyes open and eyes closed:")
print(f"t_stat: {t_stat_occ}")
print(f"p_value: {p_val_occ}")
print(f"\nInterhemispheric connectivity difference between eyes open and eyes closed:")
print(f"t_stat: {t_stat_hem}")
print(f"p_value: {p_val_hem}")

#%%
# 2. correlations functional connectivity and network metrics 
br_ih_closed = []
br_of_closed = []
br_mst_d_closed = []
br_mst_lf_closed = []
br_ih_open = []
br_of_open = []
br_mst_d_open = []
br_mst_lf_open = []

for participant in participants:
    for file in all_FC_result_files: 
        if participant in file:
            path = os.path.join(FC_result_folder, file)
            data = np.load(path, allow_pickle=True)
            br_of_closed.append(data['lobe_connectivity'].item()['broad_closed'][3,0])
            br_ih_closed.append(data['hemisphere_connectivity'].item()['broad_closed']['between'])
            br_mst_d_closed.append(data['minimum_spanning_tree'].item()['broad_closed']['diameter'])
            br_mst_lf_closed.append(data['minimum_spanning_tree'].item()['broad_closed']['leaf_fraction'])
            br_of_open.append(data['lobe_connectivity'].item()['broad_open'][3,0])
            br_ih_open.append(data['hemisphere_connectivity'].item()['broad_open']['between'])
            br_mst_d_open.append(data['minimum_spanning_tree'].item()['broad_open']['diameter'])
            br_mst_lf_open.append(data['minimum_spanning_tree'].item()['broad_open']['leaf_fraction'])
            break 

FC_metrics_closed = {
    'fronto-occipital connectivity': br_of_closed,
    'interhemispheric connectivity': br_ih_closed,
    'MST diameter': br_mst_d_closed,
    'MST leaf fraction': br_mst_lf_closed,
}

FC_metrics_open = {
    'fronto-occipital connectivity': br_of_open,
    'interhemispheric connectivity': br_ih_open,
    'MST diameter': br_mst_d_open,
    'MST leaf fraction': br_mst_lf_open,
}

titles = list(FC_metrics_closed.keys())

fig1, axs1 = plt.subplots(2,2, figsize=(9,9))
for i, ax in enumerate(axs1.flatten()):
    x = np.array(PSI_scores)
    y = np.array(FC_metrics_closed[titles[i]])
    r, p = spearmanr(x, y)
    for age, color in age_categories.items():
        mask = ages == age
        ax.scatter(x[mask], y[mask], label=age, color=color)
    coefficients = np.polyfit(x,y,1)
    trendline = np.poly1d(coefficients)
    ax.plot(x, trendline(x), color='gray', alpha=0.5, label='Trendline')
    ax.set_title(fr'$\rho$ = {r:.3f}, p = {p:.3f}')
    ax.set_xlabel('PSI score')
    ax.set_ylabel(f'{titles[i]}')
    ax.legend(title='Age')
fig1.suptitle('Eyes closed condition')
plt.show()

fig2, axs2 = plt.subplots(2,2, figsize=(9,9))
for i, ax in enumerate(axs2.flatten()):
    x = np.array(PSI_scores)
    y = np.array(FC_metrics_open[titles[i]])
    r, p = spearmanr(x, y)
    for age, color in age_categories.items():
        mask = ages == age
        ax.scatter(x[mask], y[mask], label=age, color=color)
    coefficients = np.polyfit(x,y,1)
    trendline = np.poly1d(coefficients)
    ax.plot(x, trendline(x), color='gray', alpha=0.5, label='Trendline')
    ax.set_title(fr'$\rho$ = {r:.3f}, p = {p:.3f}')
    ax.set_xlabel('PSI score')
    ax.set_ylabel(f'{titles[i]}')
    ax.legend(title='Age')
fig2.suptitle('Eyes open condition')
plt.show()

#%% 
# 3. correlation iAPF 
plt.figure(figsize=(7,7))
x = np.array(PSI_scores)
y = np.array(clinical_iapf_assessment)
r, p = spearmanr(x, y)
for age, color in age_categories.items():
    mask = ages == age
    plt.scatter(x[mask], y[mask], label=age, color=color)
coefficients = np.polyfit(x,y,1)
trendline = np.poly1d(coefficients)
#plt.plot(x, trendline(x), color='gray', alpha=0.5, label='Trendline') #optionally
plt.title(fr'$\rho$ = {r:.3f}, p = {p:.3f}')
plt.xlabel('PSI score')
plt.ylabel('Individual alpha peak frequency (iAPF)')
plt.legend(title='Age')
plt.show()
