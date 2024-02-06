#%%imports
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
#%%load data
df = pd.read_csv("Funnel_Base_Data.csv")
#df.head()
#%% Different dfs for each category
category_A_df = df[df['Category'] == 'CAT_A']
category_B_df = df[df['Category'] == 'CAT_B']
category_C_df = df[df['Category'] == 'CAT_C']
category_D_df = df[df['Category'] == 'CAT_D']
#%%
'''Create different dataframes for each category of the customers for easier analysis. 
For each category, analyse the entire funnel to identify the major bottlenecks and then look for possible reasons and solutions.'''
category_stats = pd.DataFrame(columns=['Category', 'Starts', 'Eligible', 'Applied', 'Submits', 'Fraud', 'Nach', 'LA', 'Disbursed', 'Rejected'])

category_stats.loc[0] = ['CAT_A',
                         category_A_df['Starts'].sum(),
                         category_A_df['Eligible'].sum(),
                         category_A_df['Applied'].sum(),
                         category_A_df['Submits'].sum(),
                         category_A_df['Fraud'].sum(),
                         category_A_df['Nach'].sum(),
                         category_A_df['LA'].sum(),
                         category_A_df['Disbursed'].sum(),
                         category_A_df['Rejected'].sum()]

category_stats.loc[1] = ['CAT_B',
                         category_B_df['Starts'].sum(),
                         category_B_df['Eligible'].sum(),
                         category_B_df['Applied'].sum(),
                         category_B_df['Submits'].sum(),
                         category_B_df['Fraud'].sum(),
                         category_B_df['Nach'].sum(),
                         category_B_df['LA'].sum(),
                         category_B_df['Disbursed'].sum(),
                         category_B_df['Rejected'].sum()]

category_stats.loc[2] = ['CAT_C',
                         category_C_df['Starts'].sum(),
                         category_C_df['Eligible'].sum(),
                         category_C_df['Applied'].sum(),
                         category_C_df['Submits'].sum(),
                         category_C_df['Fraud'].sum(),
                         category_C_df['Nach'].sum(),
                         category_C_df['LA'].sum(),
                         category_C_df['Disbursed'].sum(),
                         category_C_df['Rejected'].sum()]

category_stats.loc[3] = ['CAT_D',
                         category_D_df['Starts'].sum(),
                         category_D_df['Eligible'].sum(),
                         category_D_df['Applied'].sum(),
                         category_D_df['Submits'].sum(),
                         category_D_df['Fraud'].sum(),
                         category_D_df['Nach'].sum(),
                         category_D_df['LA'].sum(),
                         category_D_df['Disbursed'].sum(),
                         category_D_df['Rejected'].sum()]
#%%
category_stats['Start_to_Eligible'] = (category_stats['Eligible'] / category_stats['Starts']) * 100
category_stats['Eligible_to_Submit'] = (category_stats['Submits'] / category_stats['Eligible']) * 100
category_stats['Eligible_to_Applied'] = (category_stats['Applied'] / category_stats['Eligible']) * 100
category_stats['Applied_to_Submit'] = (category_stats['Submits'] / category_stats['Applied']) * 100
category_stats['Submit_to_Disbursed'] = (category_stats['Disbursed'] / category_stats['Submits']) * 100
category_stats['Start_to_Submit'] = (category_stats['Submits'] / category_stats['Starts']) * 100
category_stats['Start_to_Disbursed'] = (category_stats['Disbursed'] / category_stats['Starts']) * 100
category_stats['Start_to_Fraud'] = (category_stats['Fraud'] / category_stats['Starts']) * 100
category_stats['Start_to_LA'] = (category_stats['LA'] / category_stats['Starts']) * 100
category_stats['Start_to_Nach'] = (category_stats['Nach'] / category_stats['Starts']) * 100
category_stats['Submit_to_Nach'] = (category_stats['Nach'] / category_stats['Submits']) * 100
category_stats['Nach_to_LA'] = (category_stats['LA'] / category_stats['Nach']) * 100
category_stats['LA_to_Disbursed'] = (category_stats['Disbursed'] / category_stats['LA']) * 100
category_stats['Start_to_Rejected'] = (category_stats['Rejected'] / category_stats['Starts']) * 100
category_stats['Eligible_to_Rejected'] = (category_stats['Rejected'] / category_stats['Eligible']) * 100
category_stats['Submit_to_Rejected'] = (category_stats['Rejected'] / category_stats['Submits']) * 100
category_stats['Applied_to_Rejected'] = (category_stats['Rejected'] / category_stats['Applied']) * 100
#%%
category_stats_transposed = category_stats.transpose()
category_stats_transposed.to_excel('category_stats_transposed.xlsx', index=True)
#%% Applied to reject etc.
'''
submit_customers = df[df['Applied'] == 1]
category_A_eli = submit_customers[submit_customers['Category'] == 'CAT_A']
category_B_eli = submit_customers[submit_customers['Category'] == 'CAT_B']
category_C_eli = submit_customers[submit_customers['Category'] == 'CAT_C']
category_D_eli = submit_customers[submit_customers['Category'] == 'CAT_D']

category_stats_eli = pd.DataFrame(columns=['Category', 'Starts', 'Eligible', 'Applied', 'Submits', 'Fraud', 'Nach', 'LA', 'Disbursed', 'Rejected'])
category_stats_eli.loc[0] = ['CAT_A',
                         category_A_eli['Starts'].sum(),
                         category_A_eli['Eligible'].sum(),
                         category_A_eli['Applied'].sum(),
                         category_A_eli['Submits'].sum(),
                         category_A_eli['Fraud'].sum(),
                         category_A_eli['Nach'].sum(),
                         category_A_eli['LA'].sum(),
                         category_A_eli['Disbursed'].sum(),
                         category_A_eli['Rejected'].sum()]

category_stats_eli.loc[1] = ['CAT_B',
                         category_B_eli['Starts'].sum(),
                         category_B_eli['Eligible'].sum(),
                         category_B_eli['Applied'].sum(),
                         category_B_eli['Submits'].sum(),
                         category_B_eli['Fraud'].sum(),
                         category_B_eli['Nach'].sum(),
                         category_B_eli['LA'].sum(),
                         category_B_eli['Disbursed'].sum(),
                         category_B_eli['Rejected'].sum()]

category_stats_eli.loc[2] = ['CAT_C',
                         category_C_eli['Starts'].sum(),
                         category_C_eli['Eligible'].sum(),
                         category_C_eli['Applied'].sum(),
                         category_C_eli['Submits'].sum(),
                         category_C_eli['Fraud'].sum(),
                         category_C_eli['Nach'].sum(),
                         category_C_eli['LA'].sum(),
                         category_C_eli['Disbursed'].sum(),
                         category_C_eli['Rejected'].sum()]

category_stats_eli.loc[3] = ['CAT_D',
                         category_D_eli['Starts'].sum(),
                         category_D_eli['Eligible'].sum(),
                         category_D_eli['Applied'].sum(),
                         category_D_eli['Submits'].sum(),
                         category_D_eli['Fraud'].sum(),
                         category_D_eli['Nach'].sum(),
                         category_D_eli['LA'].sum(),
                         category_D_eli['Disbursed'].sum(),
                         category_D_eli['Rejected'].sum()]

category_stats_eli['Eligible_to_Rejected'] = (category_stats_eli['Rejected'] / category_stats_eli['Applied']) * 100
'''
#%%
average_loan_amount = df.groupby('Category')['Disbursed_Amt'].mean()
