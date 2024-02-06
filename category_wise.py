#%%imports
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
#%%load data
df = pd.read_csv("Category_wise_Funnel.csv")
df.head()
#%%
# Group the data by strategy and calculate loan application outcomes
grouped = df.groupby('Strategy')['application_status'].value_counts().unstack().fillna(0)

# Calculate the total number of applications for each strategy
grouped['Total'] = grouped.sum(axis=1)

# Calculate the approval rates for each strategy
grouped['Rejection Rate'] = (grouped['REJECTED'] / grouped['Total']) * 100

# Print the results
print(grouped)
#%%category wise
# Group the data by strategy and calculate loan application outcomes
cats = df.groupby('Strategy')['Category'].value_counts().unstack().fillna(0)

# Calculate the total number of applications for each strategy
cats['Total'] = cats.sum(axis=1)

# Calculate the total number of applications for each category
cats.loc['Total'] = cats.sum(axis=0)

#%%
# Count non-null values in disbursal_date column for each strategy
non_null_counts = df.groupby('Strategy')['disbursal_date'].count()

# Calculate total count of rows in each strategy group
total_counts = df.groupby('Strategy')['disbursal_date'].size()

# Calculate the percentage of non-null values
percent_non_null = (non_null_counts / total_counts) * 100

# Print the results
print(percent_non_null)
#%% cat b
cat_b_customers = df[df['Category'] == 'CAT_B']
grouped_b = cat_b_customers.groupby('Strategy')['application_status'].value_counts().unstack().fillna(0)
grouped_b['Total'] = grouped_b.sum(axis=1)
grouped_b['Disbursal Rate'] = (grouped_b['DISBURSED'] / grouped_b['Total']) * 100
grouped_b['Rejection Rate'] = (grouped_b['REJECTED'] / grouped_b['Total']) * 100
grouped_b['% user cancelled'] = (grouped_b['USER_CANCELLED'] / grouped_b['Total']) * 100
#%% cat wise loan amt
avg_ticket_size = df.groupby('Category')['loan_amount'].mean()
#%% cat wise disbursal
# Group the data by category and calculate the required statistics
category_stats = df.groupby('Category').agg(
    disbursed_count=('application_status', lambda x: (x == 'DISBURSED').sum()),
    closed_count=('application_status', lambda x: (x == 'CLOSED').sum()),
    converted=('application_status', lambda x: ((x == 'DISBURSED') | (x == 'CLOSED')).sum()),
    total_applicants=('Category', 'count'))

# Calculate the conversion rate
category_stats['conversion_rate'] = (category_stats['converted'] / category_stats['total_applicants']) * 100
#%%loan amounts
# Filter the rows where the disbursal date is not null
filtered_df = df[df['disbursal_date'].notnull()]

# Calculate the total loan amount and average loan amount across categories
loan_amount_stats = filtered_df.groupby('Category')['loan_amount'].agg([
    ('total_loan_amount', 'sum'),
    ('average_loan_amount', 'mean')])

# Calculate the contribution of each category to the overall volume
total_loan_amount = loan_amount_stats['total_loan_amount'].sum()
loan_amount_stats['contribution'] = (loan_amount_stats['total_loan_amount'] / total_loan_amount) * 100

# Display the loan amount statistics
print(loan_amount_stats)
#%%
# Calculate category-wise count of non-null values in disbursal_date column
category_disbursal_counts = df.groupby('Category')['disbursal_date'].count()

# Display the results
print(category_disbursal_counts)
#%%
category_applied_counts = df[df['applied_amount'] > 0].groupby('Category')['id'].count().reset_index()
category_applied_counts.columns = ['Category', 'Number of Applications Applied']

print(category_applied_counts)
#%%
category_submission_counts = df[df['submission_date'].notnull()].groupby('Category')['id'].count().reset_index()
category_submission_counts.columns = ['Category', 'Number of People Submitted']

print(category_submission_counts)

