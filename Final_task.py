#%%imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#%%load data
df = pd.read_csv("Employment_Verification_Risk_Data.csv")
df.head()
#%% MONTHLY INCOME
# Create four income buckets with equal distribution
df['Income_Bucket'] = pd.qcut(df['monthly_income'], q=4)

# Calculate risk for each income bucket
risk_dict = {}

for bucket in df['Income_Bucket'].unique():
    bucket_customers = df[df['Income_Bucket'] == bucket]
    non_payment_count = bucket_customers[bucket_customers['Emi_Paid'] == 0]['id'].count()
    risk = (non_payment_count / len(bucket_customers))*100
    risk_dict[str(bucket)] = risk

# Create a DataFrame to store the risk values
risk_df = pd.DataFrame.from_dict(risk_dict, orient='index', columns=['Risk'])
risk_df.index.name = 'Income_Bucket'

# Sort the DataFrame by risk in descending order
risk_df = risk_df.sort_values(by='Risk', ascending=False)
#%% INDUSTRY
industry_risk = df.groupby('Industry')['Emi_Paid'].apply(lambda x: ((x == 0).sum() / len(x)) * 100).to_frame(name='Industry Risk')
industry_risk = industry_risk.sort_values(by = 'Industry Risk', ascending=False)
print(industry_risk)

#%% SCORE
# Create four equal-sized buckets based on score
score_buckets = pd.qcut(df['score'], q=4)

# Calculate risk for each bucket
score_risk = df.groupby(score_buckets)['Emi_Paid'].apply(lambda x: (x == 0).sum() / len(x) * 100).to_frame(name='Score Risk')

# Sort the risk percentages in descending order
score_risk = score_risk.sort_values(by = 'Score Risk',ascending=False)

print(score_risk)

#%% PENTILE
pentile_risk = df.groupby('pentile')['Emi_Paid'].apply(lambda x: ((x == 0).sum() / len(x)) * 100).to_frame(name='Pentile Risk')
pentile_risk = pentile_risk.sort_values(by = 'Pentile Risk', ascending=False)
print(pentile_risk)

pentile_counts = df['pentile'].value_counts().to_frame(name='Number of Customers').sort_index()
print(pentile_counts)

#%%STRATEGY
strategy_risk = df.groupby('Strategy').apply(lambda x: (x.loc[x['Emi_Paid'] == 0, 'loan_amount'].sum() / x['loan_amount'].sum()) * 100)
strategy_risk = strategy_risk.sort_values(ascending=False)
print(strategy_risk)
#%%BIVARIATE ANALYSIS

# Create four buckets for 'number_of_loans' column
loans_bucket_edges = pd.qcut(df['number_of_loans'], q=[0, 0.25, 0.5, 0.75, 1], duplicates='drop', labels=False)
df['number_of_loans_bucket'] = pd.cut(df['number_of_loans'], bins=len(loans_bucket_edges.unique()), labels=False)

# Create four buckets for 'loans_above_1_lakh' column
amount_bucket_edges = pd.qcut(df['loans_above_1_lakh'], q=[0, 0.25, 0.5, 0.75, 1], duplicates='drop', labels=False)
df['loans_above_1_lakh_bucket'] = pd.cut(df['loans_above_1_lakh'], bins=len(amount_bucket_edges.unique()), labels=False)

# Print the quartile values
print("Quartile Values:")
print("number_of_loans:")
print(df.groupby('number_of_loans_bucket')['number_of_loans'].agg(['min', 'max']))
print("loans_above_1_lakh:")
print(df.groupby('loans_above_1_lakh_bucket')['loans_above_1_lakh'].agg(['min', 'max']))

# Calculate risk for each combination of buckets
risk_matrix = pd.pivot_table(df, index='number_of_loans_bucket', columns='loans_above_1_lakh_bucket',
                             values='Emi_Paid', aggfunc=lambda x: (x == 0).mean() * 100)

# Display the correlation matrix of risk values
print("Risk Correlation Matrix:")
print(risk_matrix)

# Calculate the number of people in each quartile
number_of_loans_counts = df['number_of_loans_bucket'].value_counts().sort_index()
loans_above_1_lakh_counts = df['loans_above_1_lakh_bucket'].value_counts().sort_index()

# Print the number of people in each quartile
print("Number of People in Each Quartile:")
print("number_of_loans:")
print(number_of_loans_counts)
print("loans_above_1_lakh:")
print(loans_above_1_lakh_counts)

#%%
# Calculate the number of people in each risk matrix cell
risk_matrix_counts = pd.pivot_table(df, index='number_of_loans_bucket', columns='loans_above_1_lakh_bucket',
                                    values='Emi_Paid', aggfunc='count')

# Print the number of people in each risk matrix cell
print("Number of People in Each Risk Matrix Cell:")
print(risk_matrix_counts)
#%% AW Risk - Pentile
amount_weighted_risk = df.groupby('pentile').apply(lambda x: x.loc[x['Emi_Paid'] == 0, 'loan_amount'].sum() / x['loan_amount'].sum())
amount_weighted_risk = amount_weighted_risk.rename('Amount Weighted Risk')
amount_weighted_risk = amount_weighted_risk.sort_values(ascending=False)
print(amount_weighted_risk)

#%%
# Filter data for employment verified segment
EMPLOYMENT_PROOF = df[df['Strategy'] == 'EMPLOYMENT_PROOF']

# Filter data for no employment verification segment
NO_EMPLOYMENT_PROOF_REQUIRED = df[df['Strategy'] == 'NO_EMPLOYMENT_PROOF_REQUIRED']

# Calculate the amount weighted risk for each company partner within the employment verified segment
employment_proof_risk = EMPLOYMENT_PROOF.groupby('App_Name').apply(lambda x: (x.loc[x['Emi_Paid'] == 0, 'loan_amount'].sum() / x['loan_amount'].sum()) * 100)
employment_proof_risk = employment_proof_risk.sort_values( ascending=False)

# Calculate the amount weighted risk for each company partner within the no employment verification segment
no_employment_proof_risk = NO_EMPLOYMENT_PROOF_REQUIRED.groupby('App_Name').apply(lambda x: (x.loc[x['Emi_Paid'] == 0, 'loan_amount'].sum() / x['loan_amount'].sum()) * 100)
no_employment_proof_risk = no_employment_proof_risk.sort_values(ascending=False)

#%% strategy wise risk
# Calculate the amount weighted risk for the employment proof required segment
employment_proof_required_risk = (EMPLOYMENT_PROOF['Emi_Paid'] == 0).sum() / EMPLOYMENT_PROOF['loan_amount'].sum()

# Calculate the amount weighted risk for the employment proof not required segment
employment_proof_not_required_risk = (NO_EMPLOYMENT_PROOF_REQUIRED['Emi_Paid'] == 0).sum() / NO_EMPLOYMENT_PROOF_REQUIRED['loan_amount'].sum()

#%%SEGMENTATION
# Calculate the percentage of applications in the employment proof flow
employment_proof_flow_percentage = (df['Strategy'] == 'NO_EMPLOYMENT_PROOF_REQUIRED').mean() * 100

print(f"Percentage of applications in the no employment proof flow: {employment_proof_flow_percentage:.2f}%")
#%%
# Calculate the percentage of people under each app for the employment verification flow
employment_verification_percentage = (EMPLOYMENT_PROOF.groupby('App_Name').size() / len(EMPLOYMENT_PROOF)) * 100

print("Percentage of people under each app for the employment verification flow:")
print(employment_verification_percentage)
#%%
amount_weighted_risk = EMPLOYMENT_PROOF.groupby('pentile').apply(lambda x: (x.loc[x['Emi_Paid'] == 0, 'loan_amount'].sum() / x['loan_amount'].sum()) * 100)
amount_weighted_risk = amount_weighted_risk.rename('Amount Weighted Risk')
amount_weighted_risk = amount_weighted_risk.sort_values(ascending=False)
print(amount_weighted_risk)
#%%
pentile_counts = EMPLOYMENT_PROOF['pentile'].value_counts()
pentile_percentages = pentile_counts / pentile_counts.sum() * 100
print("Percentage of Applications in Each Pentile (Employment Proof Segment):")
print(pentile_percentages)
#%%
EMPLOYMENT_PROOF['income_bucket'] = pd.qcut(EMPLOYMENT_PROOF['monthly_income'], q=4, duplicates='drop')
amount_weighted_risk_income = EMPLOYMENT_PROOF.groupby('income_bucket').apply(lambda x: (x.loc[x['Emi_Paid'] == 0, 'loan_amount'].sum() / x['loan_amount'].sum()) * 100)
amount_weighted_risk_income = amount_weighted_risk_income.rename('Amount Weighted Risk')
amount_weighted_risk_income = amount_weighted_risk_income.sort_values(ascending=False)
#%% app name and pentile
employment_proof_data = df[df['Strategy'] == 'EMPLOYMENT_PROOF']

# Create a contingency table
contingency_table = pd.crosstab(employment_proof_data['App_Name'], employment_proof_data['pentile'], normalize='all') * 100
pd.set_option('display.max_columns', None)
# Print the contingency table
print(contingency_table)

# Calculate the amount-weighted risk for each combination of App_Name and pentile
amount_weighted_risk = employment_proof_data.groupby(['App_Name', 'pentile']).apply(lambda x: (x.loc[x['Emi_Paid'] == 0, 'loan_amount'].sum() / x['loan_amount'].sum()) * 100)
amount_weighted_risk = amount_weighted_risk.rename('Amount Weighted Risk')

# Create a contingency table with amount-weighted risk values
contingency_table1 = pd.pivot_table(amount_weighted_risk.reset_index(), index='App_Name', columns='pentile', values='Amount Weighted Risk')

# Print the contingency table
print(contingency_table1)
#%%BIVARIAE ANALYSIS: app name and credit score
# Specify the number of bins or range size
num_bins = 6

credit_score_bins = pd.cut(EMPLOYMENT_PROOF['score'], bins=num_bins)
grouped_data = EMPLOYMENT_PROOF.groupby(['App_Name', credit_score_bins])
percentage_of_customers = grouped_data.size() / len(EMPLOYMENT_PROOF) * 100
amount_weighted_risk = grouped_data.apply(lambda x: (x['loan_amount'] * (x['Emi_Paid'] == 0)).sum() / x['loan_amount'].sum() * 100)

bivariate_matrix = pd.pivot_table(data=EMPLOYMENT_PROOF, values='score', index='App_Name', columns=credit_score_bins, aggfunc=len, fill_value=0)
bivariate_matrix_percentage = bivariate_matrix.div(len(EMPLOYMENT_PROOF)) * 100
bivariate_matrix_percentage.replace(0, np.nan, inplace=True)
risk_matrix = pd.pivot_table(data=EMPLOYMENT_PROOF, values='risk', index='App_Name', columns=credit_score_bins, aggfunc=np.mean, fill_value=0)

# Print the bivariate analysis matrix
print("Bivariate Analysis - Percentage of Customers:")
print(bivariate_matrix_percentage)

# Print the risk matrix
print("Risk Matrix:")
print(risk_matrix)
#%% WITHOUT XX
filtered_df = EMPLOYMENT_PROOF[EMPLOYMENT_PROOF['App_Name'] != 'XX']

#%% decision tree
filtered_df['Target'] = filtered_df.apply(lambda row: 'Low Risk' if (row['pentile'] in [1, 2] and row['score'] > 698) else 'High Risk', axis=1)

# Create the decision tree classifier with custom split criteria
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)

# Prepare the features and target variables
X = filtered_df[['pentile', 'score']]
y = filtered_df['Target']

# Define the split criteria for the decision tree
root_split = filtered_df['pentile'].apply(lambda x: 'Left' if x in [1, 2] else 'Right')
left_split = filtered_df['score'].apply(lambda x: 'Left' if x > 698 else 'Right')

# Build the decision tree structure
tree_structure = {
    0: {'left': '1', 'right': '2'},
    '1': {'left': '3', 'right': '4'},
    '2': 'High Risk',
    '3': 'Low Risk',
    '4': 'High Risk'
}

# Assign the decision tree structure to the classifier
clf.tree_ = tree_structure

# Visualize the decision tree
fig, ax = plt.subplots(figsize=(8, 8))
tree.plot_tree(clf, filled=True, ax=ax)

# Display the decision tree
plt.show()
#%% Create the decision tree model with the specified splits
clf = DecisionTreeClassifier(random_state=0)
clf.fit(filtered_df[['pentile', 'score']], filtered_df['Target'])

# Plot the decision tree
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf, feature_names=['Pentile', 'Score'], class_names=['Low Risk', 'High Risk'], filled=True, ax=ax)
plt.show()
#%%
filtered_df['Target'] = filtered_df.apply(lambda row: 'Low Risk' if (row['pentile'] in [1, 2] and row['score'] > 760) else 'High Risk', axis=1)

# Create the decision tree classifier with custom split criteria
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)

# Prepare the features and target variables
X = filtered_df[['pentile', 'score']]
y = filtered_df['Target']

# Define the split criteria for the decision tree
root_split = filtered_df['pentile'].apply(lambda x: 'Left' if x in [1, 2] else 'Right')
left_split = filtered_df['score'].apply(lambda x: 'Left' if x > 698 else 'Right')

# Fit the decision tree classifier
clf.fit(X, y)

# Assign the class labels to the leaf nodes
clf.classes_ = ['Low Risk', 'High Risk', 'High Risk']

# Plot the decision tree
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf, feature_names=['Pentile', 'Score'], class_names=['High Risk', 'Low Risk'], filled=True, ax=ax)
plt.show()

