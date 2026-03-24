import pandas as pd
import numpy as np


#this will load that dataset
df = pd.read_csv("data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip() #stripping away the extra space before the text in the headers

# Ensure numeric columns
numeric_cols = [
    'income_annum',
    'loan_amount',
    'loan_term',
    'fico_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]
df['income_annum'] = (
    df['income_annum']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['loan_status'] = (
    df['loan_status']
    .astype(str)
    .str.strip()
    .str.title()
)

df['self_employed'] = (
    df['self_employed']
    .astype(str)
    .str.strip()
    .str.title()
)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values in key columns
df_clean = df.dropna(subset=['income_annum'])
print()
#finding the number of loans approved or rejected
counts = df_clean['loan_status'].value_counts()
print(f"Total loans approved or rejected: ")
approved = counts.get('Approved', 0)
rejected = counts.get('Rejected', 0)
print(f"Approved: {approved}")
print(f"Rejected: {rejected}")
print(f"Percentage of loans approved: {((approved / (approved + rejected)) * 100):.2f}%")

#question #1 -  Are self‑employed applicants approved at lower rates than salaried applicants?
print()
print("QUESTION 1: Are self‑employed applicants approved at lower rates than salaried applicants?")
 #find self-employed - what % are approved
self_approved = len(df_clean[
    (df_clean['self_employed'] == 'Yes') &
    (df_clean['loan_status'] == 'Approved')
])

self_denied = len(df_clean[
    (df_clean['self_employed'] == 'Yes') &
    (df_clean['loan_status'] == 'Rejected')
])

self_employed = (self_approved / (self_approved + self_denied)) * 100
print(f"The percentage of self_employed approved loans is {self_employed:.2f}%.")
 #find salaried - what % are approved
not_self_approved = len(df_clean[
    (df_clean['self_employed'] == 'No') &
    (df_clean['loan_status'] == 'Approved')
])

not_self_denied = len(df_clean[
    (df_clean['self_employed'] == 'No') &
    (df_clean['loan_status'] == 'Rejected')
])

not_self_employed = (not_self_approved / (not_self_approved + not_self_denied)) * 100
print(f"The percentage of NOT self_employed approved loans is {not_self_employed:.2f}%.")


#question #2 - What percentage of approved loans go to applicants with below‑median income?
print()
print("QUESTION 2: What percentage of approved loans go to applicants with below‑median income?")

# find the median income
median_income = df_clean['income_annum'].median()
print(f"The median income is ${median_income:.2f}.")
# A = count of approved loans below median income
count_a = len(df_clean[
    (df_clean['income_annum'] <= median_income) &
    (df_clean['loan_status'] == 'Approved')
])
print(f"Number of loans approved below the Median Income: {count_a}")
# B = count of approved loans above median income
count_b = len(df_clean[
    (df_clean['income_annum'] > median_income) &
    (df_clean['loan_status'] == 'Approved')
])
print(f"Number of loans approved above the Median Income: {count_b}")

#percentage = A / (A+B)
percentage = (count_a / (count_a + count_b)) * 100
print(f"The percentage of loans to below-median income is {percentage:.2f}%.")
print()

#question #3 - Given a borrower’s annual income, requested loan amount, and FICO score,
# what is the estimated probability that their loan will be approved?
income = float(input("What is your annual income? "))
loan = float(input("What is your loan amount? "))
fico = float(input("What is your FICO score? "))
years = float(input("How many years to you want the loan? "))

#figure out percentage of income for the loan
#total income through life of the loan
income_percentage = (loan / (income * years)) * 100
print(f"The percentage of your income to service the loan is {income_percentage:.2f}%.")
monthly = (income_percentage * income) / 12 / 100
print(f"Your monthly payment on principle is: ${monthly:.2f}.")