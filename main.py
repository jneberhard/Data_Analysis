import pandas as pd
import numpy as np


#this will load that dataset ==  df means data or dataframe == conventional shorthand in python
df = pd.read_csv("data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip() #stripping away the extra space before the text in the headers

# Ensure numeric columns
numeric_cols = [
    'no_of_dependents',
    'income_annum',
    'loan_amount',
    'loan_term',
    'fico_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]

#clean the data in each column so it is uniform
df['income_annum'] = (
    df['income_annum']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['loan_term'] = (
    df['loan_term']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['fico_score'] = (
    df['fico_score']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['loan_amount'] = (
    df['loan_amount']
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

df['no_of_dependents'] = (
    df['no_of_dependents']
    .astype(str)
    .str.strip()
)

df['self_employed'] = (
    df['self_employed']
    .astype(str)
    .str.strip()
    .str.title()
)

df['residential_assets_value'] = (
    df['residential_assets_value']
    .astype(str)
    .str.strip()
    .str.title()
)

df['commercial_assets_value'] = (
    df['commercial_assets_value']
    .astype(str)
    .str.strip()
    .str.title()
)

df['luxury_assets_value'] = (
    df['luxury_assets_value']
    .astype(str)
    .str.strip()
    .str.title()
)

df['bank_asset_value'] = (
    df['bank_asset_value']
    .astype(str)
    .str.strip()
    .str.title()
)



#convert numeric columns from str values to numbers
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') # use coerce to convert to NaN if string can't be converted to a number

# Drop rows with missing values in key columns
df_clean = df.dropna(subset=['income_annum'])
print()
#finding the number of loans approved or rejected
counts = df_clean['loan_status'].value_counts()
print("Total loans approved or rejected: ")
approved = counts.get('Approved', 0)
rejected = counts.get('Rejected', 0)
print(f"Approved: {approved}")
print(f"Rejected: {rejected}")
print(f"Percentage of loans approved: {((approved / (approved + rejected)) * 100):.2f}%")

########################################################################################################
#question #1 -  Are self‑employed applicants approved at lower rates than salaried applicants?
########################################################################################################

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

########################################################################################################
#question #2 - What percentage of approved loans go to applicants with below‑median income?
########################################################################################################
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

########################################################################################################
#question #3 - Given a borrower’s annual income, requested loan amount, FICO score, and assets,
# what is the probability of getting a loan?
########################################################################################################

#convert loan status to numeric for modeling
df_clean['loan_status'] = df_clean['loan_status'].map({
'Approved': 1,
'Rejected': 0
})

#make income and loan ration matter
df_clean['loan_to_income_ratio'] = df_clean['loan_amount'] / df_clean['income_annum']
df_clean['high_loan_risk'] = (df_clean['loan_to_income_ratio'] > 4).astype(int)   # > 4 because that is about what real life situations are

# make assets matter
df_clean['total_assets'] = df_clean['luxury_assets_value'] + df_clean['bank_asset_value'] + df_clean['residential_assets_value'] + df_clean['commercial_assets_value']

#omit the outliers
df_clean['log_income'] = np.log1p(df_clean['income_annum'])
df_clean['log_assets'] = np.log1p(df_clean['total_assets'])

df_clean['loan_to_asset_ratio'] = df_clean['loan_amount'] / (df_clean['total_assets'] + 1)

#create the data model
feature_cols = ['loan_to_income_ratio', 'fico_score', 'loan_term', 'no_of_dependents', 'log_income', 'log_assets', 'loan_to_asset_ratio']
X = df_clean[feature_cols].values  #the features
y = df_clean['loan_status'].values   #the target

#normalize/standardization
means = X.mean(axis=0)   #calculates average for each column.
standards = X.std(axis=0)  #standard deviation - how spread out the numbers are for each item
standards[standards == 0] = 1
X_scaled = (X - means) / standards  #subtracts the mean and divides by the standard deviation - standarization formula

X_final = np.c_[np.ones(X_scaled.shape[0]), X_scaled] # adding the bios or intercept column


#Make part of the dataset training and testing
split_index = int(len(X_final) * 0.8)

X_train = X_final[:split_index]
X_test = X_final[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# sigmoid calculation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# compute weights - how much each feature influences the prediction
def train_logistic(X, y, lr=0.1, epochs=1000):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        preds = sigmoid(np.dot(X, weights))
        error = preds - y
        gradient = np.dot(X.T, error) / len(y)
        weights -= lr * gradient
    return weights

weights = train_logistic(X_train, y_train)

cont = True

#keep running the calculations until the user is done running calculations
while cont:
    # what is the estimated probability that their loan will be approved? User input
    income = float(input("What is your annual income? "))
    loan = float(input("What is your loan amount? "))
    fico = float(input("What is your FICO score? "))
    years = float(input("How many years to you want the loan? "))
    dependents = float(input("How many dependents do you have? "))
    asset = float(input("What is the value of your assets? "))
    cash = float(input("How much cash on hand? "))

    log_income = np.log1p(income)
    log_assets = np.log1p(asset + cash)

    loan_to_income_ratio = loan / (income + 1)
    loan_to_asset_ratio = loan / (asset + cash + 1)

    #normalize the input (just like the learning model was normailzed)
    user_raw = np.array([loan_to_income_ratio, fico, years, dependents, log_income, log_assets, loan_to_asset_ratio])   #make sure same order as training model
    user_scaled = (user_raw - means) / standards
    user_final = np.insert(user_scaled, 0, 1)

    #prediction model
    z = np.dot(user_final, weights)
    probability = sigmoid(z)

    #figure monthly payment
    monthly = loan / (years * 12)

    #figure out percentage of income for the loan
    monthly_income = income / 12
    income_percentage = (monthly / monthly_income) * 100

    #risk assesment
    if loan_to_income_ratio < 2:
        risk = "LOW"
    elif loan_to_income_ratio < 4:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    adjusted = probability

    # FICO penalties - will lower the % the lower credit you have
    if fico < 580:
        adjusted *= 0.3   # very poor credit
    elif fico < 620:
        adjusted *= 0.5   # poor credit
    elif fico < 670:
        adjusted *= 0.75  # fair credit

    # Loan-to-income penalties
    if loan_to_income_ratio > 8:
        adjusted *= 0.3
    elif loan_to_income_ratio > 6:
        adjusted *= 0.5
    elif loan_to_income_ratio > 4:
        adjusted *= 0.7

    # Debt-to-income penalty (standard bank threshold is 43%)
    if income_percentage > 43:
        adjusted *= 0.6

    # Cap it
    adjusted = min(adjusted, 0.97)

    #printing results
    print("Results:")
    print(f"- Risk Assessment: {risk}")
    print(f"- Approval Probability: {adjusted * 100:.1f}%")
    print()
    print(f"The percentage of your income to service the loan is {income_percentage:.2f}%.")
    print(f"Your monthly payment on principle is: ${monthly:.2f}.")
    print()
    again = input("Do you want to do another calculation? (Y/N)").strip().lower()
    if again != "y":
        cont = False
print()
print("Thank you.  Have a great day!!!!")
print()
