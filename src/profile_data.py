import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Basic information
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Column names
print("\n========== COLUMNS ==========")
print(df.columns.tolist())

# 4. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 5. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Duplicate rows
print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

# 7. Fraud distribution
print("\n========== FRAUD DISTRIBUTION ==========")
print(df["Class"].value_counts())

# 8. Fraud percentage
print("\n========== FRAUD PERCENTAGE ==========")
print(df["Class"].value_counts(normalize=True) * 100)

# 9. Amount statistics
print("\n========== AMOUNT STATISTICS ==========")
print(df["Amount"].describe())

# 10. First 5 rows
print("\n========== SAMPLE DATA ==========")
print(df.head())