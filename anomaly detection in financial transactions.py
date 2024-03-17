import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

#  Import  dataset
df = pd.read_csv('/content/transaction_anomalies_dataset.csv')

#   null values
null_values = df.isnull().sum()

#  column information
column_info = df.info()

statistics = df.describe()

# Plot distribution of Transaction Amount
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.histplot(df['Transaction_Amount'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

# Transaction Amount by Account Type
plt.subplot(2, 2, 2)
sns.boxplot(x='Account_Type', y='Transaction_Amount', data=df)
plt.title('Distribution of Transaction Amount by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Transaction Amount')



# average transaction by age
average_transaction_by_age = df.groupby('Age')['Transaction_Amount'].mean()

plt.subplot(2, 2, 3)
plt.scatter(average_transaction_by_age.index, average_transaction_by_age.values, color='skyblue')
plt.title('Average Transaction Amount by Age')
plt.xlabel('Age')
plt.ylabel('Average Transaction Amount')
plt.grid(True)

# transactions by day of the week
plt.subplot(2, 2, 4)
df['Day_of_Week'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Count of Transactions by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count of Transactions')
plt.xticks(rotation=45)

correlation_matrix = df.corr()

# Plot the  heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of All Columns')
plt.tight_layout()
plt.show()

# Show 
plt.tight_layout()
plt.show()

import numpy as np
np.random.seed(0)
df['True_Label'] = np.random.randint(0, 2, size=len(df))

#  Detect anomalies using Isolation Forest
model = IsolationForest(contamination=0.1)
model.fit(df[['Transaction_Amount', 'Transaction_Volume']])
df['Anomaly'] = model.predict(df[['Transaction_Amount', 'Transaction_Volume']])

#Convert predictions into binary values and visualize anomalies
df['Anomaly'] = df['Anomaly'].replace({-1: 0, 1: 1})
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['Transaction_Amount'], c=df['Anomaly'], cmap='coolwarm')
plt.title('Anamolies Detected by Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Transaction Amount')
plt.colorbar(label='Anomaly (0: Anomaly, 1: Normal)')

# classification report
y_true = df['True_Label']
y_pred = df['Anomaly']
report = classification_report(y_true, y_pred)


print("Clasification Report:")
print(report)


def detect_anomaly(transaction_amount, avg_transaction_amount, frequency_of_transactions):
    
    new_data = pd.DataFrame({
        'Transaction_Amount': [transaction_amount],
        'Transaction_Volume': [1], 
    })

   
    new_data['Anomaly'] = model.predict(new_data[['Transaction_Amount', 'Transaction_Volume']])
    new_data['Anomaly'] = new_data['Anomaly'].replace({-1: 'Anomaly', 1: 'Normal'}) 

    return new_data['Anomaly'].iloc[0]  


amount = float(input("Enter the transaction amount: "))
avg_amount = float(input("Enter the average transaction amount: "))
frequency = int(input("Enter the frequency of transactions: "))


new_data = pd.DataFrame({
    'Transaction_Amount': [amount],
    'Transaction_Volume': [1],  
})
new_data['Anomaly'] = model.predict(new_data[['Transaction_Amount', 'Transaction_Volume']])
new_data['Anomaly'] = new_data['Anomaly'].replace({-1: 'Anomaly', 1: 'Normal'})  

# anomaly detection result
print("Anomaly Detection Result for Manually Entered Data:")
print(new_data[['Transaction_Amount', 'Anomaly']])

