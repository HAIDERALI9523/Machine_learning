import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('/content/tips.csv')


#  null values
null_values = df.isnull().sum()
print('Null Values:')
print(null_values)

# column info
print('Info:')
print(df.info())

print('Statistics:')
print(df.describe())

df.columns = df.columns.str.strip()

df_subset = df[['total_bill', 'size', 'day']]

# scatter plot
fig = px.scatter(df_subset, x='total_bill', y='size', color='day', title='Total Bill vs. Number of People by Day')
fig.update_layout(xaxis_title='Total Bill ($)', yaxis_title='Number of People', legend_title='Day of Week')
fig.show()


# Report and the analysis
print("Report:")
print("1. Correlation Analysis:")
correlation_matrix = df_subset.corr()
print(correlation_matrix)

print("\n2. Average Total Bill by Day:")
avg_total_bill_by_day = df_subset.groupby('day')['total_bill'].mean().sort_values(ascending=False)
print(avg_total_bill_by_day)

print("\n3. Average Number of People by Day:")
avg_size_by_day = df_subset.groupby('day')['size'].mean().sort_values(ascending=False)
print(avg_size_by_day)

df.columns = df.columns.str.strip()


df_subset = df[['total_bill', 'size', 'sex']]


fig = px.scatter(df_subset, x='total_bill', y='size', color='sex', title='Total Bill vs. Number of People by Gender')
fig.update_layout(xaxis_title='Total Bill ($)', yaxis_title='Number of People', legend_title='Gender')
fig.show()

df.columns = df.columns.str.strip()


df_subset = df[['total_bill', 'size', 'time']]

fig = px.scatter(df_subset, x='total_bill', y='size', color='time', title='Total Bill vs. Number of People by Meal Time')
fig.update_layout(xaxis_title='Total Bill ($)', yaxis_title='Number of People', legend_title='Meal Time')
fig.show()

print("Report:")
print("1. Correlation Analysis:")
correlation_matrix = df_subset.corr()
print(correlation_matrix)

print("\n2. Average Total Bill:")
avg_total_bill_lunch = df_subset[df_subset['time'] == 'Lunch']['total_bill'].mean()
avg_total_bill_dinner = df_subset[df_subset['time'] == 'Dinner']['total_bill'].mean()
print(f"Average total bill for lunch: ${avg_total_bill_lunch:.2f}")
print(f"Average total bill for dinner: ${avg_total_bill_dinner:.2f}")

print("\n3. Average Number of People:")
avg_size_lunch = df_subset[df_subset['time'] == 'Lunch']['size'].mean()
avg_size_dinner = df_subset[df_subset['time'] == 'Dinner']['size'].mean()
print(f"Average number of people for lunch: {avg_size_lunch:.2f}")
print(f"Average number of people for dinner: {avg_size_dinner:.2f}")


df.columns = df.columns.str.strip()


tips_by_day = df.groupby('day')['tip'].sum().reset_index()

fig = px.bar(tips_by_day, x='day', y='tip', title='Tips Given to Waiters by Day of the Week')
fig.update_layout(xaxis_title='Day of Week', yaxis_title='Total Tips ($)')
fig.show()

max_tips_day = tips_by_day.loc[tips_by_day['tip'].idxmax()]['day']
max_tips_amount = tips_by_day['tip'].max()


print("Report:")
print(f"The day with the highest tips given to waiters is {max_tips_day}, with a total of ${max_tips_amount:.2f} in tips.")

df.columns = df.columns.str.strip()

tips_by_gender = df.groupby('sex')['tip'].sum().reset_index()


print("Report:")
print(tips_by_gender)


fig = px.bar(tips_by_gender, x='sex', y='tip', title='Total Tips Given to Waiters by Gender')
fig.update_layout(xaxis_title='Gender', yaxis_title='Total Tips ($)')
fig.show()

df.columns = df.columns.str.strip()

tips_by_day = df.groupby('day')['tip'].sum().reset_index()


print("Report:")
print(tips_by_day)

fig = px.bar(tips_by_day, x='day', y='tip', title='Total Tips Given to Waiters by Day of the Week')
fig.update_layout(xaxis_title='Day of Week', yaxis_title='Total Tips ($)')
fig.show()

df.columns = df.columns.str.strip()


tips_by_smoker = df.groupby('smoker')['tip'].sum().reset_index()


print("Report:")
print(tips_by_smoker)

#  smoker vs. non-smoker
fig = px.bar(tips_by_smoker, x='smoker', y='tip', title='Total Tips Given to Waiters by Smoking Status')
fig.update_layout(xaxis_title='Smoker (Yes/No)', yaxis_title='Total Tips ($)')
fig.show()


df.columns = df.columns.str.strip()

#  total tips for each group
tips_by_time = df.groupby('time')['tip'].sum().reset_index()

print("Report:")
print(tips_by_time)

#  lunch or dinner
fig = px.bar(tips_by_time, x='time', y='tip', title='Total Tips Given to Waiters by Meal Time')
fig.update_layout(xaxis_title='Meal Time (Lunch/Dinner)', yaxis_title='Total Tips ($)')
fig.show()


print(df.head())

df = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'])


X = df.drop('tip', axis=1)
y = df['tip']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the actual vs. predicted values
fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Tip', 'y': 'Predicted Tip'}, title='Actual vs. Predicted Tips')
fig.show()

print("Report:")
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
