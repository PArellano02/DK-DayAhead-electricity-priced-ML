# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
from xgboost import XGBRegressor, plot_importance
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import seaborn as sns
from PyPDF2 import PdfMerger
from reportlab.lib import colors
import warnings

warnings.filterwarnings('ignore')

pdf_paths = []

# Load datasets
ninja_pv_wind_path = '/Users/pedroarellano/Desktop/Side Hustle/ninja_pv_wind_profiles_singleindex.csv'
renewable_capacity_path = '/Users/pedroarellano/Desktop/Side Hustle/renewable_capacity_timeseries.csv'
time_series_data_path = '/Users/pedroarellano/Desktop/Side Hustle/time_series_60min_singleindex.csv'
weather_data_path = '/Users/pedroarellano/Desktop/Side Hustle/weather_data.csv'

# Check if files exist
for file_path in [ninja_pv_wind_path, renewable_capacity_path, time_series_data_path, weather_data_path]:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please ensure the file is correctly uploaded or the path is accurate.")
        exit(1)

# Load datasets
ninja_pv_wind = pd.read_csv(ninja_pv_wind_path)
renewable_capacity = pd.read_csv(renewable_capacity_path)
time_series_data = pd.read_csv(time_series_data_path)
weather_data = pd.read_csv(weather_data_path)


# Rename 'time' column to 'utc_timestamp' in ninja_pv_wind dataset
if 'time' in ninja_pv_wind.columns:
    ninja_pv_wind.rename(columns={'time': 'utc_timestamp'}, inplace=True)

# Convert timestamp columns to datetime format
for df in [ninja_pv_wind, renewable_capacity, time_series_data, weather_data]:
    if 'utc_timestamp' in df.columns:
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        df.set_index('utc_timestamp', inplace=True)
    elif 'cet_cest_timestamp' in df.columns:
        df['cet_cest_timestamp'] = pd.to_datetime(df['cet_cest_timestamp'])
        df.set_index('cet_cest_timestamp', inplace=True)
    elif 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.set_index('date_time', inplace=True)

# Filter data for years 2015 - 2019
data_filtered = {
    'ninja_pv_wind': ninja_pv_wind[(ninja_pv_wind.index.year >= 2015) & (ninja_pv_wind.index.year <= 2019)],
   # 'renewable_capacity': renewable_capacity[(renewable_capacity.index.year >= 2015) & (renewable_capacity.index.year <= 2019)],
    'time_series_data': time_series_data[(time_series_data.index.year >= 2015) & (time_series_data.index.year <= 2019)],
    'weather_data': weather_data[(weather_data.index.year >= 2015) & (weather_data.index.year <= 2019)],
}

# Merge datasets on timestamp
merged_data = pd.merge(data_filtered['time_series_data'], data_filtered['ninja_pv_wind'], left_index=True, right_index=True, how='inner')
#merged_data = pd.merge(merged_data, data_filtered['renewable_capacity'], left_index=True, right_index=True, how='inner')
merged_data = pd.merge(merged_data, data_filtered['weather_data'], left_index=True, right_index=True, how='inner')


# Convert 'cet_cest_timestamp' to datetime format with UTC
if 'cet_cest_timestamp' in merged_data.columns:
    merged_data['cet_cest_timestamp'] = pd.to_datetime(merged_data['cet_cest_timestamp'], utc=True)

# Extract features from 'cet_cest_timestamp'
if 'cet_cest_timestamp' in merged_data.columns:
    merged_data['cet_cest_timestamp'] = pd.to_datetime(merged_data['cet_cest_timestamp'])
    merged_data['cet_cest_year'] = merged_data['cet_cest_timestamp'].dt.year
    merged_data['cet_cest_month'] = merged_data['cet_cest_timestamp'].dt.month
    merged_data['cet_cest_day'] = merged_data['cet_cest_timestamp'].dt.day
    merged_data['cet_cest_hour'] = merged_data['cet_cest_timestamp'].dt.hour
    merged_data['cet_cest_minute'] = merged_data['cet_cest_timestamp'].dt.minute
    merged_data['cet_cest_second'] = merged_data['cet_cest_timestamp'].dt.second

# Define the seasons
winter_months = [12, 1, 2]
spring_months = [3, 4, 5]
summer_months = [6, 7, 8]
fall_months = [9, 10, 11]

# Create binary seasonal variables
merged_data['winter'] = merged_data['cet_cest_month'].apply(lambda x: 1 if x in winter_months else 0)
merged_data['spring'] = merged_data['cet_cest_month'].apply(lambda x: 1 if x in spring_months else 0)
merged_data['summer'] = merged_data['cet_cest_month'].apply(lambda x: 1 if x in summer_months else 0)
merged_data['fall'] = merged_data['cet_cest_month'].apply(lambda x: 1 if x in fall_months else 0)

# Split data into training and testing sets
train_data = merged_data[(merged_data.index.year < 2019)] 
test_data = merged_data[(merged_data.index.year == 2019)]



# Define the file path where you want to save the CSV file
csv_file_path = '/Users/pedroarellano/Desktop/Side Hustle/merged_data.csv'

# Save the merged_data DataFrame to a CSV file
merged_data.to_csv(csv_file_path, index=False)

print(f"CSV file saved to {csv_file_path}")

# x_variables =

ts_vars = ["DK_load_forecast_entsoe_transparency",	'DK_solar_capacity','DK_wind_capacity', 'DK_wind_offshore_capacity', 
         'DK_wind_onshore_capacity',	'DK_1_load_forecast_entsoe_transparency','DK_2_load_forecast_entsoe_transparency', 'NO_2_load_forecast_entsoe_transparency', 
         'NO_3_load_forecast_entsoe_transparency',    'NO_4_load_forecast_entsoe_transparency', 'NO_5_load_forecast_entsoe_transparency', 'AT_load_forecast_entsoe_transparency',
         'CZ_load_forecast_entsoe_transparency', 'CH_load_forecast_entsoe_transparency', 'NL_load_forecast_entsoe_transparency', 'SE_load_forecast_entsoe_transparency',
         'SE_wind_capacity', 'SE_wind_onshore_capacity', 'SE_wind_offshore_capacity' , 'BE_load_forecast_entsoe_transparency',
         'DE_load_forecast_entsoe_transparency',	"DE_solar_capacity", 'DE_solar_profile', 	
         'DE_wind_capacity',	'DE_wind_profile', 'DE_wind_offshore_capacity', 'DE_wind_offshore_profile',
         'DE_wind_onshore_capacity',	'DE_wind_onshore_profile', 'DE_50hertz_load_forecast_entsoe_transparency',
         'DE_amprion_load_forecast_entsoe_transparency',	'DE_tennet_load_forecast_entsoe_transparency',
         'DE_transnetbw_load_forecast_entsoe_transparency', 'NO_2_load_forecast_entsoe_transparency', 
         'NO_3_load_forecast_entsoe_transparency',	'NO_4_load_forecast_entsoe_transparency', 'NO_5_load_forecast_entsoe_transparency', 'ES_load_forecast_entsoe_transparency',
         'FI_load_forecast_entsoe_transparency']

x_variables1 = ts_vars + ['cet_cest_year', 'cet_cest_month', 'cet_cest_day', 'cet_cest_hour']

# Define features (X) and target (y)
# Replace 'target_column' with the actual column name you want to predict
if 'DK_1_price_day_ahead' not in merged_data.columns:
    raise ValueError("The column 'target_column' is not found in the dataset. Please specify the correct target column.")

train_data = train_data.dropna(subset=x_variables1 + ['DK_1_price_day_ahead'])
test_data = test_data.dropna(subset=x_variables1 + ['DK_1_price_day_ahead'])

X_train = train_data[x_variables1]
y_train = train_data['DK_1_price_day_ahead']
X_test = test_data[x_variables1]
y_test = test_data['DK_1_price_day_ahead']

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=50, criterion= "squared_error",  random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Model Performance:\nMAE: {mae}\nRMSE: {rmse}\nR2: {r2}\nTrain R2: {train_r2}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance Model 1')
plt.gca().invert_yaxis()
#plt.show()

# Save processed data and feature importance
results_model_1df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})
results_model_1df.to_csv('results1.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

# creating variables for second model

ts_vars = ts_vars

# Check if 'utc_timestamp' exists in time_series_data before dropping
if 'utc_timestamp' in time_series_data.columns:
    bad_ts_vars = time_series_data.drop(columns=ts_vars + ['utc_timestamp'])
else:
    bad_ts_vars = time_series_data.drop(columns= ts_vars)

# Extract column names from bad_ts_vars
bad_ts_vars_columns = bad_ts_vars.columns.tolist() # + ['cet_cest_second'] + ['cet_cest_minute']


# Create x_variables2 by excluding columns in bad_ts_vars_columns from merged_data
x_variables2 = [col for col in merged_data.columns if col not in bad_ts_vars_columns]



X_train = train_data[x_variables2]
y_train = train_data['DK_1_price_day_ahead']
X_test = test_data[x_variables2]
y_test = test_data['DK_1_price_day_ahead']

weights = train_data['cet_cest_year'].apply(lambda x: 1 if x in [2018] else 1)

print(' ')
print('XGBoost Model All variables')

# Hyperparameters tuned using GridSearchCV
# Train the final model with the best parameters
XGB_1 = XGBRegressor(n_estimators =110, learning_rate = 0.01, subsample= 0.4, colsample_bytree= .8, max_depth=11, n_jobs=-1, random_state=42)
XGB_1.fit(X_train, y_train, sample_weight=weights)

# Predict on test data
y_pred = XGB_1.predict(X_test)
y_train_pred = XGB_1.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Decimals
mae = f"{mae:.3f}"
rmse = f"{rmse:.3f}"
r2 = f"{r2:.4f}"
train_r2 = f"{train_r2:.4f}"

num_features = len(X_train.columns)

# Evaluate the model no January
test_data_no_jan = test_data[test_data['cet_cest_month'] != 1]
y_test_no_jan = test_data_no_jan['DK_1_price_day_ahead']
X_test_no_jan = test_data_no_jan[x_variables2]

#testing without january
y_pred_no_jan = XGB_1.predict(X_test_no_jan)
y_train_pred_no_jan = XGB_1.predict(X_train)

mae_no_jan = mean_absolute_error(y_test_no_jan, y_pred_no_jan)
rmse_no_jan = mean_squared_error(y_test_no_jan, y_pred_no_jan)
r2_no_jan = r2_score(y_test_no_jan, y_pred_no_jan)
train_r2_no_jan = r2_score(y_train, y_train_pred_no_jan)

# Decimals

mae_no_jan = f"{mae_no_jan:.3f}"
rmse_no_jan = f"{rmse_no_jan:.3f}"
r2_no_jan = f"{r2_no_jan:.4f}"
train_r2_no_jan = f"{train_r2_no_jan:.4f}"

# Create a table to compare the metrics
data = [
    ["Metric", "Model Performance"],
    ["MAE", mae],
    ["RMSE", rmse],
    ["R2", r2],
    ["Train R2", train_r2],
    ["Features", num_features]
]

table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

print(f"Model Performance:\nMAE: {mae}\nRMSE: {rmse}\nR2: {r2}\nTrain R2: {train_r2}")

# Feature importance
plt.figure(figsize=(10, 8))
plot_importance(XGB_1, importance_type='weight', max_num_features=10)
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

residuals = y_test - y_pred

# Add residuals to the test data
test_data['residuals'] = residuals

#line of best fit
slope, intercept = np.polyfit(test_data['cet_cest_timestamp'].astype(int), residuals, 1)
line_of_best_fit = slope * test_data['cet_cest_timestamp'].astype(int) + intercept

# Create a residual plot
plt.figure(figsize=(10, 6))
plt.scatter( test_data['cet_cest_timestamp'],residuals, alpha=0.5)
plt.plot(test_data['cet_cest_timestamp'], line_of_best_fit, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.grid(True)
plt.legend()
# Add text annotations for slope and intercept
plt.text(0.05, 0.95, f'Slope: {slope:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('residuals_grap.png')
plt.close()


# Add predictions to the test data
test_data['predictions'] = y_pred

# Group by hour and calculate the mean residuals for each hour
residuals_by_hour = test_data.groupby(test_data['cet_cest_timestamp'].dt.hour)['residuals'].mean().reset_index()

# Calculate the line of best fit for residuals by hour
slope_hour, intercept_hour = np.polyfit(residuals_by_hour['cet_cest_timestamp'], residuals_by_hour['residuals'], 1)
line_of_best_fit_hour = slope_hour * residuals_by_hour['cet_cest_timestamp'] + intercept_hour

# Group by month and calculate the mean residuals for each month
residuals_by_month = test_data.groupby(test_data['cet_cest_timestamp'].dt.month)['residuals'].mean().reset_index()

# Calculate the line of best fit for residuals by month
slope_month, intercept_month = np.polyfit(residuals_by_month['cet_cest_timestamp'], residuals_by_month['residuals'], 1)
line_of_best_fit_month = slope_month * residuals_by_month['cet_cest_timestamp'] + intercept_month

# Calculate RMSE by month and hour
rmse_by_month_hour = test_data.groupby(['cet_cest_month', 'cet_cest_hour']).apply(lambda x: np.sqrt(mean_squared_error(x['DK_1_price_day_ahead'], x['predictions']))).reset_index(name='rmse')

# Pivot the data for heatmap
rmse_pivot = rmse_by_month_hour.pivot(index='cet_cest_month', columns= 'cet_cest_hour', values='rmse')


# Plot RMSE by month and save as image
plt.figure(figsize=(10, 6))
plt.scatter(residuals_by_month['cet_cest_timestamp'], residuals_by_month['residuals'], alpha=0.5)
plt.plot(residuals_by_month['cet_cest_timestamp'], line_of_best_fit_month, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Residuals')
plt.title('Residuals by Month')
plt.text(0.05, 0.95, f'Slope: {slope_month:.4f}, Intercept: {intercept_month:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('rmse_by_month.png')
plt.close()

# Plot RMSE by hour and save as image
plt.figure(figsize=(10, 6))
plt.scatter(residuals_by_hour['cet_cest_timestamp'], residuals_by_hour['residuals'], alpha=0.5)
plt.plot(residuals_by_hour['cet_cest_timestamp'], line_of_best_fit_hour, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Residuals')
plt.title('Residuals by Hour')
plt.text(0.05, 0.95, f'Slope: {slope_hour:.4f}, Intercept: {intercept_hour:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('rmse_by_hour.png')
plt.legend()
plt.close()

# Plot RMSE by month and hour (heatmap) and save as image
plt.figure(figsize=(24, 16))
sns.heatmap(rmse_pivot, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 14})
plt.xlabel('Hour')
plt.ylabel('Month')
plt.title('RMSE by Month and Hour')
plt.savefig('rmse_by_month_hour.png')
plt.close()

# Create a PDF document and include the saved images
pdf_path = 'XGB_1_performance_report.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
elements = []
pdf_paths.append(pdf_path)

# Add images to the PDF
styles = getSampleStyleSheet()
elements.append(Paragraph('Performance Metrics: All variables', styles['Title']))
elements.append(table)
elements.append(Image('residuals_grap.png', width=400, height=300))
elements.append(Image('rmse_by_month.png', width=400, height=300))
elements.append(Image('rmse_by_hour.png', width=400, height=300))
elements.append(Image('rmse_by_month_hour.png', width=450, height=330))
elements.append(Image('feature_importance.png', width=400, height=300))

# Build the PDF
doc.build(elements)

print(f"PDF report saved to {pdf_path}")


# Load the daily dataset
daily_path = '/Users/pedroarellano/Desktop/Side Hustle/daily.csv'
daily = pd.read_csv(daily_path)

# Ensure the date column in the daily dataset is in datetime format
daily['date'] = pd.to_datetime(daily['date'])

# Extract the date from the hourly timestamp in the merged_data
merged_data['date'] = merged_data['cet_cest_timestamp'].dt.date

# Ensure the date column in merged_data is in datetime format
merged_data['date'] = pd.to_datetime(merged_data['date'])

# Merge the daily dataset with the hourly dataset on the date
merged_data = pd.merge(merged_data, daily, on='date', how='left')

# Drop the date column as it's no longer needed
merged_data.drop(columns=['date'], inplace=True)

merged_data.rename(columns={'Price': 'NG_Price'}, inplace=True)

# Split data into training and testing sets
train_data = merged_data[merged_data['cet_cest_timestamp'].dt.year < 2019]
test_data = merged_data[merged_data['cet_cest_timestamp'].dt.year == 2019]



# Define features (X) and target (y)
x_variables3 = x_variables2 + ['NG_Price'] # Add NG 'Price' to the features


train_data = train_data.dropna(subset=x_variables3 + ['DK_1_price_day_ahead'])
test_data = test_data.dropna(subset=x_variables3 + ['DK_1_price_day_ahead'])

X_train = train_data[x_variables3]
y_train = train_data['DK_1_price_day_ahead']
X_test = test_data[x_variables3]
y_test = test_data['DK_1_price_day_ahead']
y_train = train_data['DK_1_price_day_ahead']

print(' ')
print(' ')

print('Including Natural Gas Price')


print(' ')
print('XGBoost Model with Natural Gas Price')

# Train the final model with the best parameters
XGB_2 = XGBRegressor(n_estimators= 120, learning_rate = 0.01, subsample= 0.2, colsample_bytree= 0.6, max_depth= 9, n_jobs=-1, random_state=42)
XGB_2.fit(X_train, y_train)


# Predict on test data
y_pred = XGB_2.predict(X_test)
y_train_pred = XGB_2.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Decimals
mae = f"{mae:.3f}"
rmse = f"{rmse:.3f}"
r2 = f"{r2:.4f}"
train_r2 = f"{train_r2:.4f}"


# Evaluate the model no January
test_data_no_jan = test_data[test_data['cet_cest_month'] != 1]
y_test_no_jan = test_data_no_jan['DK_1_price_day_ahead']
X_test_no_jan = test_data_no_jan[x_variables3]

#testing without january
y_pred_no_jan = XGB_2.predict(X_test_no_jan)
y_train_pred_no_jan = XGB_2.predict(X_train)

mae_no_jan = mean_absolute_error(y_test_no_jan, y_pred_no_jan)
rmse_no_jan = mean_squared_error(y_test_no_jan, y_pred_no_jan)
r2_no_jan = r2_score(y_test_no_jan, y_pred_no_jan)
train_r2_no_jan = r2_score(y_train, y_train_pred_no_jan)

# Decimals

mae_no_jan = f"{mae_no_jan:.3f}"
rmse_no_jan = f"{rmse_no_jan:.3f}"
r2_no_jan = f"{r2_no_jan:.4f}"
train_r2_no_jan = f"{train_r2_no_jan:.4f}"

num_features = len(X_train.columns)

# Create a table to compare the metrics
data = [
    ["Metric", "Model Performance"],
    ["MAE", mae],
    ["RMSE", rmse],
    ["R2", r2],
    ["Train R2", train_r2 ],
    ['Features', num_features]
]


table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

print(f"Model Performance:\nMAE: {mae}\nRMSE: {rmse}\nR2: {r2}\nTrain R2: {train_r2}")


# Add predictions to the test data
test_data['predictions'] = y_pred


residuals = y_test - y_pred
# Add residuals to the test data
test_data['residuals'] = residuals


# Group by hour and calculate the mean residuals for each hour
residuals_by_hour = test_data.groupby(test_data['cet_cest_timestamp'].dt.hour)['residuals'].mean().reset_index()

# Calculate the line of best fit for residuals by hour
slope_hour, intercept_hour = np.polyfit(residuals_by_hour['cet_cest_timestamp'], residuals_by_hour['residuals'], 1)
line_of_best_fit_hour = slope_hour * residuals_by_hour['cet_cest_timestamp'] + intercept_hour

# Group by month and calculate the mean residuals for each month
residuals_by_month = test_data.groupby(test_data['cet_cest_timestamp'].dt.month)['residuals'].mean().reset_index()

# Calculate the line of best fit for residuals by month
slope_month, intercept_month = np.polyfit(residuals_by_month['cet_cest_timestamp'], residuals_by_month['residuals'], 1)
line_of_best_fit_month = slope_month * residuals_by_month['cet_cest_timestamp'] + intercept_month

# Calculate RMSE by month and hour
rmse_by_month_hour = test_data.groupby(['cet_cest_month', 'cet_cest_hour']).apply(lambda x: np.sqrt(mean_squared_error(x['DK_1_price_day_ahead'], x['predictions']))).reset_index(name='rmse')

# Pivot the data for heatmap
rmse_pivot = rmse_by_month_hour.pivot(index='cet_cest_month', columns= 'cet_cest_hour', values='rmse')


# Plot RMSE by month and save as image
plt.figure(figsize=(10, 6))
plt.scatter(residuals_by_month['cet_cest_timestamp'], residuals_by_month['residuals'], alpha=0.5)
plt.plot(residuals_by_month['cet_cest_timestamp'], line_of_best_fit_month, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Residuals')
plt.title('Residuals by Month')
plt.text(0.05, 0.95, f'Slope: {slope_month:.4f}, Intercept: {intercept_month:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('rmse_by_month.png')
plt.close()

# Plot RMSE by hour and save as image
plt.figure(figsize=(10, 6))
plt.scatter(residuals_by_hour['cet_cest_timestamp'], residuals_by_hour['residuals'], alpha=0.5)
plt.plot(residuals_by_hour['cet_cest_timestamp'], line_of_best_fit_hour, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Residuals')
plt.title('Residuals by Hour')
plt.text(0.05, 0.95, f'Slope: {slope_hour:.4f}, Intercept: {intercept_hour:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('rmse_by_hour.png')
plt.legend()
plt.close()

# Plot RMSE by month and hour (heatmap) and save as image
plt.figure(figsize=(24, 16))
sns.heatmap(rmse_pivot, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 14})
plt.xlabel('Hour')
plt.ylabel('Month')
plt.title('RMSE by Month and Hour')
plt.savefig('rmse_by_month_hour.png')
plt.close()

# Feature importance
plt.figure(figsize=(10, 8))
plot_importance(XGB_2, importance_type='weight', max_num_features=10)
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()


#line of best fit
slope, intercept = np.polyfit(test_data['cet_cest_timestamp'].astype(int), residuals, 1)
line_of_best_fit = slope * test_data['cet_cest_timestamp'].astype(int) + intercept

# Create a residual plot
plt.figure(figsize=(10, 6))
plt.scatter( test_data['cet_cest_timestamp'],residuals, alpha=0.5)
plt.plot(test_data['cet_cest_timestamp'], line_of_best_fit, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.grid(True)
plt.legend()
# Add text annotations for slope and intercept
plt.text(0.05, 0.95, f'Slope: {slope:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('residuals_grap.png')
plt.close()

# Create a PDF document and include the saved images
pdf_path = 'XGB_2_performance_report.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
elements = []
pdf_paths.append(pdf_path)

# Add images to the PDF
styles = getSampleStyleSheet()
elements.append(Paragraph('Performance Metrics: All variables + NG Prices', styles['Title']))
elements.append(table)
elements.append(Image('residuals_grap.png',width= 400, height=300))
elements.append(Image('rmse_by_month.png', width=400, height=300))
elements.append(Image('rmse_by_hour.png', width=400, height=300))
elements.append(Image('rmse_by_month_hour.png', width=450, height=330))
elements.append(Image('feature_importance.png', width= 400, height=300))

# Build the PDF
doc.build(elements)

print(f"PDF report saved to {pdf_path}")

model = XGB_2.fit(X_train, y_train)

importances = model.feature_importances_
important_features = X_train.columns[importances > 0.003]
X_train_XGB_imp = X_train[important_features]
X_test_XGB_imp = X_test[important_features]

print(' ')
print('XGBoost Model important features')
# Train a XGBoost model


XGB_3 = XGBRegressor(n_estimators= 90, learning_rate= 0.01, subsample= 0.4, colsample_bytree= .8, max_depth=9, n_jobs=-1, random_state=42)
XGB_3.fit(X_train_XGB_imp, y_train)


# Predict on test data
y_pred = XGB_3.predict(X_test_XGB_imp)
y_train_pred = XGB_3.predict(X_train_XGB_imp)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Decimals
mae = f"{mae:.3f}"
rmse = f"{rmse:.3f}"
r2 = f"{r2:.4f}"
train_r2 = f"{train_r2:.4f}"


# Evaluate the model no January
test_data_no_jan = test_data[test_data['cet_cest_month'] > 1]
y_test_no_jan = test_data_no_jan['DK_1_price_day_ahead']
X_test_no_jan = test_data_no_jan[important_features]

#testing without january
y_pred_no_jan = XGB_3.predict(X_test_no_jan)
y_train_pred_no_jan = XGB_3.predict(X_train_XGB_imp)

mae_no_jan = mean_absolute_error(y_test_no_jan, y_pred_no_jan)
rmse_no_jan = mean_squared_error(y_test_no_jan, y_pred_no_jan)
r2_no_jan = r2_score(y_test_no_jan, y_pred_no_jan)
train_r2_no_jan = r2_score(y_train, y_train_pred_no_jan)

# Decimals

mae_no_jan = f"{mae_no_jan:.3f}"
rmse_no_jan = f"{rmse_no_jan:.3f}"
r2_no_jan = f"{r2_no_jan:.4f}"
train_r2_no_jan = f"{train_r2_no_jan:.4f}"

num_features = len(important_features)

# Create a table to compare the metrics
data = [
    ["Metric", "Model Performance"],
    ["MAE", mae],
    ["RMSE", rmse],
    ["R2", r2],
    ["Train R2", train_r2 ],
    ['Features', num_features]
]

table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))


print(f"Model Performance:\nMAE: {mae}\nRMSE: {rmse}\nR2: {r2}\nTrain R2: {train_r2}")

# Add predictions to the test data
test_data['predictions'] = y_pred

residuals = y_test - y_pred
# Add residuals to the test data
test_data['residuals'] = residuals

# Group by hour and calculate the mean residuals for each hour
residuals_by_hour = test_data.groupby(test_data['cet_cest_timestamp'].dt.hour)['residuals'].mean().reset_index()

# Calculate the line of best fit for residuals by hour
slope_hour, intercept_hour = np.polyfit(residuals_by_hour['cet_cest_timestamp'], residuals_by_hour['residuals'], 1)
line_of_best_fit_hour = slope_hour * residuals_by_hour['cet_cest_timestamp'] + intercept_hour

# Group by month and calculate the mean residuals for each month
residuals_by_month = test_data.groupby(test_data['cet_cest_timestamp'].dt.month)['residuals'].mean().reset_index()

# Calculate the line of best fit for residuals by month
slope_month, intercept_month = np.polyfit(residuals_by_month['cet_cest_timestamp'], residuals_by_month['residuals'], 1)
line_of_best_fit_month = slope_month * residuals_by_month['cet_cest_timestamp'] + intercept_month


# Calculate RMSE by month
rmse_by_month = test_data.groupby('cet_cest_month').apply(lambda x: np.sqrt(mean_squared_error(x['DK_1_price_day_ahead'], x['predictions']))).reset_index(name='rmse')

# Calculate RMSE by hour
rmse_by_hour = test_data.groupby('cet_cest_hour').apply(lambda x: np.sqrt(mean_squared_error(x['DK_1_price_day_ahead'], x['predictions']))).reset_index(name='rmse')

# Calculate RMSE by month and hour
rmse_by_month_hour = test_data.groupby(['cet_cest_month', 'cet_cest_hour']).apply(lambda x: np.sqrt(mean_squared_error(x['DK_1_price_day_ahead'], x['predictions']))).reset_index(name='rmse')

# Pivot the data for heatmap
rmse_pivot = rmse_by_month_hour.pivot(index='cet_cest_month', columns= 'cet_cest_hour', values='rmse')


# Plot RMSE by month and save as image
plt.figure(figsize=(10, 6))
plt.scatter(residuals_by_month['cet_cest_timestamp'], residuals_by_month['residuals'], alpha=0.5)
plt.plot(residuals_by_month['cet_cest_timestamp'], line_of_best_fit_month, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Residuals')
plt.title('Residuals by Month')
plt.text(0.05, 0.95, f'Slope: {slope_month:.4f}, Intercept: {intercept_month:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('rmse_by_month.png')
plt.close()

# Plot RMSE by hour and save as image
plt.figure(figsize=(10, 6))
plt.scatter(residuals_by_hour['cet_cest_timestamp'], residuals_by_hour['residuals'], alpha=0.5)
plt.plot(residuals_by_hour['cet_cest_timestamp'], line_of_best_fit_hour, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Residuals')
plt.title('Residuals by Hour')
plt.text(0.05, 0.95, f'Slope: {slope_hour:.4f}, Intercept: {intercept_hour:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('rmse_by_hour.png')
plt.legend()
plt.close()

# Plot RMSE by month and hour (heatmap) and save as image
plt.figure(figsize=(24, 16))
sns.heatmap(rmse_pivot, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 14})
plt.xlabel('Hour')
plt.ylabel('Month')
plt.title('RMSE by Month and Hour')
plt.savefig('rmse_by_month_hour.png')
plt.close()

# Feature importance
plt.figure(figsize=(10, 8))
plot_importance(XGB_3, importance_type='weight', max_num_features=10)
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

residuals = y_test - y_pred
#line of best fit
slope, intercept = np.polyfit(test_data['cet_cest_timestamp'].astype(int), residuals, 1)
line_of_best_fit = slope * test_data['cet_cest_timestamp'].astype(int) + intercept

# Create a residual plot
plt.figure(figsize=(10, 6))
plt.scatter( test_data['cet_cest_timestamp'],residuals, alpha=0.5)
plt.plot(test_data['cet_cest_timestamp'], line_of_best_fit, color='red', label='Line of Best Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.grid(True)
plt.legend()
# Add text annotations for slope and intercept
plt.text(0.05, 0.95, f'Slope: {slope:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('residuals_grap.png')
plt.close()

# Create a PDF document and include the saved images
pdf_path = 'XGB_3_performance_report.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
elements = []
pdf_paths.append(pdf_path)

# Add images to the PDF
# Add performance metrics to the PDF
styles = getSampleStyleSheet()
elements.append(Paragraph('Performance Metrics: Important Features Model', styles['Title']))
elements.append(table)
elements.append(Image('residuals_grap.png',width= 400, height=300))
elements.append(Image('rmse_by_month.png', width=400, height=300))
elements.append(Image('rmse_by_hour.png', width=400, height=300))
elements.append(Image('rmse_by_month_hour.png', width=450, height=330))
elements.append(Image('feature_importance.png', width=400, height=300))

# Build the PDF
doc.build(elements)

print(f"PDF report saved to {pdf_path}")


#Summary statistic

#heatmap 2019
# Calculate average prices by month
prices_by_month = test_data.groupby('cet_cest_month')['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")

# Calculate average prices by hour
prices_by_hour = test_data.groupby('cet_cest_hour')['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")


# Calculate average prices by month and hours
prices_by_month_hour = test_data.groupby(['cet_cest_month', 'cet_cest_hour'])['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")

# Pivot the data for heatmap
price_pivot = prices_by_month_hour.pivot(index='cet_cest_month', columns='cet_cest_hour', values='Average Price')

#Heatmap of average prices by month and hour
plt.figure(figsize=(24, 16))
sns.heatmap(price_pivot, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 14})
plt.xlabel('Hour')
plt.ylabel('Month')
plt.title('2019 Average Prices by Month and Hour')
plt.savefig('actual_prices.png')
plt.close()

#heatmap 2015-2018

# Calculate average prices by month
prices_by_month = train_data.groupby('cet_cest_month')['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")

# Calculate average prices by hour
prices_by_hour = train_data.groupby('cet_cest_hour')['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")

# Calculate average prices by month and hours
prices_by_month_hour = train_data.groupby(['cet_cest_month', 'cet_cest_hour'])['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")

# Pivot the data for heatmap
price_pivot = prices_by_month_hour.pivot(index='cet_cest_month', columns='cet_cest_hour', values='Average Price')

#Heatmap of average prices by month and hour
plt.figure(figsize=(24, 16))
sns.heatmap(price_pivot, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 14})
plt.xlabel('Hour')
plt.ylabel('Month')
plt.title('2015-2018 Average Prices by Month and Hour')
plt.savefig('actual_prices_TrnD.png')
plt.close()

# price of electricity time series
plt.figure(figsize=(10, 6))
plt.plot(test_data['cet_cest_timestamp'], test_data['DK_1_price_day_ahead'], label='Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Prices of Electricity in Denmark')
plt.legend()
plt.savefig('actual.png')
plt.close()

#metrics for table
tr_avg = f"{train_data['DK_1_price_day_ahead'].mean():.2f}"
test_avg = f"{test_data['DK_1_price_day_ahead'].mean():.2f}"
tr_stdev = f"{train_data['DK_1_price_day_ahead'].std():.2f}"
test_stdev = f"{test_data['DK_1_price_day_ahead'].std():.2f}"
tr_min = f"{train_data['DK_1_price_day_ahead'].min():.2f}"
test_min = f"{test_data['DK_1_price_day_ahead'].min():.2f}"
tr_max = f"{train_data['DK_1_price_day_ahead'].max():.2f}"
test_max = f"{test_data['DK_1_price_day_ahead'].max():.2f}"

# Create a table to compare the metrics
data = [
    ["Metric", "2015-2018", "2019"],
    ["AVG Price", tr_avg, test_avg],
    ["STDEV", tr_stdev, test_stdev],
    ["Min", tr_min, test_min],
    ["Max", tr_max, test_max]
]

table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))

#Probs not Duck year graph
train_data['load_net_wnd_pv'] = train_data['DK_load_actual_entsoe_transparency'] - train_data['DK_solar_generation_actual'] - train_data['DK_wind_generation_actual']
test_data['load_net_wnd_pv'] = test_data['DK_load_actual_entsoe_transparency'] - test_data['DK_solar_generation_actual'] - test_data['DK_wind_generation_actual']


#separate years
data_15 = train_data[(train_data['cet_cest_year'] == 2015)]
data_16 = train_data[(train_data['cet_cest_year'] == 2016)]
data_17 = train_data[(train_data['cet_cest_year'] == 2017)]
data_18 = train_data[(train_data['cet_cest_year'] == 2018)]

#Groupby hour
price_by_hour15 = data_15.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour16 = data_16.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour17 = data_17.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour18 = data_18.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour19 = test_data.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')

year_dict = {
    '2015': price_by_hour15,
    '2016': price_by_hour16,
    '2017': price_by_hour17,
    '2018': price_by_hour18,
    '2019': price_by_hour19
}
# Plot the average price per hour for each year
plt.figure(figsize=(14, 8))
for year,data in year_dict.items():
    plt.plot(data['cet_cest_hour'], data['price'], label=(year))

plt.xlabel('Hour of the Day')
plt.ylabel('Average Price')
plt.title('Average Price per Hour for Each Year')
plt.legend(title='Year')
plt.grid(True)
plt.savefig('average_price_per_hour_per_year.png')
plt.close()


# Create a PDF document and include the saved images
pdf_path = 'summary_statistics.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
elements = []

#Groupby hour
load_by_hour15 = data_15.groupby('cet_cest_hour').apply(lambda x: x['load_net_wnd_pv'].mean()).reset_index(name='load')
load_by_hour16 = data_16.groupby('cet_cest_hour').apply(lambda x: x['load_net_wnd_pv'].mean()).reset_index(name='load')
load_by_hour17 = data_17.groupby('cet_cest_hour').apply(lambda x: x['load_net_wnd_pv'].mean()).reset_index(name='load')
load_by_hour18 = data_18.groupby('cet_cest_hour').apply(lambda x: x['load_net_wnd_pv'].mean()).reset_index(name='load')
load_by_hour19 = test_data.groupby('cet_cest_hour').apply(lambda x: x['load_net_wnd_pv'].mean()).reset_index(name='load')



year_dict = {
    '2015': load_by_hour15,
    '2016': load_by_hour16,
    '2017': load_by_hour17,
    '2018': load_by_hour18,
    '2019': load_by_hour19
}

# Plot the average load per hour for each year
plt.figure(figsize=(14, 8))
for year,data in year_dict.items():
    plt.plot(data['cet_cest_hour'], data['load'], label=(year))
plt.xlabel('Hour of the Day')
plt.ylabel('Average Net Load')
plt.title('Average Load Net Wind and PV per Hour for Each Year')
plt.legend(title='Year')
plt.grid(True)
plt.savefig('average_Load_per_hour_per_year.png')
plt.close()

# Create a PDF document and include the saved images
pdf_path = 'summary_statistics.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
elements = []


# Add images to the PDF
styles = getSampleStyleSheet()
elements.append(Paragraph('A View at the Denmark Electricity Market', styles['Title']))
elements.append(Image('DK_IEA_graph.png', width=400, height= 300))
elements.append(Paragraph('Project Overview', styles['Heading3']))
elements.append(Paragraph('Over the past two decades, Denmark has seen a tremendous shift in their energy landscape. Since the year 2000, Denmark evolved from relying on coal for their electricity generation to relying primarily on wind, biofuels and solar energy. The high penetration of intermittent resources such as wind and solar present an opportunity for batteries and other forms of energy arbitrage. However, to perform a profitable arbitrage strategy, one must be able to reliably predict electrity prices. In this project, I use load, resource capacity, weather, and simualated capacity factor data to predict day ahead prices in DK1, a bidding zone in Denmark. This document presents the results and analysis of 3 XGBoost models as well as some vizualizations to gain a better understanding of the day ahead market in DK1.'))
elements.append(Paragraph(' ', styles['Heading3']))
elements.append(table)
elements.append(Image('actual_prices_TrnD.png', width=400, height=300))
elements.append(Image('actual_prices.png', width=400, height=300))
elements.append(Image('average_price_per_hour_per_year.png', width=400, height=300))
elements.append(Image('average_Load_per_hour_per_year.png', width=400, height=300))

# Build the PDF
doc.build(elements)

print(f"PDF report saved to {pdf_path}")


merged_pdf_path = 'model_performance_report.pdf'
merger = PdfMerger()

merger.append('summary_statistics.pdf')

for pdf in pdf_paths:
    merger.append(pdf)

merger.write(merged_pdf_path)
merger.close()

print(f"PDF report saved to {merged_pdf_path}")