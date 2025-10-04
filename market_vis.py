# Import necessary libraries
import pandas as pd
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

merged_data_path = "/Users/pedroarellano/Desktop/Side Hustle/merged_data.csv"

merged_data = pd.read_csv(merged_data_path)


#heatmap 2015-2019
# Calculate average prices by month
prices_by_month = merged_data.groupby('cet_cest_month')['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")

# Calculate average prices by hour
prices_by_hour = merged_data.groupby('cet_cest_hour')['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")


# Calculate average prices by month and hours
prices_by_month_hour = merged_data.groupby(['cet_cest_month', 'cet_cest_hour'])['DK_1_price_day_ahead'].mean().reset_index(name="Average Price")

# Pivot the data for heatmap
price_pivot = prices_by_month_hour.pivot(index='cet_cest_month', columns='cet_cest_hour', values='Average Price')

#Heatmap of average prices by month and hour
plt.figure(figsize=(24, 16))
sns.heatmap(price_pivot, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 14})
plt.xlabel('Hour')
plt.ylabel('Month')
plt.title('Average Prices by Month and Hour')
plt.savefig('/Users/pedroarellano/Desktop/Side Hustle/Market Visuals/DK1_Montly_Prices.png')
plt.close()









#Daily price curves 

#separate years
data_15 = merged_data[(merged_data['cet_cest_year'] == 2015)]
data_16 = merged_data[(merged_data['cet_cest_year'] == 2016)]
data_17 = merged_data[(merged_data['cet_cest_year'] == 2017)]
data_18 = merged_data[(merged_data['cet_cest_year'] == 2018)]
data_19 = merged_data[(merged_data['cet_cest_year'] == 2019)]

#Groupby hour
price_by_hour15 = data_15.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour16 = data_16.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour17 = data_17.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour18 = data_18.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')
price_by_hour19 = data_19.groupby('cet_cest_hour').apply(lambda x: x['DK_1_price_day_ahead'].mean()).reset_index(name='price')

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
plt.savefig('/Users/pedroarellano/Desktop/Side Hustle/Market Visuals/average_price_per_hour_per_year.png')
plt.close()






# Step 1: Sort prices in descending order
sorted_prices = merged_data['DK_1_price_day_ahead'].sort_values(ascending=False).reset_index(drop=True)

# Step 2: Create x-axis (percentage of time or hours)
duration = (sorted_prices.index + 1) / len(sorted_prices) * 100  # percent of time

# Step 3: Plot
plt.figure(figsize=(10, 6))
plt.bar(duration, sorted_prices, label='DK1 Price Duration Curve', width=1.0)  # Use plt.bar for a bar plot
plt.xlabel('Percentage of Time (%)')
plt.ylabel('Electricity Price (€/MWh)')
plt.title('DK1 Price Duration Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/pedroarellano/Desktop/Side Hustle/Market Visuals/Price_duration.png')
plt.close()