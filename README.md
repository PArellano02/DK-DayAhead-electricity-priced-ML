# Day-Ahead Electricity Price Prediction for DK1 Bidding Zone

This was my first attempt at creating an XGBooost model to predict Day Ahead electricity prices in Denmark. Also made some graphs to explain some of the patterns in day-ahead prices in DK and show some of the shortcomings of my model. Maybe when I have some more time I will try to fix this up a little bit and capture that day-to-day variability a lot better!

## Key Results

Here are some of the key results from the analysis:

### Feature Importance

![Feature Importance](https://raw.githubusercontent.com/PArellano02/DK-DayAhead-electricity-priced-ML/main/output/feature_importance.png)

### Actual vs. Predicted Prices

![Actual vs. Predicted](https://raw.githubusercontent.com/PArellano02/DK-DayAhead-electricity-priced-ML/main/output/actual_vs_predicted.png)

### Average Price per Hour for Each Year

![Average Price per Hour](https://raw.githubusercontent.com/PArellano02/DK-DayAhead-electricity-priced-ML/main/output/average_price_per_hour_per_year.png)

### RMSE by Month and Hour

![RMSE by Month and Hour](https://raw.githubusercontent.com/PArellano02/DK-DayAhead-electricity-priced-ML/main/output/rmse_by_month_hour.png)

## Full Report

For a more detailed analysis, please see the full report:

[Model Performance Report](https://github.com/PArellano02/DK-DayAhead-electricity-priced-ML/blob/main/output/model_performance_report.pdf)
