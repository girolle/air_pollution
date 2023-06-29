import csv
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Function to parse air pollution data from CSV
def parse_air_pollution_data(filename):
    data = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)

        # Skip the header row
        next(reader)

        for row in reader:
            date = row[0]
            pm25 = float(row[1]) if row[1].strip() != '' else None
            pm10 = float(row[2]) if row[2].strip() != '' else None
            o3 = float(row[3]) if row[3].strip() != '' else None
            no2 = float(row[4]) if row[4].strip() != '' else None

            data.append({
                'date': date,
                'pm25': pm25,
                'pm10': pm10,
                'o3': o3,
                'no2': no2
            })

    return data

# Function to perform SARIMA prediction for a given variable
def perform_sarima(data, variable):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    df.set_index('date', inplace=True)

    # Generate a complete date range from 2013/12/31 to 2023/6/8 with daily frequency
    date_range = pd.date_range(start='2013-12-31', end='2023-06-08', freq='D')
    df = df.reindex(date_range)

    # Fit SARIMA model to the data
    # SARIMA(p, d, q)(P, D, Q, S)
    # - p: Autoregressive order (number of lagged observations)
    # - d: Differencing order (order of differencing to make the time series stationary)
    # - q: Moving average order (number of lagged forecast errors)
    # - P: Seasonal autoregressive order (seasonal lagged observations)
    # - D: Seasonal differencing order (seasonal order of differencing)
    # - Q: Seasonal moving average order (seasonal number of lagged forecast errors)
    # - S: Seasonal period (number of time steps in each season)
    model = SARIMAX(df[variable], order=(1, 1, 1), seasonal_order=(1, 0, 0, 7))
    result = model.fit()

    # Make a one-day-ahead forecast for 2023/6/9
    forecast = result.get_forecast(steps=1)
    predicted_value = forecast.predicted_mean[0]

    return predicted_value

# Parse the air pollution data
filename = 'paris-air-quality.csv'
parsed_data = parse_air_pollution_data(filename)

# Variables for prediction
variables = ['pm25', 'pm10', 'o3', 'no2']

# Perform predictions for each variable
predictions = {}
for variable in variables:
    prediction = perform_sarima(parsed_data, variable)
    predictions[variable] = prediction

# Write predictions to a file
output_filename = 'predictions.csv'
with open(output_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Variable', 'Predicted Value'])
    for variable, prediction in predictions.items():
        writer.writerow([variable, prediction])

# Compare with real values for 2023/6/9
real_values = {
    'pm25': 98,
    'pm10': 31,
    'o3': 41,
    'no2': 22
}

# Print and compare predicted values with real values
print("Predicted values for 2023/6/9:")
for variable, prediction in predictions.items():
    real_value = real_values[variable]
    print(f"{variable}: Predicted={prediction:.1f}, Real={real_value}")
