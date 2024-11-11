# EX.NO.09   A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on Summer olympic medals forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of Summer olympic medals
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("/content/Summer_olympic_Medals.csv")

# Create a 'Total_Medals' column by summing Gold, Silver, and Bronze columns
data['Total_Medals'] = data['Gold'] + data['Silver'] + data['Bronze']

# Convert 'Year' to datetime format and set as index
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Group by 'Year' and sum the 'Total_Medals' for each year
yearly_medals = data.groupby(data['Year'].dt.year)['Total_Medals'].sum().reset_index()
yearly_medals.columns = ['Year', 'Total_Medals']

# Set 'Year' as the index
yearly_medals['Year'] = pd.to_datetime(yearly_medals['Year'], format='%Y')
yearly_medals.set_index('Year', inplace=True)

# Define ARIMA model function
def arima_model(data, target_variable, order):
    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast on the test set
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Year')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Call the ARIMA model on the yearly medals data
arima_model(yearly_medals, 'Total_Medals', order=(5, 1, 0))
```

### OUTPUT:

![image](https://github.com/user-attachments/assets/ed492b43-7a44-45e9-905b-4cc32b8ad08e)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
