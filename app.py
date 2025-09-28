from flask import Flask, render_template, request, abort
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# Import forecasting functions from model scripts
from models.arima_model import train_and_forecast as arima_forecast
from models.random_forest_model import train_and_forecast as rf_forecast
from models.lstm_model import train_and_forecast as lstm_forecast

app = Flask(__name__)

# Load dataset once on startup
DATA_PATH = r"data/Combined_Gold_Silver_Data.csv"

try:
    # Use the exact column names from your CSV: 'date', 'gold_close', 'silver_close'
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date')
except FileNotFoundError:
    print(f"ERROR: Dataset not found at {DATA_PATH}")
    df = None
except ValueError as ve:
    print(f"ERROR: {ve}")
    df = None

# Extract price series and dates if data loaded
if df is not None:
    gold_prices = df['gold_close'].values
    silver_prices = df['silver_close'].values
    dates = df['date']
else:
    gold_prices = None
    silver_prices = None
    dates = None


def create_plot(dates, actual, forecast, metal, model_name):
    """
    Creates a matplotlib plot comparing actual historical prices and forecasted prices,
    then converts the plot image to a base64 string for embedding in HTML.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label=f'Historical {metal} Price', color='blue')
    future_dates = pd.date_range(dates.iloc[-1], periods=len(forecast) + 1, freq='D')[1:]
    plt.plot(future_dates, forecast, label='Forecasted Price', color='red', linestyle='--')
    plt.title(f'{metal} Price Forecast using {model_name}')
    plt.xlabel('Date')
    plt.ylabel(f'{metal} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_base64 = base64.b64encode(img.getvalue()).decode()
    return plot_base64


@app.route('/')
def home():
    """Render the Home page."""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the About page."""
    return render_template('about.html')


@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """
    Render the Forecast page with form and results.
    Handles form POST submission to run selected model forecasting on chosen metal.
    """
    if df is None:
        # Dataset not loaded
        return render_template('forecast.html', error="Dataset not found. Please check the data folder.")

    if request.method == 'POST':
        # Get form data
        metal = request.form.get('metal')
        model_name = request.form.get('model')
        try:
            horizon = int(request.form.get('horizon', 7))
        except ValueError:
            horizon = 7

        # Validate inputs
        if metal not in ['Gold', 'Silver']:
            abort(400, description="Invalid metal selection.")
        if model_name not in ['ARIMA', 'RandomForest', 'LSTM']:
            abort(400, description="Invalid model selection.")
        if not (1 <= horizon <= 30):
            horizon = 7  # Default forecast horizon

        # Select price series
        prices = gold_prices if metal == 'Gold' else silver_prices

        # Run the selected model
        if model_name == 'ARIMA':
            forecast_values = arima_forecast(prices, horizon)
        elif model_name == 'RandomForest':
            forecast_values = rf_forecast(prices, horizon)
        else:  # LSTM
            forecast_values = lstm_forecast(prices, horizon)

        # Convert forecast to list and round for display
        forecast_values_rounded = [float(np.round(val, 2)) for val in forecast_values]

        # Create plot image string for embedding
        plot_url = create_plot(dates, prices, forecast_values, metal, model_name)

        # Render forecast results with plot
        return render_template('forecast.html',
                               metal=metal,
                               model=model_name,
                               horizon=horizon,
                               forecast=forecast_values_rounded,
                               plot_url=plot_url)

    # GET request: just render the form
    return render_template('forecast.html')


@app.route('/contact')
def contact():
    """Render the Contact page."""
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('Developer.html')

@app.route('/portfolio/simranjeet')
def portfolio_simranjeet():
    return render_template('portfolio_simranjeet.html')

@app.route('/portfolio/akshaykumar')
def portfolio_akshaykumar():
    return render_template('portfolio_akshaykumar.html')


if __name__ == '__main__':
    app.run(debug=True)
