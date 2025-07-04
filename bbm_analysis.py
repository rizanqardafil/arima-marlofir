"""
BBM Analysis Module
Contains all ARIMA analysis functions
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def test_stationarity(timeseries):
    """Test stationarity using Dickey-Fuller test"""
    result = adfuller(timeseries, autolag='AIC')
    return {
        'statistic': result[0],
        'pvalue': result[1],
        'is_stationary': result[1] <= 0.05
    }


def find_best_arima_order(series, max_p=3, max_q=3, max_d=2):
    """Find best ARIMA order using AIC"""
    best_aic = float('inf')
    best_params = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    return best_params if best_params else (1, 1, 1), best_aic


def run_arima_analysis(series, location, bbm_type, forecast_periods=12):
    """Run complete ARIMA analysis"""
    
    # Test stationarity
    stationarity = test_stationarity(series)
    
    # Find best ARIMA order
    best_order, best_aic = find_best_arima_order(series)
    
    # Fit model
    try:
        model = ARIMA(series, order=best_order)
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_periods)
        conf_int = fitted_model.get_forecast(steps=forecast_periods).conf_int()
        
        # Calculate metrics
        if len(series) >= 8:
            split_point = int(len(series) * 0.8)
            train_data = series[:split_point]
            test_data = series[split_point:]
            
            train_model = ARIMA(train_data, order=best_order).fit()
            test_forecast = train_model.forecast(steps=len(test_data))
            
            rmse = np.sqrt(mean_squared_error(test_data, test_forecast))
            mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100
        else:
            rmse = mape = None
        
        return {
            'location': location,
            'bbm_type': bbm_type,
            'arima_order': best_order,
            'aic': best_aic,
            'bic': fitted_model.bic,
            'rmse': rmse,
            'mape': mape,
            'is_stationary': stationarity['is_stationary'],
            'forecast': forecast,
            'conf_int': conf_int,
            'model': fitted_model
        }
        
    except Exception as e:
        return {
            'location': location,
            'bbm_type': bbm_type,
            'error': str(e)
        }


def create_forecast_chart(historical_data, forecast_result, dates):
    """Create forecast visualization"""
    
    if 'error' in forecast_result:
        return None
    
    # Prepare data
    historical_dates = dates
    forecast_periods = len(forecast_result['forecast'])
    last_date = historical_dates[-1]
    forecast_dates = [last_date + timedelta(days=30*(i+1)) for i in range(forecast_periods)]
    
    # Create figure
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_data,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_result['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval
    if forecast_result['conf_int'] is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=list(forecast_result['conf_int'].iloc[:, 1]) + list(forecast_result['conf_int'].iloc[:, 0])[::-1],
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
    
    # Layout
    fig.update_layout(
        title=f"ARIMA Forecast - {forecast_result['location']} ({forecast_result['bbm_type']})",
        xaxis_title="Tanggal",
        yaxis_title="Konsumsi BBM (Liter)",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def generate_bbm_data(locations, num_months, base_params):
    """Generate realistic BBM data with seasonal patterns"""
    
    bbm_data = {}
    
    for loc in locations:
        # Get parameters for this location
        base_t1 = base_params[f"{loc}_base_t1"]
        var_t1 = base_params[f"{loc}_var_t1"]
        base_t2 = base_params[f"{loc}_base_t2"]
        var_t2 = base_params[f"{loc}_var_t2"]
        
        # Set consistent seed for location
        np.random.seed(hash(loc) % 1000)
        
        # BBM Tipe 1 data with seasonal pattern
        t1_data = []
        for i in range(num_months):
            seasonal = 1 + 0.2 * np.sin(2 * np.pi * i / 12)  # Seasonal pattern
            trend = 1 + 0.02 * i  # 2% growth per month
            variation = np.random.normal(1.0, var_t1/100)
            value = base_t1 * seasonal * trend * variation
            t1_data.append(max(value, base_t1 * 0.5))
        
        # BBM Tipe 2 data with higher seasonal variation
        t2_data = []
        for i in range(num_months):
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * i / 12)  # Higher seasonal variation
            trend = 1 + 0.03 * i  # 3% growth per month
            variation = np.random.normal(1.0, var_t2/100)
            value = base_t2 * seasonal * trend * variation
            t2_data.append(max(value, base_t2 * 0.5))
        
        bbm_data[loc] = {
            'bbm_tipe_1': t1_data,
            'bbm_tipe_2': t2_data
        }
    
    return bbm_data


def generate_vehicle_data(num_months, vehicle_params):
    """Generate vehicle count data"""
    
    vehicle_types = ['Kendaraan Air', 'Roda Dua', 'Roda Tiga', 'Roda Empat', 'Roda Lima', 'Alat Berat']
    vehicle_data = {}
    
    for i, vehicle_type in enumerate(vehicle_types):
        base_count = vehicle_params.get(f"vehicle_{i}_base", [100, 500, 200, 300, 50, 25][i])
        var_pct = vehicle_params.get(f"vehicle_{i}_var", 15)
        
        np.random.seed(42 + i)
        
        monthly_counts = []
        for month in range(num_months):
            # Growth trend
            growth = 1 + (0.01 * month)  # 1% growth per month
            
            # Seasonal variation (less for vehicles)
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
            
            # Random variation
            variation = np.random.normal(1.0, var_pct/100)
            
            final_count = base_count * growth * seasonal * variation
            monthly_counts.append(max(int(final_count), int(base_count * 0.7)))
        
        vehicle_data[vehicle_type] = monthly_counts
    
    return vehicle_data


def generate_wave_data(num_months, wave_params):
    """Generate wave height data"""
    
    wave_locations = ['Pantai Utara', 'Pantai Selatan', 'Pantai Timur', 'Pantai Barat']
    wave_data = {}
    
    for i, location in enumerate(wave_locations):
        base_height = wave_params.get(f"wave_{i}_base", [1.5, 2.0, 1.8, 1.6][i])
        var_pct = wave_params.get(f"wave_{i}_var", 30)
        
        np.random.seed(50 + i)
        
        monthly_heights = []
        for month in range(num_months):
            # Strong seasonal pattern (higher waves in certain months)
            seasonal = 1 + 0.4 * np.sin(2 * np.pi * month / 12 + np.pi/3)  # Peak around month 8-10
            
            # Random variation (waves are quite variable)
            variation = np.random.normal(1.0, var_pct/100)
            
            final_height = base_height * seasonal * variation
            monthly_heights.append(max(final_height, base_height * 0.4))  # Min 40% of base
        
        wave_data[f"Wave_{location}"] = monthly_heights
    
    return wave_data


def create_summary_table(analysis_results):
    """Create summary table from analysis results"""
    
    summary_data = []
    for (loc, bbm_type), result in analysis_results.items():
        if 'error' not in result:
            summary_data.append({
                'Lokasi': loc,
                'BBM Type': bbm_type,
                'ARIMA Order': f"{result['arima_order']}",
                'AIC': f"{result['aic']:.2f}",
                'BIC': f"{result['bic']:.2f}",
                'RMSE': f"{result['rmse']:.0f}" if result['rmse'] else "N/A",
                'MAPE (%)': f"{result['mape']:.1f}" if result['mape'] else "N/A",
                'Stationary': "✅ Yes" if result['is_stationary'] else "❌ No"
            })
    
    return pd.DataFrame(summary_data)


def create_forecast_table(result, dates):
    """Create detailed forecast table"""
    
    if 'error' in result:
        return pd.DataFrame()
    
    last_date = dates[-1]
    forecast_dates = [last_date + timedelta(days=30*(i+1)) for i in range(len(result['forecast']))]
    
    forecast_df = pd.DataFrame({
        'Bulan': [d.strftime('%b %Y') for d in forecast_dates],
        'Forecast (Liter)': [f"{val:,.0f}" for val in result['forecast']],
        'Lower CI': [f"{val:,.0f}" for val in result['conf_int'].iloc[:, 0]] if result['conf_int'] is not None else ["N/A"] * len(result['forecast']),
        'Upper CI': [f"{val:,.0f}" for val in result['conf_int'].iloc[:, 1]] if result['conf_int'] is not None else ["N/A"] * len(result['forecast'])
    })
    
    return forecast_df