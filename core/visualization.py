"""
Visualization Engine - Interactive Charts & Tables
Enhanced visualization engine for BBM analysis with explicit function exports
FIXED: Datetime handling in plotly charts
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.colors as pc

class VisualizationEngine:
    """
    Advanced visualization engine for BBM analysis with interactive capabilities
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3
        self.forecast_colors = {
            'historical': '#1f77b4',
            'forecast': '#ff7f0e', 
            'confidence': 'rgba(255, 127, 14, 0.2)',
            'trend': '#2ca02c'
        }
        
        self.default_layout = {
            'template': theme,
            'hovermode': 'x unified',
            'showlegend': True,
            'height': 500
        }
    
    def create_forecast_chart(self, historical_data: List[float], 
                            forecast_result: Dict[str, Any], 
                            dates: List[datetime],
                            title: str = None,
                            y_axis_title: str = "Values") -> go.Figure:
        """Create ARIMA forecast visualization with confidence intervals"""
        if 'error' in forecast_result:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {forecast_result['error']}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(title="Forecast Error", **self.default_layout)
            return fig
        
        # Prepare forecast data
        if 'forecast' in forecast_result:
            forecast_values = forecast_result['forecast']
            if isinstance(forecast_values, dict) and 'values' in forecast_values:
                forecast_values = forecast_values['values']
        else:
            forecast_values = []
        
        # Generate forecast dates
        if dates:
            last_date = dates[-1]
            forecast_dates = [
                last_date + timedelta(days=30*(i+1)) 
                for i in range(len(forecast_values))
            ]
        else:
            forecast_dates = list(range(len(historical_data), len(historical_data) + len(forecast_values)))
        
        # Create figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=dates,
            y=historical_data,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color=self.forecast_colors['historical'], width=2),
            marker=dict(size=4),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
        ))
        
        # Forecast line
        if forecast_values:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.forecast_colors['forecast'], width=2, dash='dash'),
                marker=dict(size=4, symbol='diamond'),
                hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
            ))
        
        # Confidence intervals
        if 'conf_int' in forecast_result and forecast_result['conf_int'] is not None:
            conf_int = forecast_result['conf_int']
            if hasattr(conf_int, 'iloc'):
                lower_ci = conf_int.iloc[:, 0].tolist()
                upper_ci = conf_int.iloc[:, 1].tolist()
            elif isinstance(conf_int, dict):
                lower_ci = conf_int.get('lower', [])
                upper_ci = conf_int.get('upper', [])
            else:
                lower_ci = upper_ci = []
            
            if lower_ci and upper_ci and len(lower_ci) == len(forecast_dates):
                # Add confidence interval area
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=upper_ci + lower_ci[::-1],
                    fill='tonexty',
                    fillcolor=self.forecast_colors['confidence'],
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add confidence interval lines
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=upper_ci,
                    mode='lines',
                    name='Upper CI',
                    line=dict(color=self.forecast_colors['forecast'], width=1, dash='dot'),
                    showlegend=False,
                    hovertemplate='<b>Upper CI</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=lower_ci,
                    mode='lines',
                    name='Lower CI',
                    line=dict(color=self.forecast_colors['forecast'], width=1, dash='dot'),
                    showlegend=False,
                    hovertemplate='<b>Lower CI</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
                ))
        
        # FIXED: Add vertical line to separate historical and forecast
        # Convert datetime to string to avoid plotly datetime issues
        if dates and forecast_dates:
            try:
                # Use add_shape instead of add_vline for better datetime compatibility
                divider_x = dates[-1]
                
                fig.add_shape(
                    type="line",
                    x0=divider_x, x1=divider_x,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="gray", width=2, dash="dash"),
                    opacity=0.7
                )
                
                # Add annotation manually
                fig.add_annotation(
                    x=divider_x,
                    y=1.02,
                    yref="paper",
                    text="Forecast Start",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    xanchor="center"
                )
            except Exception as e:
                # If vertical line fails, just skip it
                print(f"Warning: Could not add divider line: {e}")
        
        # Layout configuration
        chart_title = title or f"ARIMA Forecast - {forecast_result.get('location', 'Unknown')} ({forecast_result.get('data_type', 'Data')})"
        
        fig.update_layout(
            title=dict(
                text=chart_title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            **self.default_layout
        )
        
        return fig
    
    def create_comparison_chart(self, data_dict: Dict[str, Dict[str, List]], 
                              chart_type: str = 'line',
                              title: str = "Comparison Analysis") -> go.Figure:
        """Create comparison chart for multiple data series"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (series_name, series_data) in enumerate(data_dict.items()):
            dates = series_data.get('dates', [])
            values = series_data.get('values', [])
            color = colors[i % len(colors)]
            
            if chart_type == 'line':
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines+markers',
                    name=series_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{series_name}</b><br>Date: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>'
                ))
            
            elif chart_type == 'bar':
                fig.add_trace(go.Bar(
                    x=dates,
                    y=values,
                    name=series_name,
                    marker_color=color,
                    hovertemplate=f'<b>{series_name}</b><br>Date: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Date",
            yaxis_title="Values",
            **self.default_layout
        )
        
        return fig
    
    def create_metrics_table(self, analysis_results: Dict[str, Any],
                           table_type: str = 'model_summary') -> pd.DataFrame:
        """Create summary table from analysis results"""
        if table_type == 'model_summary':
            return self._create_model_summary_table(analysis_results)
        elif table_type == 'forecast_summary':
            return self._create_forecast_summary_table(analysis_results)
        elif table_type == 'performance':
            return self._create_performance_table(analysis_results)
        else:
            return pd.DataFrame()
    
    def create_summary_table(self, analysis_results: Dict[str, Any],
                           table_type: str = 'model_summary') -> pd.DataFrame:
        """Create summary table - alias for create_metrics_table"""
        return self.create_metrics_table(analysis_results, table_type)
    
    def create_forecast_table(self, forecast_result: Dict[str, Any], 
                            dates: List[datetime]) -> pd.DataFrame:
        """Create forecast data table"""
        if 'error' in forecast_result:
            return pd.DataFrame({'Error': [forecast_result['error']]})
        
        if 'forecast' not in forecast_result:
            return pd.DataFrame({'Error': ['No forecast data available']})
        
        forecast_values = forecast_result['forecast']
        if isinstance(forecast_values, dict) and 'values' in forecast_values:
            forecast_values = forecast_values['values']
        
        if not forecast_values:
            return pd.DataFrame({'Error': ['Empty forecast data']})
        
        # Generate forecast dates
        if dates:
            last_date = dates[-1]
            forecast_dates = [
                last_date + timedelta(days=30*(i+1)) 
                for i in range(len(forecast_values))
            ]
        else:
            forecast_dates = [f"Period {i+1}" for i in range(len(forecast_values))]
        
        # Create forecast table
        table_data = {
            'Date': [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d) for d in forecast_dates],
            'Forecast': [f"{val:,.2f}" for val in forecast_values]
        }
        
        # Add confidence intervals if available
        if 'conf_int' in forecast_result and forecast_result['conf_int'] is not None:
            conf_int = forecast_result['conf_int']
            if hasattr(conf_int, 'iloc'):
                lower_ci = conf_int.iloc[:, 0].tolist()
                upper_ci = conf_int.iloc[:, 1].tolist()
                table_data['Lower CI'] = [f"{val:,.2f}" for val in lower_ci]
                table_data['Upper CI'] = [f"{val:,.2f}" for val in upper_ci]
        
        return pd.DataFrame(table_data)
    
    def _create_model_summary_table(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """Create model summary table"""
        summary_data = []
        
        for (location, data_type), result in analysis_results.items():
            if 'error' not in result:
                if 'model_selection' in result:
                    arima_order = result['model_selection']['selected_order']
                    aic = result['model_fit']['aic']
                    bic = result['model_fit']['bic']
                    rmse = result['validation'].get('rmse', None)
                    mape = result['validation'].get('mape', None)
                    is_stationary = result['stationarity']['overall']['is_stationary']
                else:
                    arima_order = result.get('arima_order', 'N/A')
                    aic = result.get('aic', 0)
                    bic = result.get('bic', 0)
                    rmse = result.get('rmse', None)
                    mape = result.get('mape', None)
                    is_stationary = result.get('is_stationary', False)
                
                summary_data.append({
                    'Location': location,
                    'Data Type': data_type,
                    'ARIMA Order': str(arima_order),
                    'AIC': f"{aic:.2f}" if aic else "N/A",
                    'BIC': f"{bic:.2f}" if bic else "N/A",
                    'RMSE': f"{rmse:.2f}" if rmse else "N/A",
                    'MAPE (%)': f"{mape:.1f}" if mape else "N/A",
                    'Stationary': "✅ Yes" if is_stationary else "❌ No"
                })
            else:
                summary_data.append({
                    'Location': location,
                    'Data Type': data_type,
                    'ARIMA Order': 'ERROR',
                    'AIC': 'ERROR',
                    'BIC': 'ERROR',
                    'RMSE': 'ERROR',
                    'MAPE (%)': 'ERROR',
                    'Stationary': 'ERROR'
                })
        
        return pd.DataFrame(summary_data)
    
    def _create_forecast_summary_table(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """Create forecast summary table"""
        forecast_data = []
        
        for (location, data_type), result in analysis_results.items():
            if 'error' not in result and 'forecast' in result:
                forecast_values = result['forecast']
                if isinstance(forecast_values, dict) and 'values' in forecast_values:
                    forecast_values = forecast_values['values']
                
                if forecast_values:
                    forecast_data.append({
                        'Location': location,
                        'Data Type': data_type,
                        'Forecast Mean': f"{np.mean(forecast_values):,.2f}",
                        'Forecast Std': f"{np.std(forecast_values):,.2f}",
                        'Min Forecast': f"{np.min(forecast_values):,.2f}",
                        'Max Forecast': f"{np.max(forecast_values):,.2f}",
                        'Growth Rate': f"{((forecast_values[-1] / forecast_values[0]) - 1) * 100:.1f}%" if len(forecast_values) > 1 else "N/A"
                    })
        
        return pd.DataFrame(forecast_data)
    
    def _create_performance_table(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """Create performance comparison table"""
        performance_data = []
        
        for (location, data_type), result in analysis_results.items():
            if 'error' not in result:
                if 'validation' in result:
                    validation = result['validation']
                    rmse = validation.get('rmse', None)
                    mape = validation.get('mape', None)
                    mae = validation.get('mae', None)
                    r_squared = validation.get('r_squared', None)
                else:
                    rmse = result.get('rmse', None)
                    mape = result.get('mape', None)
                    mae = None
                    r_squared = None
                
                if 'forecast_quality' in result:
                    quality = result['forecast_quality']['overall']
                else:
                    if mape is not None:
                        if mape < 5:
                            quality = 'Excellent'
                        elif mape < 10:
                            quality = 'Good'
                        elif mape < 20:
                            quality = 'Acceptable'
                        else:
                            quality = 'Poor'
                    else:
                        quality = 'Unknown'
                
                performance_data.append({
                    'Location': location,
                    'Data Type': data_type,
                    'RMSE': f"{rmse:.2f}" if rmse else "N/A",
                    'MAE': f"{mae:.2f}" if mae else "N/A",
                    'MAPE (%)': f"{mape:.1f}" if mape else "N/A",
                    'R²': f"{r_squared:.3f}" if r_squared else "N/A",
                    'Quality': quality
                })
        
        return pd.DataFrame(performance_data)

# Initialize global visualization engine instance
_viz_engine = VisualizationEngine()

# EXPLICIT FUNCTION EXPORTS - Fix for import errors
def create_forecast_chart(historical_data: List[float], 
                         forecast_result: Dict[str, Any], 
                         dates: List[datetime],
                         title: str = None,
                         y_axis_title: str = "Values") -> go.Figure:
    """
    Create ARIMA forecast visualization with confidence intervals
    EXPLICIT EXPORT FUNCTION for external imports
    """
    return _viz_engine.create_forecast_chart(historical_data, forecast_result, dates, title, y_axis_title)

def create_comparison_chart(data_dict: Dict[str, Dict[str, List]], 
                           chart_type: str = 'line',
                           title: str = "Comparison Analysis") -> go.Figure:
    """
    Create comparison chart for multiple data series
    EXPLICIT EXPORT FUNCTION for external imports
    """
    return _viz_engine.create_comparison_chart(data_dict, chart_type, title)

def create_metrics_table(analysis_results: Dict[str, Any],
                        table_type: str = 'model_summary') -> pd.DataFrame:
    """
    Create summary table from analysis results
    EXPLICIT EXPORT FUNCTION for external imports
    """
    return _viz_engine.create_metrics_table(analysis_results, table_type)

# Legacy compatibility functions
def create_summary_table(analysis_results: Dict[str, Any],
                        table_type: str = 'model_summary') -> pd.DataFrame:
    """Legacy compatibility function"""
    return create_metrics_table(analysis_results, table_type)

def quick_forecast_chart(historical_data: List[float], forecast_data: List[float], 
                        title: str = "Quick Forecast") -> go.Figure:
    """Quick forecast chart creation"""
    forecast_result = {
        'forecast': {'values': forecast_data},
        'location': 'Quick Analysis',
        'data_type': 'Data'
    }
    
    dates = [datetime.now() - timedelta(days=30*i) for i in range(len(historical_data))][::-1]
    
    return create_forecast_chart(historical_data, forecast_result, dates, title)

def create_summary_dashboard(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary dashboard from analysis results"""
    return {
        'model_summary': create_metrics_table(analysis_results, 'model_summary'),
        'forecast_summary': create_metrics_table(analysis_results, 'forecast_summary'),
        'performance_table': create_metrics_table(analysis_results, 'performance')
    }

# Export all functions for easy import
__all__ = [
    'VisualizationEngine',
    'create_forecast_chart',
    'create_comparison_chart', 
    'create_metrics_table',
    'create_summary_table',
    'quick_forecast_chart',
    'create_summary_dashboard'
]