"""
ARIMA Engine - Core Time Series Analysis
/Users/sociolla/Documents/BBM/core/arima_engine.py

Comprehensive ARIMA analysis engine for BBM forecasting
Includes stationarity testing, model selection, validation, and forecasting
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

# Statistical analysis libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ARIMAEngine:
    """
    Comprehensive ARIMA analysis engine with advanced time series capabilities
    
    Features:
    - Stationarity testing (ADF, KPSS)
    - Automatic ARIMA order selection
    - Model validation and diagnostics
    - Forecasting with confidence intervals
    - Performance metrics calculation
    """
    
    def __init__(self, max_p: int = 3, max_d: int = 2, max_q: int = 3):
        """
        Initialize ARIMA Engine with search parameters
        
        Args:
            max_p: Maximum AR order to test
            max_d: Maximum differencing order to test  
            max_q: Maximum MA order to test
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        
        # Results storage
        self.analysis_results = {}
        self.model_diagnostics = {}
    
    def test_stationarity(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Comprehensive stationarity testing using multiple methods
        
        Args:
            series: Time series data
            alpha: Significance level for tests
            
        Returns:
            Dictionary with stationarity test results
        """
        results = {}
        
        try:
            # Augmented Dickey-Fuller Test
            adf_result = adfuller(series, autolag='AIC')
            results['adf'] = {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] <= alpha,
                'interpretation': 'Stationary' if adf_result[1] <= alpha else 'Non-stationary'
            }
            
            # KPSS Test (null hypothesis: series is stationary)
            try:
                kpss_result = kpss(series, regression='c', nlags='auto')
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'pvalue': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > alpha,  # Note: reversed logic for KPSS
                    'interpretation': 'Stationary' if kpss_result[1] > alpha else 'Non-stationary'
                }
            except Exception as e:
                results['kpss'] = {'error': f"KPSS test failed: {str(e)}"}
            
            # Overall assessment
            adf_stationary = results['adf']['is_stationary']
            kpss_stationary = results.get('kpss', {}).get('is_stationary', adf_stationary)
            
            results['overall'] = {
                'is_stationary': adf_stationary and kpss_stationary,
                'confidence': 'High' if adf_stationary == kpss_stationary else 'Medium',
                'recommendation': self._get_stationarity_recommendation(adf_stationary, kpss_stationary)
            }
            
        except Exception as e:
            results['error'] = f"Stationarity testing failed: {str(e)}"
        
        return results
    
    def _get_stationarity_recommendation(self, adf_stationary: bool, kpss_stationary: bool) -> str:
        """Get recommendation based on stationarity tests"""
        if adf_stationary and kpss_stationary:
            return "Series is stationary - proceed with ARIMA modeling"
        elif not adf_stationary and not kpss_stationary:
            return "Series is non-stationary - consider differencing"
        elif adf_stationary and not kpss_stationary:
            return "Mixed signals - series may be trend-stationary"
        else:
            return "Mixed signals - series may have structural breaks"
    
    def find_best_arima_order(self, series: pd.Series, 
                            information_criterion: str = 'aic') -> Tuple[Tuple[int, int, int], float]:
        """
        Find optimal ARIMA order using grid search
        
        Args:
            series: Time series data
            information_criterion: 'aic', 'bic', or 'hqic'
            
        Returns:
            Tuple of (best_order, best_criterion_value)
        """
        best_criterion = float('inf')
        best_order = None
        search_results = []
        
        # Grid search over (p,d,q) combinations
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        # Skip if no AR or MA terms and no differencing
                        if p == 0 and q == 0 and d == 0:
                            continue
                            
                        # Fit ARIMA model
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        # Get information criterion
                        if information_criterion.lower() == 'aic':
                            criterion_value = fitted_model.aic
                        elif information_criterion.lower() == 'bic':
                            criterion_value = fitted_model.bic
                        elif information_criterion.lower() == 'hqic':
                            criterion_value = fitted_model.hqic
                        else:
                            criterion_value = fitted_model.aic
                        
                        # Store results
                        search_results.append({
                            'order': (p, d, q),
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'hqic': fitted_model.hqic,
                            'selected_criterion': criterion_value
                        })
                        
                        # Update best model
                        if criterion_value < best_criterion:
                            best_criterion = criterion_value
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        # Skip problematic model combinations
                        continue
        
        # Store search results for analysis
        self.model_diagnostics['search_results'] = search_results
        
        return best_order if best_order else (1, 1, 1), best_criterion
    
    def validate_model(self, series: pd.Series, order: Tuple[int, int, int], 
                      train_ratio: float = 0.8) -> Dict[str, float]:
        """
        Validate ARIMA model using train-test split
        
        Args:
            series: Time series data
            order: ARIMA order (p,d,q)
            train_ratio: Proportion of data for training
            
        Returns:
            Dictionary with validation metrics
        """
        if len(series) < 8:
            return {'error': 'Insufficient data for validation (need at least 8 points)'}
        
        try:
            # Split data
            split_point = int(len(series) * train_ratio)
            train_data = series[:split_point]
            test_data = series[split_point:]
            
            if len(test_data) == 0:
                return {'error': 'No test data available'}
            
            # Fit model on training data
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # Generate forecasts
            n_forecast = len(test_data)
            forecast = fitted_model.forecast(steps=n_forecast)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mae = mean_absolute_error(test_data, forecast)
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            # Additional metrics
            mse = mean_squared_error(test_data, forecast)
            
            # Calculate R-squared equivalent for time series
            ss_res = np.sum((test_data - forecast) ** 2)
            ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'mse': mse,
                'r_squared': r_squared,
                'train_size': len(train_data),
                'test_size': len(test_data)
            }
            
        except Exception as e:
            return {'error': f"Validation failed: {str(e)}"}
    
    def run_diagnostic_tests(self, fitted_model) -> Dict[str, Any]:
        """
        Run comprehensive diagnostic tests on fitted ARIMA model
        
        Args:
            fitted_model: Fitted ARIMA model
            
        Returns:
            Dictionary with diagnostic test results
        """
        diagnostics = {}
        
        try:
            # Ljung-Box test for residual autocorrelation
            residuals = fitted_model.resid
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            diagnostics['ljung_box'] = {
                'statistic': ljung_box['lb_stat'].iloc[-1],
                'pvalue': ljung_box['lb_pvalue'].iloc[-1],
                'is_white_noise': ljung_box['lb_pvalue'].iloc[-1] > 0.05,
                'interpretation': 'Residuals are white noise' if ljung_box['lb_pvalue'].iloc[-1] > 0.05 else 'Residuals show autocorrelation'
            }
            
            # Residual statistics
            diagnostics['residuals'] = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': pd.Series(residuals).skew(),
                'kurtosis': pd.Series(residuals).kurtosis(),
                'jarque_bera_pvalue': None  # Could add Jarque-Bera test
            }
            
            # Model information
            diagnostics['model_info'] = {
                'log_likelihood': fitted_model.llf,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'n_observations': fitted_model.nobs
            }
            
        except Exception as e:
            diagnostics['error'] = f"Diagnostic tests failed: {str(e)}"
        
        return diagnostics
    
    def run_arima_analysis(self, series: pd.Series, location: str, data_type: str, 
                          forecast_periods: int = 12) -> Dict[str, Any]:
        """
        Run complete ARIMA analysis pipeline
        
        Args:
            series: Time series data
            location: Location identifier
            data_type: Type of data being analyzed
            forecast_periods: Number of periods to forecast
            
        Returns:
            Comprehensive analysis results
        """
        analysis_id = f"{location}_{data_type}"
        
        try:
            # Step 1: Stationarity testing
            stationarity_results = self.test_stationarity(series)
            
            # Step 2: Find best ARIMA order
            best_order, best_aic = self.find_best_arima_order(series)
            
            # Step 3: Fit final model
            model = ARIMA(series, order=best_order)
            fitted_model = model.fit()
            
            # Step 4: Model validation
            validation_results = self.validate_model(series, best_order)
            
            # Step 5: Diagnostic tests
            diagnostic_results = self.run_diagnostic_tests(fitted_model)
            
            # Step 6: Generate forecasts
            forecast = fitted_model.forecast(steps=forecast_periods)
            forecast_ci = fitted_model.get_forecast(steps=forecast_periods).conf_int()
            
            # Step 7: Calculate forecast quality metrics
            forecast_quality = self._assess_forecast_quality(fitted_model, validation_results)
            
            # Compile comprehensive results
            results = {
                'metadata': {
                    'location': location,
                    'data_type': data_type,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'series_length': len(series),
                    'forecast_periods': forecast_periods
                },
                'stationarity': stationarity_results,
                'model_selection': {
                    'selected_order': best_order,
                    'selection_criterion': 'AIC',
                    'selected_criterion_value': best_aic,
                    'search_space': f"p≤{self.max_p}, d≤{self.max_d}, q≤{self.max_q}"
                },
                'model_fit': {
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'hqic': fitted_model.hqic,
                    'log_likelihood': fitted_model.llf
                },
                'validation': validation_results,
                'diagnostics': diagnostic_results,
                'forecast': {
                    'values': forecast.tolist(),
                    'confidence_intervals': {
                        'lower': forecast_ci.iloc[:, 0].tolist(),
                        'upper': forecast_ci.iloc[:, 1].tolist()
                    }
                },
                'forecast_quality': forecast_quality,
                'model_object': fitted_model  # For advanced usage
            }
            
            # Store results
            self.analysis_results[analysis_id] = results
            
            return results
            
        except Exception as e:
            error_result = {
                'metadata': {
                    'location': location,
                    'data_type': data_type,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'series_length': len(series) if series is not None else 0
                },
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            self.analysis_results[analysis_id] = error_result
            return error_result
    
    def _assess_forecast_quality(self, fitted_model, validation_results: Dict) -> Dict[str, str]:
        """
        Assess overall forecast quality based on multiple criteria
        
        Args:
            fitted_model: Fitted ARIMA model
            validation_results: Validation metrics
            
        Returns:
            Quality assessment dictionary
        """
        quality = {}
        
        try:
            # MAPE-based quality assessment
            if 'mape' in validation_results and validation_results['mape'] is not None:
                mape = validation_results['mape']
                if mape < 5:
                    quality['accuracy'] = 'Excellent'
                elif mape < 10:
                    quality['accuracy'] = 'Good'
                elif mape < 20:
                    quality['accuracy'] = 'Acceptable'
                else:
                    quality['accuracy'] = 'Poor'
            else:
                quality['accuracy'] = 'Unknown'
            
            # AIC-based model complexity assessment
            aic = fitted_model.aic
            if aic < 100:
                quality['model_fit'] = 'Excellent'
            elif aic < 200:
                quality['model_fit'] = 'Good'
            elif aic < 500:
                quality['model_fit'] = 'Acceptable'
            else:
                quality['model_fit'] = 'Poor'
            
            # Overall assessment
            accuracy_scores = {'Excellent': 4, 'Good': 3, 'Acceptable': 2, 'Poor': 1, 'Unknown': 2}
            fit_scores = {'Excellent': 4, 'Good': 3, 'Acceptable': 2, 'Poor': 1}
            
            avg_score = (accuracy_scores[quality['accuracy']] + fit_scores[quality['model_fit']]) / 2
            
            if avg_score >= 3.5:
                quality['overall'] = 'Excellent'
            elif avg_score >= 2.5:
                quality['overall'] = 'Good'
            elif avg_score >= 1.5:
                quality['overall'] = 'Acceptable'
            else:
                quality['overall'] = 'Poor'
                
        except Exception:
            quality = {
                'accuracy': 'Unknown',
                'model_fit': 'Unknown', 
                'overall': 'Unknown'
            }
        
        return quality
    
    def get_analysis_summary(self) -> pd.DataFrame:
        """
        Get summary of all completed analyses
        
        Returns:
            DataFrame with analysis summary
        """
        summary_data = []
        
        for analysis_id, results in self.analysis_results.items():
            if 'error' not in results:
                row = {
                    'Analysis ID': analysis_id,
                    'Location': results['metadata']['location'],
                    'Data Type': results['metadata']['data_type'],
                    'ARIMA Order': str(results['model_selection']['selected_order']),
                    'AIC': f"{results['model_fit']['aic']:.2f}",
                    'Validation RMSE': f"{results['validation'].get('rmse', 0):.2f}" if 'rmse' in results['validation'] else 'N/A',
                    'Validation MAPE': f"{results['validation'].get('mape', 0):.1f}%" if 'mape' in results['validation'] else 'N/A',
                    'Forecast Quality': results['forecast_quality']['overall'],
                    'Is Stationary': results['stationarity']['overall']['is_stationary']
                }
            else:
                row = {
                    'Analysis ID': analysis_id,
                    'Location': results['metadata']['location'],
                    'Data Type': results['metadata']['data_type'],
                    'ARIMA Order': 'ERROR',
                    'AIC': 'ERROR',
                    'Validation RMSE': 'ERROR',
                    'Validation MAPE': 'ERROR',
                    'Forecast Quality': 'ERROR',
                    'Is Stationary': 'ERROR'
                }
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def export_results(self, filepath: str = None) -> str:
        """
        Export analysis results to JSON file
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Filepath of exported file
        """
        import json
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"arima_analysis_results_{timestamp}.json"
        
        # Prepare results for JSON serialization (remove model objects)
        exportable_results = {}
        for analysis_id, results in self.analysis_results.items():
            exportable_results[analysis_id] = results.copy()
            if 'model_object' in exportable_results[analysis_id]:
                del exportable_results[analysis_id]['model_object']
        
        with open(filepath, 'w') as f:
            json.dump(exportable_results, f, indent=2, default=str)
        
        return filepath

# Utility functions for external use
def quick_arima_forecast(data: list, forecast_periods: int = 12) -> Dict[str, Any]:
    """
    Quick ARIMA forecast for simple use cases
    
    Args:
        data: List of numerical values
        forecast_periods: Number of periods to forecast
        
    Returns:
        Simple forecast results
    """
    engine = ARIMAEngine()
    series = pd.Series(data)
    results = engine.run_arima_analysis(series, "QuickAnalysis", "Data", forecast_periods)
    
    if 'error' not in results:
        return {
            'forecast': results['forecast']['values'],
            'order': results['model_selection']['selected_order'],
            'quality': results['forecast_quality']['overall']
        }
    else:
        return {'error': results['error']}

def batch_arima_analysis(data_dict: Dict[str, list], forecast_periods: int = 12) -> Dict[str, Any]:
    """
    Run ARIMA analysis on multiple time series
    
    Args:
        data_dict: Dictionary with {name: data_list} pairs
        forecast_periods: Number of periods to forecast
        
    Returns:
        Dictionary with all analysis results
    """
    engine = ARIMAEngine()
    all_results = {}
    
    for name, data in data_dict.items():
        series = pd.Series(data)
        results = engine.run_arima_analysis(series, "Batch", name, forecast_periods)
        all_results[name] = results
    
    return all_results