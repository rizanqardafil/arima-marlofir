"""
Data Generator - Realistic Time Series Generation
/Users/sociolla/Documents/BBM/core/data_generator.py

Comprehensive data generation engine for BBM analysis
Generates realistic time series data with seasonal patterns, trends, and noise
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import random
import math

class DataGenerator:
    """
    Advanced data generation engine for BBM and related time series data
    
    Features:
    - Seasonal pattern generation
    - Trend injection
    - Realistic noise simulation
    - Multiple data types support
    - Configurable parameters
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Data Generator
        
        Args:
            random_seed: Optional seed for reproducible results
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Configuration constants
        self.transport_modes = {
            '🚤 Kapal Nelayan': {'base_consumption': 800, 'efficiency': 0.8, 'seasonal_amplitude': 0.25},
            '🏍️ Ojek Pangkalan': {'base_consumption': 150, 'efficiency': 1.2, 'seasonal_amplitude': 0.15},
            '🚗 Mobil Pribadi': {'base_consumption': 200, 'efficiency': 1.0, 'seasonal_amplitude': 0.10},
            '🚛 Truck Angkutan': {'base_consumption': 500, 'efficiency': 0.7, 'seasonal_amplitude': 0.20},
            '⛵ Kapal Penumpang': {'base_consumption': 1200, 'efficiency': 0.6, 'seasonal_amplitude': 0.30},
            '🏭 Generator/Mesin': {'base_consumption': 300, 'efficiency': 0.9, 'seasonal_amplitude': 0.05}
        }
        
        self.vehicle_types = {
            'Kendaraan Air': {'growth_rate': 0.015, 'volatility': 0.20},
            'Roda Dua': {'growth_rate': 0.008, 'volatility': 0.12},
            'Roda Tiga': {'growth_rate': 0.012, 'volatility': 0.15},
            'Roda Empat': {'growth_rate': 0.010, 'volatility': 0.13},
            'Roda Lima': {'growth_rate': 0.020, 'volatility': 0.25},
            'Alat Berat': {'growth_rate': 0.025, 'volatility': 0.30}
        }
        
        self.wave_locations = {
            'Pantai Utara': {'base_height': 1.5, 'storm_frequency': 0.15},
            'Pantai Selatan': {'base_height': 2.0, 'storm_frequency': 0.20},
            'Pantai Timur': {'base_height': 1.8, 'storm_frequency': 0.18},
            'Pantai Barat': {'base_height': 1.6, 'storm_frequency': 0.12}
        }
    
    def generate_seasonal_pattern(self, length: int, amplitude: float = 0.2, 
                                 peak_month: int = 6, pattern_type: str = 'sine') -> np.ndarray:
        """
        Generate seasonal pattern for time series
        
        Args:
            length: Number of time points
            amplitude: Seasonal amplitude (0-1)
            peak_month: Month when pattern peaks (1-12)
            pattern_type: 'sine', 'cosine', or 'mixed'
            
        Returns:
            Seasonal pattern array
        """
        time_points = np.arange(length)
        
        if pattern_type == 'sine':
            # Standard sine wave with phase shift for peak month
            phase_shift = (peak_month - 1) * (2 * np.pi / 12)
            seasonal = amplitude * np.sin(2 * np.pi * time_points / 12 + phase_shift)
        
        elif pattern_type == 'cosine':
            # Cosine wave pattern
            phase_shift = (peak_month - 1) * (2 * np.pi / 12)
            seasonal = amplitude * np.cos(2 * np.pi * time_points / 12 + phase_shift)
        
        elif pattern_type == 'mixed':
            # Mixed pattern with multiple harmonics
            primary = amplitude * np.sin(2 * np.pi * time_points / 12)
            secondary = (amplitude * 0.3) * np.sin(4 * np.pi * time_points / 12)
            seasonal = primary + secondary
        
        else:
            # Default to sine
            seasonal = amplitude * np.sin(2 * np.pi * time_points / 12)
        
        return seasonal
    
    def generate_trend_pattern(self, length: int, trend_type: str = 'linear', 
                              growth_rate: float = 0.02) -> np.ndarray:
        """
        Generate trend pattern for time series
        
        Args:
            length: Number of time points
            trend_type: 'linear', 'exponential', 'polynomial', or 'none'
            growth_rate: Monthly growth rate
            
        Returns:
            Trend pattern array
        """
        time_points = np.arange(length)
        
        if trend_type == 'linear':
            # Linear trend
            trend = time_points * growth_rate
        
        elif trend_type == 'exponential':
            # Exponential growth/decay
            trend = np.power(1 + growth_rate, time_points) - 1
        
        elif trend_type == 'polynomial':
            # Polynomial trend (quadratic)
            trend = growth_rate * time_points + (growth_rate * 0.1) * np.power(time_points, 2) / length
        
        elif trend_type == 'logarithmic':
            # Logarithmic growth (slowing over time)
            trend = growth_rate * np.log1p(time_points)
        
        else:
            # No trend
            trend = np.zeros(length)
        
        return trend
    
    def generate_noise(self, length: int, noise_type: str = 'gaussian', 
                      volatility: float = 0.1) -> np.ndarray:
        """
        Generate noise component for time series
        
        Args:
            length: Number of time points
            noise_type: 'gaussian', 'uniform', 'laplace', or 'mixed'
            volatility: Noise volatility (standard deviation)
            
        Returns:
            Noise array
        """
        if noise_type == 'gaussian':
            # Gaussian/normal noise
            noise = np.random.normal(0, volatility, length)
        
        elif noise_type == 'uniform':
            # Uniform noise
            bound = volatility * np.sqrt(3)  # Match variance with gaussian
            noise = np.random.uniform(-bound, bound, length)
        
        elif noise_type == 'laplace':
            # Laplace noise (heavier tails)
            scale = volatility / np.sqrt(2)
            noise = np.random.laplace(0, scale, length)
        
        elif noise_type == 'mixed':
            # Mixed noise (combination of gaussian and occasional spikes)
            base_noise = np.random.normal(0, volatility * 0.8, length)
            spikes = np.random.choice([0, 1], size=length, p=[0.95, 0.05])
            spike_values = np.random.normal(0, volatility * 3, length)
            noise = base_noise + spikes * spike_values
        
        else:
            # Default to gaussian
            noise = np.random.normal(0, volatility, length)
        
        return noise
    
    def generate_bbm_data(self, locations: List[str], num_months: int, 
                         parameters: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """
        Generate BBM consumption data for multiple locations
        
        Args:
            locations: List of location names
            num_months: Number of months to generate
            parameters: Configuration parameters for each location
            
        Returns:
            Dictionary with BBM data for each location
        """
        bbm_data = {}
        
        for loc in locations:
            # Set location-specific seed for reproducibility
            if self.random_seed:
                np.random.seed(hash(loc) % 10000)
            
            # Get parameters
            base_t1 = parameters.get(f"{loc}_base_t1", 5000)
            var_t1 = parameters.get(f"{loc}_var_t1", 20) / 100
            base_t2 = parameters.get(f"{loc}_base_t2", 10000)
            var_t2 = parameters.get(f"{loc}_var_t2", 25) / 100
            
            # BBM Tipe 1 generation
            seasonal_t1 = self.generate_seasonal_pattern(
                num_months, amplitude=0.2, peak_month=7, pattern_type='sine'
            )
            trend_t1 = self.generate_trend_pattern(
                num_months, trend_type='linear', growth_rate=0.02
            )
            noise_t1 = self.generate_noise(
                num_months, noise_type='gaussian', volatility=var_t1
            )
            
            t1_multiplier = 1 + seasonal_t1 + trend_t1 + noise_t1
            t1_data = (base_t1 * t1_multiplier).clip(min=base_t1 * 0.3).tolist()
            
            # BBM Tipe 2 generation (different characteristics)
            seasonal_t2 = self.generate_seasonal_pattern(
                num_months, amplitude=0.3, peak_month=6, pattern_type='mixed'
            )
            trend_t2 = self.generate_trend_pattern(
                num_months, trend_type='exponential', growth_rate=0.025
            )
            noise_t2 = self.generate_noise(
                num_months, noise_type='mixed', volatility=var_t2
            )
            
            t2_multiplier = 1 + seasonal_t2 + trend_t2 + noise_t2
            t2_data = (base_t2 * t2_multiplier).clip(min=base_t2 * 0.4).tolist()
            
            bbm_data[loc] = {
                'bbm_tipe_1': t1_data,
                'bbm_tipe_2': t2_data
            }
        
        return bbm_data
    
    def generate_vehicle_data(self, num_months: int, 
                            parameters: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Generate vehicle count data
        
        Args:
            num_months: Number of months to generate
            parameters: Configuration parameters for each vehicle type
            
        Returns:
            Dictionary with vehicle count data
        """
        vehicle_data = {}
        
        for i, (vehicle_type, config) in enumerate(self.vehicle_types.items()):
            # Set vehicle-specific seed
            if self.random_seed:
                np.random.seed(self.random_seed + i + 100)
            
            # Get parameters
            base_count = parameters.get(f"vehicle_{i}_base", [100, 500, 200, 300, 50, 25][i])
            var_pct = parameters.get(f"vehicle_{i}_var", 15) / 100
            
            # Generate patterns
            seasonal = self.generate_seasonal_pattern(
                num_months, amplitude=0.1, peak_month=8, pattern_type='sine'
            )
            trend = self.generate_trend_pattern(
                num_months, trend_type='linear', growth_rate=config['growth_rate']
            )
            noise = self.generate_noise(
                num_months, noise_type='gaussian', volatility=var_pct
            )
            
            # Combine patterns
            multiplier = 1 + seasonal + trend + noise
            vehicle_counts = (base_count * multiplier).clip(min=base_count * 0.5)
            
            # Convert to integers and ensure positive
            vehicle_data[vehicle_type] = [max(1, int(count)) for count in vehicle_counts]
        
        return vehicle_data
    
    def generate_wave_data(self, num_months: int, 
                          parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Generate wave height data
        
        Args:
            num_months: Number of months to generate
            parameters: Configuration parameters for each wave location
            
        Returns:
            Dictionary with wave height data
        """
        wave_data = {}
        
        for i, (location, config) in enumerate(self.wave_locations.items()):
            # Set location-specific seed
            if self.random_seed:
                np.random.seed(self.random_seed + i + 200)
            
            # Get parameters
            base_height = parameters.get(f"wave_{i}_base", config['base_height'])
            var_pct = parameters.get(f"wave_{i}_var", 30) / 100
            
            # Generate monsoon seasonal pattern (higher waves in certain months)
            seasonal = self.generate_seasonal_pattern(
                num_months, amplitude=0.4, peak_month=10, pattern_type='mixed'
            )
            
            # Add storm events (random spikes)
            storm_events = np.random.choice(
                [0, 1], size=num_months, 
                p=[1 - config['storm_frequency'], config['storm_frequency']]
            )
            storm_intensity = np.random.exponential(0.5, num_months) * storm_events
            
            # Base noise
            noise = self.generate_noise(
                num_months, noise_type='laplace', volatility=var_pct
            )
            
            # Combine all components
            wave_multiplier = 1 + seasonal + noise
            wave_heights = base_height * wave_multiplier + storm_intensity
            
            # Ensure realistic wave heights (min 0.2m, max 8m)
            wave_heights = np.clip(wave_heights, 0.2, 8.0)
            
            wave_data[f"Wave_{location}"] = wave_heights.tolist()
        
        return wave_data
    
    def calculate_transport_consumption(self, unit_count: int, mode_name: str, 
                                      base_consumption: float, days_in_month: int = 30) -> float:
        """
        Calculate BBM consumption for transport mode
        
        Args:
            unit_count: Number of transport units
            mode_name: Transport mode name
            base_consumption: Base consumption per unit per day
            days_in_month: Days in the month
            
        Returns:
            Monthly BBM consumption
        """
        if mode_name not in self.transport_modes:
            return 0.0
        
        mode_config = self.transport_modes[mode_name]
        efficiency_factor = mode_config['efficiency']
        
        daily_consumption = base_consumption * efficiency_factor
        monthly_consumption = unit_count * daily_consumption * days_in_month
        
        return monthly_consumption
    
    def generate_transport_data(self, locations: List[str], num_months: int, 
                               parameters: Dict[str, Any], 
                               dates: List[datetime]) -> Dict[str, Dict[str, Any]]:
        """
        Generate transport mode BBM consumption data
        
        Args:
            locations: List of location names
            num_months: Number of months to generate
            parameters: Configuration parameters
            dates: List of dates for the time series
            
        Returns:
            Dictionary with transport data for each location
        """
        transport_data = {}
        
        for loc in locations:
            transport_data[loc] = {'dates': dates, 'modes': {}}
            
            for mode_name, mode_config in self.transport_modes.items():
                # Get parameters
                unit_count = parameters.get(f"{loc}_{mode_name}_units", 0)
                base_cons = parameters.get(f"{loc}_{mode_name}_base", mode_config['base_consumption'])
                variation = parameters.get(f"{loc}_{mode_name}_var", 15) / 100
                
                if unit_count > 0:
                    # Set mode-specific seed
                    if self.random_seed:
                        np.random.seed(hash(f"{loc}_{mode_name}") % 10000)
                    
                    # Generate patterns specific to transport mode
                    seasonal_amplitude = mode_config.get('seasonal_amplitude', 0.15)
                    seasonal = self.generate_seasonal_pattern(
                        num_months, amplitude=seasonal_amplitude, peak_month=7
                    )
                    
                    trend = self.generate_trend_pattern(
                        num_months, trend_type='linear', growth_rate=0.01
                    )
                    
                    noise = self.generate_noise(
                        num_months, noise_type='gaussian', volatility=variation
                    )
                    
                    # Calculate base consumption for each month
                    monthly_consumption = []
                    for i in range(num_months):
                        # Days in month (approximate)
                        days = 30 if i % 12 not in [1] else 28  # Feb approximation
                        
                        base_monthly = self.calculate_transport_consumption(
                            unit_count, mode_name, base_cons, days
                        )
                        
                        # Apply patterns
                        multiplier = 1 + seasonal[i] + trend[i] + noise[i]
                        final_consumption = base_monthly * multiplier
                        
                        # Ensure positive consumption
                        monthly_consumption.append(max(final_consumption, base_monthly * 0.3))
                    
                    transport_data[loc]['modes'][mode_name] = monthly_consumption
        
        return transport_data
    
    def generate_custom_series(self, length: int, base_value: float, 
                              seasonal_params: Dict[str, Any] = None,
                              trend_params: Dict[str, Any] = None,
                              noise_params: Dict[str, Any] = None) -> List[float]:
        """
        Generate custom time series with specified parameters
        
        Args:
            length: Number of time points
            base_value: Base value for the series
            seasonal_params: Seasonal pattern parameters
            trend_params: Trend pattern parameters  
            noise_params: Noise parameters
            
        Returns:
            Generated time series
        """
        # Default parameters
        seasonal_params = seasonal_params or {'amplitude': 0.2, 'peak_month': 6}
        trend_params = trend_params or {'type': 'linear', 'growth_rate': 0.01}
        noise_params = noise_params or {'type': 'gaussian', 'volatility': 0.1}
        
        # Generate components
        seasonal = self.generate_seasonal_pattern(
            length, 
            amplitude=seasonal_params.get('amplitude', 0.2),
            peak_month=seasonal_params.get('peak_month', 6),
            pattern_type=seasonal_params.get('pattern_type', 'sine')
        )
        
        trend = self.generate_trend_pattern(
            length,
            trend_type=trend_params.get('type', 'linear'),
            growth_rate=trend_params.get('growth_rate', 0.01)
        )
        
        noise = self.generate_noise(
            length,
            noise_type=noise_params.get('type', 'gaussian'),
            volatility=noise_params.get('volatility', 0.1)
        )
        
        # Combine components
        multiplier = 1 + seasonal + trend + noise
        series = (base_value * multiplier).clip(min=base_value * 0.1)
        
        return series.tolist()
    
    def generate_correlated_series(self, reference_series: List[float], 
                                  correlation: float = 0.7,
                                  noise_level: float = 0.1) -> List[float]:
        """
        Generate time series correlated with reference series
        
        Args:
            reference_series: Reference time series
            correlation: Desired correlation coefficient (-1 to 1)
            noise_level: Additional noise level
            
        Returns:
            Correlated time series
        """
        reference = np.array(reference_series)
        length = len(reference)
        
        # Generate correlated noise
        independent_noise = np.random.normal(0, 1, length)
        correlated_component = correlation * (reference - np.mean(reference)) / np.std(reference)
        independent_component = np.sqrt(1 - correlation**2) * independent_noise
        
        # Add additional noise
        additional_noise = self.generate_noise(length, volatility=noise_level)
        
        # Combine components
        correlated_series = correlated_component + independent_component + additional_noise
        
        # Scale to similar range as reference
        correlated_series = correlated_series * np.std(reference) + np.mean(reference)
        
        # Ensure positive values
        correlated_series = np.maximum(correlated_series, np.mean(reference) * 0.1)
        
        return correlated_series.tolist()
    
    def validate_generated_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate generated data for quality and realism
        
        Args:
            data: Generated data dictionary
            
        Returns:
            Validation report
        """
        validation_report = {
            'total_series': 0,
            'valid_series': 0,
            'issues': [],
            'statistics': {}
        }
        
        def validate_series(series: List[float], series_name: str) -> Dict[str, Any]:
            """Validate individual series"""
            series_array = np.array(series)
            
            issues = []
            stats = {
                'mean': np.mean(series_array),
                'std': np.std(series_array),
                'min': np.min(series_array),
                'max': np.max(series_array),
                'cv': np.std(series_array) / np.mean(series_array) if np.mean(series_array) != 0 else float('inf')
            }
            
            # Check for issues
            if np.any(series_array <= 0):
                issues.append(f"{series_name}: Contains non-positive values")
            
            if stats['cv'] > 2.0:
                issues.append(f"{series_name}: Very high coefficient of variation ({stats['cv']:.2f})")
            
            if np.any(np.isnan(series_array)) or np.any(np.isinf(series_array)):
                issues.append(f"{series_name}: Contains NaN or infinite values")
            
            return stats, issues
        
        # Validate different data types
        for data_type, data_content in data.items():
            if isinstance(data_content, dict):
                for key, series in data_content.items():
                    if isinstance(series, list) and len(series) > 0:
                        validation_report['total_series'] += 1
                        stats, issues = validate_series(series, f"{data_type}.{key}")
                        validation_report['statistics'][f"{data_type}.{key}"] = stats
                        validation_report['issues'].extend(issues)
                        
                        if not issues:
                            validation_report['valid_series'] += 1
        
        validation_report['success_rate'] = (
            validation_report['valid_series'] / validation_report['total_series']
            if validation_report['total_series'] > 0 else 0
        )
        
        return validation_report

# Utility functions for external use
def quick_data_generation(data_type: str, length: int = 12, **kwargs) -> List[float]:
    """
    Quick data generation for simple use cases
    
    Args:
        data_type: Type of data ('bbm', 'vehicle', 'wave', 'transport')
        length: Number of data points
        **kwargs: Additional parameters
        
    Returns:
        Generated data series
    """
    generator = DataGenerator()
    
    if data_type.lower() == 'bbm':
        base_value = kwargs.get('base_value', 5000)
        return generator.generate_custom_series(length, base_value)
    
    elif data_type.lower() == 'vehicle':
        base_value = kwargs.get('base_value', 100)
        seasonal_params = {'amplitude': 0.1, 'peak_month': 8}
        return [int(x) for x in generator.generate_custom_series(length, base_value, seasonal_params)]
    
    elif data_type.lower() == 'wave':
        base_value = kwargs.get('base_value', 1.8)
        seasonal_params = {'amplitude': 0.4, 'peak_month': 10}
        noise_params = {'type': 'laplace', 'volatility': 0.3}
        return generator.generate_custom_series(length, base_value, seasonal_params, None, noise_params)
    
    else:
        # Default
        base_value = kwargs.get('base_value', 1000)
        return generator.generate_custom_series(length, base_value)

def generate_date_range(start_date: datetime, num_months: int) -> List[datetime]:
    """
    Generate date range for time series
    
    Args:
        start_date: Starting date
        num_months: Number of months
        
    Returns:
        List of datetime objects
    """
    dates = []
    current_date = start_date
    
    for _ in range(num_months):
        dates.append(current_date)
        # Add approximately one month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return dates