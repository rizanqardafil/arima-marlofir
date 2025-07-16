"""
Configuration Settings - Centralized Config Management
/Users/sociolla/Documents/BBM/config/settings.py

Centralized configuration management for BBM Analysis Dashboard
Contains all constants, default values, and configuration parameters
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import os

class BBMConfig:
    """BBM Analysis Configuration Settings"""
    
    # ===================
    # APPLICATION SETTINGS
    # ===================
    
    APP_NAME = "BBM Analysis Dashboard"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Comprehensive BBM Consumption Forecasting with ARIMA Analysis"
    
    # Streamlit page configuration
    PAGE_CONFIG = {
        'page_title': APP_NAME,
        'page_icon': '⛽',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # ===================
    # DEFAULT LOCATIONS
    # ===================
    
    DEFAULT_LOCATIONS = [
        "Jemaja",
        "Siantan", 
        "Palmatak",
        "Kute Siantan",
        "Siantan Timur",
        "Bintan",
        "Batam",
        "Karimun",
        "Lingga",
        "Natuna"
    ]
    
    # Location metadata
    LOCATION_METADATA = {
        "Jemaja": {
            "type": "main_island",
            "population_category": "medium",
            "economic_activity": "fishing_tourism"
        },
        "Siantan": {
            "type": "sub_district", 
            "population_category": "small",
            "economic_activity": "fishing_agriculture"
        },
        "Palmatak": {
            "type": "industrial_area",
            "population_category": "medium", 
            "economic_activity": "industrial_port"
        },
        "Bintan": {
            "type": "main_island",
            "population_category": "large",
            "economic_activity": "tourism_industrial"
        },
        "Batam": {
            "type": "main_island", 
            "population_category": "large",
            "economic_activity": "industrial_port_urban"
        }
    }
    
    # ===================
    # BBM CONFIGURATION
    # ===================
    
    BBM_TYPES = {
        'BBM Tipe 1': {
            'name': 'Premium/Ron 92',
            'description': 'Gasoline for motorcycles and cars',
            'default_base_consumption': 5000,  # L/month
            'consumption_range': (1000, 50000),
            'default_variation': 20,  # ±%
            'variation_range': (5, 50),
            'seasonal_amplitude': 0.2,
            'peak_month': 7,  # July
            'growth_rate': 0.02,  # 2% per month
            'color': '#1f77b4'
        },
        'BBM Tipe 2': {
            'name': 'Solar/Diesel', 
            'description': 'Diesel for trucks, boats, and generators',
            'default_base_consumption': 10000,  # L/month
            'consumption_range': (2000, 100000),
            'default_variation': 25,  # ±%
            'variation_range': (5, 50),
            'seasonal_amplitude': 0.3,
            'peak_month': 6,  # June
            'growth_rate': 0.03,  # 3% per month
            'color': '#ff7f0e'
        }
    }
    
    # ===================
    # TRANSPORT MODE CONFIGURATION
    # ===================
    
    TRANSPORT_MODES = {
        '🚤 Kapal Nelayan': {
            'name': 'Fishing Boats',
            'category': 'marine',
            'base_consumption': 800,  # L/day/unit
            'consumption_range': (300, 2000),
            'efficiency': 0.8,
            'efficiency_range': (0.6, 1.2),
            'seasonal_amplitude': 0.25,
            'default_units': 15,
            'units_range': (0, 200),
            'default_variation': 20,
            'description': 'Traditional fishing boats - weather dependent operations',
            'fuel_type': 'diesel',
            'operational_pattern': 'weather_dependent'
        },
        '🏍️ Ojek Pangkalan': {
            'name': 'Motorcycle Taxis',
            'category': 'land_transport',
            'base_consumption': 150,  # L/day/unit
            'consumption_range': (50, 400),
            'efficiency': 1.2,
            'efficiency_range': (1.0, 1.5),
            'seasonal_amplitude': 0.15,
            'default_units': 50,
            'units_range': (0, 500),
            'default_variation': 15,
            'description': 'Motorcycle taxis for urban and inter-village transport',
            'fuel_type': 'gasoline',
            'operational_pattern': 'daily_consistent'
        },
        '🚗 Mobil Pribadi': {
            'name': 'Private Cars',
            'category': 'land_transport',
            'base_consumption': 200,  # L/day/unit
            'consumption_range': (100, 500),
            'efficiency': 1.0,
            'efficiency_range': (0.8, 1.3),
            'seasonal_amplitude': 0.10,
            'default_units': 30,
            'units_range': (0, 300),
            'default_variation': 12,
            'description': 'Private cars for personal transportation',
            'fuel_type': 'gasoline',
            'operational_pattern': 'weekend_holiday_peaks'
        },
        '🚛 Truck Angkutan': {
            'name': 'Cargo Trucks',
            'category': 'land_transport',
            'base_consumption': 500,  # L/day/unit
            'consumption_range': (200, 1200),
            'efficiency': 0.7,
            'efficiency_range': (0.5, 1.0),
            'seasonal_amplitude': 0.20,
            'default_units': 20,
            'units_range': (0, 150),
            'default_variation': 18,
            'description': 'Cargo trucks for goods transportation',
            'fuel_type': 'diesel',
            'operational_pattern': 'business_hours'
        },
        '⛵ Kapal Penumpang': {
            'name': 'Passenger Boats',
            'category': 'marine',
            'base_consumption': 1200,  # L/day/unit
            'consumption_range': (500, 3000),
            'efficiency': 0.6,
            'efficiency_range': (0.4, 0.9),
            'seasonal_amplitude': 0.30,
            'default_units': 8,
            'units_range': (0, 50),
            'default_variation': 25,
            'description': 'Passenger boats for inter-island transportation',
            'fuel_type': 'diesel',
            'operational_pattern': 'schedule_weather_dependent'
        },
        '🏭 Generator/Mesin': {
            'name': 'Generators & Engines',
            'category': 'stationary',
            'base_consumption': 300,  # L/day/unit
            'consumption_range': (100, 800),
            'efficiency': 0.9,
            'efficiency_range': (0.7, 1.1),
            'seasonal_amplitude': 0.05,
            'default_units': 25,
            'units_range': (0, 200),
            'default_variation': 10,
            'description': 'Stationary generators and industrial engines',
            'fuel_type': 'diesel',
            'operational_pattern': 'continuous_industrial'
        }
    }
    
    # Transport categories for filtering
    TRANSPORT_CATEGORIES = {
        'marine': ['🚤 Kapal Nelayan', '⛵ Kapal Penumpang'],
        'land_transport': ['🏍️ Ojek Pangkalan', '🚗 Mobil Pribadi', '🚛 Truck Angkutan'],
        'stationary': ['🏭 Generator/Mesin']
    }
    
    # ===================
    # VEHICLE CONFIGURATION
    # ===================
    
    VEHICLE_TYPES = {
        'Kendaraan Air': {
            'description': 'Water vehicles (boats, ships)',
            'default_count': 100,
            'count_range': (10, 2000),
            'growth_rate': 0.015,
            'volatility': 0.20,
            'seasonal_amplitude': 0.25
        },
        'Roda Dua': {
            'description': 'Two-wheeled vehicles (motorcycles)',
            'default_count': 500,
            'count_range': (50, 3000),
            'growth_rate': 0.008,
            'volatility': 0.12,
            'seasonal_amplitude': 0.10
        },
        'Roda Tiga': {
            'description': 'Three-wheeled vehicles (bajaj, becak motor)',
            'default_count': 200,
            'count_range': (20, 1000),
            'growth_rate': 0.012,
            'volatility': 0.15,
            'seasonal_amplitude': 0.15
        },
        'Roda Empat': {
            'description': 'Four-wheeled vehicles (cars, pickup trucks)',
            'default_count': 300,
            'count_range': (30, 2000),
            'growth_rate': 0.010,
            'volatility': 0.13,
            'seasonal_amplitude': 0.12
        },
        'Roda Lima': {
            'description': 'Five+ wheeled vehicles (trucks, buses)',
            'default_count': 50,
            'count_range': (5, 500),
            'growth_rate': 0.020,
            'volatility': 0.25,
            'seasonal_amplitude': 0.20
        },
        'Alat Berat': {
            'description': 'Heavy machinery (excavators, bulldozers)',
            'default_count': 25,
            'count_range': (5, 200),
            'growth_rate': 0.025,
            'volatility': 0.30,
            'seasonal_amplitude': 0.30
        }
    }
    
    # ===================
    # WAVE CONFIGURATION
    # ===================
    
    WAVE_LOCATIONS = {
        'Pantai Utara': {
            'description': 'Northern coastal area',
            'base_height': 1.5,  # meters
            'height_range': (0.5, 5.0),
            'storm_frequency': 0.15,
            'seasonal_pattern': 'monsoon_north',
            'peak_months': [10, 11, 12]  # Oct-Dec
        },
        'Pantai Selatan': {
            'description': 'Southern coastal area',
            'base_height': 2.0,  # meters
            'height_range': (0.8, 6.0),
            'storm_frequency': 0.20,
            'seasonal_pattern': 'monsoon_south',
            'peak_months': [6, 7, 8]  # Jun-Aug
        },
        'Pantai Timur': {
            'description': 'Eastern coastal area',
            'base_height': 1.8,  # meters
            'height_range': (0.6, 5.5),
            'storm_frequency': 0.18,
            'seasonal_pattern': 'trade_winds',
            'peak_months': [8, 9, 10]  # Aug-Oct
        },
        'Pantai Barat': {
            'description': 'Western coastal area',
            'base_height': 1.6,  # meters
            'height_range': (0.5, 4.5),
            'storm_frequency': 0.12,
            'seasonal_pattern': 'sheltered',
            'peak_months': [12, 1, 2]  # Dec-Feb
        }
    }
    
    # ===================
    # ARIMA CONFIGURATION
    # ===================
    
    ARIMA_CONFIG = {
        'default_max_p': 3,
        'default_max_d': 2,
        'default_max_q': 3,
        'max_p_range': (1, 10),
        'max_d_range': (0, 3),
        'max_q_range': (1, 10),
        'information_criteria': ['AIC', 'BIC', 'HQIC'],
        'default_criterion': 'AIC',
        'min_data_points': 8,
        'recommended_data_points': 12,
        'train_test_split_ratio': 0.8,
        'seasonal_period': 12,  # Monthly seasonality
        'stationarity_alpha': 0.05
    }
    
    # Model quality thresholds
    MODEL_QUALITY_THRESHOLDS = {
        'mape': {
            'excellent': 5.0,
            'good': 10.0,
            'acceptable': 20.0,
            'poor': float('inf')
        },
        'rmse_relative': {
            'excellent': 0.05,  # 5% of mean
            'good': 0.10,       # 10% of mean
            'acceptable': 0.20,  # 20% of mean
            'poor': float('inf')
        }
    }
    
    # ===================
    # TIME CONFIGURATION
    # ===================
    
    TIME_CONFIG = {
        'default_start_date': datetime(2023, 1, 1),
        'min_historical_months': 8,
        'max_historical_months': 36,
        'default_historical_months': 12,
        'min_forecast_months': 3,
        'max_forecast_months': 60,
        'default_forecast_months': 12,
        'days_per_month': 30.44  # Average days per month
    }
    
    # ===================
    # VISUALIZATION CONFIGURATION
    # ===================
    
    VISUALIZATION_CONFIG = {
        'themes': ['plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white'],
        'default_theme': 'plotly_white',
        'color_palettes': {
            'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'forecast': {
                'historical': '#1f77b4',
                'forecast': '#ff7f0e',
                'confidence': 'rgba(255, 127, 14, 0.2)',
                'trend': '#2ca02c'
            }
        },
        'chart_defaults': {
            'height': 500,
            'width': None,  # Auto-width
            'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},
            'font_size': 12,
            'title_font_size': 16
        },
        'export_formats': ['HTML', 'PNG', 'SVG', 'PDF'],
        'export_dimensions': {
            'width': 1200,
            'height': 600,
            'dpi': 300
        }
    }
    
    # ===================
    # FILE PATHS & STORAGE
    # ===================
    
    PATHS = {
        'core': 'core/',
        'pages': 'pages/',
        'config': 'config/',
        'exports': 'exports/',
        'logs': 'logs/',
        'temp': 'temp/'
    }
    
    # File naming conventions
    FILE_NAMING = {
        'bbm_data': 'bbm_data_{location}_{timestamp}.csv',
        'transport_data': 'transport_data_{location}_{timestamp}.csv',
        'forecast_results': 'forecast_{location}_{data_type}_{timestamp}.csv',
        'model_summary': 'model_summary_{timestamp}.csv',
        'analysis_report': 'analysis_report_{timestamp}.md',
        'charts': 'chart_{chart_type}_{timestamp}.{format}'
    }
    
    # ===================
    # ANALYSIS SETTINGS
    # ===================
    
    ANALYSIS_SETTINGS = {
        'min_units_threshold': 5,
        'efficiency_thresholds': {
            'excellent': 1.2,
            'good': 1.0,
            'average': 0.8,
            'poor': 0.0
        },
        'growth_rate_thresholds': {
            'high_growth': 0.05,      # >5% monthly
            'moderate_growth': 0.02,   # 2-5% monthly
            'stable': -0.02,          # ±2% monthly
            'declining': float('-inf') # <-2% monthly
        },
        'variability_thresholds': {
            'low': 10,     # CV < 10%
            'moderate': 25, # CV 10-25%
            'high': 50,    # CV 25-50%
            'very_high': float('inf')  # CV > 50%
        }
    }
    
    # ===================
    # UI CONFIGURATION
    # ===================
    
    UI_CONFIG = {
        'sidebar_width': 320,
        'main_content_padding': 20,
        'metric_columns': 4,
        'chart_columns': 2,
        'table_height': 400,
        'expander_defaults': {
            'expanded': False
        },
        'button_styles': {
            'primary': {'use_container_width': True, 'type': 'primary'},
            'secondary': {'use_container_width': True, 'type': 'secondary'}
        },
        'progress_update_interval': 0.1,  # seconds
        'spinner_messages': {
            'generating_data': '🔄 Generating realistic data with seasonal patterns...',
            'running_analysis': '🔬 Running ARIMA analysis... This may take a few moments...',
            'creating_charts': '📈 Creating interactive visualizations...',
            'exporting_data': '📥 Preparing data for export...'
        }
    }
    
    # ===================
    # VALIDATION RULES
    # ===================
    
    VALIDATION_RULES = {
        'locations': {
            'min_count': 1,
            'max_count': 10,
            'name_max_length': 50,
            'name_pattern': r'^[a-zA-Z0-9\s\-_]+$'
        },
        'consumption': {
            'min_value': 0,
            'max_value': 1000000,  # 1M liters
            'variation_min': 0,
            'variation_max': 100
        },
        'units': {
            'min_value': 0,
            'max_value': 10000
        },
        'time_periods': {
            'min_months': 3,
            'max_months': 120  # 10 years
        }
    }
    
    # ===================
    # ERROR MESSAGES
    # ===================
    
    ERROR_MESSAGES = {
        'insufficient_data': "⚠️ Insufficient data for ARIMA analysis. Need at least {min_points} data points.",
        'no_data_generated': "❌ No data available. Please generate data first.",
        'analysis_failed': "❌ ARIMA analysis failed: {error_detail}",
        'invalid_location': "❌ Invalid location name. Use alphanumeric characters only.",
        'invalid_consumption': "❌ Consumption value must be between {min_val} and {max_val}.",
        'export_failed': "❌ Export failed: {error_detail}",
        'file_not_found': "❌ File not found: {filename}",
        'configuration_error': "❌ Configuration error: {error_detail}"
    }
    
    # Success messages
    SUCCESS_MESSAGES = {
        'data_generated': "✅ Generated {data_type} data for {count} locations over {months} months",
        'analysis_completed': "✅ ARIMA analysis completed! Analyzed {count} time series successfully.",
        'export_completed': "✅ {export_type} export completed successfully!",
        'chart_created': "✅ {chart_type} chart created successfully!"
    }
    
    # ===================
    # LOGGING CONFIGURATION
    # ===================
    
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': 'logs/bbm_analysis.log',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'log_categories': [
            'data_generation',
            'arima_analysis', 
            'visualization',
            'export',
            'error',
            'user_action'
        ]
    }

class ConfigManager:
    """Configuration Manager for runtime configuration management"""
    
    def __init__(self):
        self.config = BBMConfig()
        self.user_overrides = {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'ARIMA_CONFIG.default_max_p')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check user overrides first
        if key_path in self.user_overrides:
            return self.user_overrides[key_path]
        
        # Navigate through config object
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = getattr(value, key)
            return value
        except AttributeError:
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value (user override)
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        self.user_overrides[key_path] = value
    
    def reset(self, key_path: str = None) -> None:
        """
        Reset configuration to defaults
        
        Args:
            key_path: Specific key to reset, or None for all
        """
        if key_path:
            self.user_overrides.pop(key_path, None)
        else:
            self.user_overrides.clear()
    
    def get_transport_mode_config(self, mode_name: str) -> Dict[str, Any]:
        """Get specific transport mode configuration"""
        return self.config.TRANSPORT_MODES.get(mode_name, {})
    
    def get_bbm_type_config(self, bbm_type: str) -> Dict[str, Any]:
        """Get specific BBM type configuration"""
        return self.config.BBM_TYPES.get(bbm_type, {})
    
    def get_location_metadata(self, location: str) -> Dict[str, Any]:
        """Get specific location metadata"""
        return self.config.LOCATION_METADATA.get(location, {})
    
    def validate_configuration(self) -> List[str]:
        """
        Validate current configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate ARIMA config
        if self.get('ARIMA_CONFIG.default_max_p') < 1:
            errors.append("ARIMA max_p must be at least 1")
        
        # Validate time config
        if self.get('TIME_CONFIG.min_historical_months') < 3:
            errors.append("Minimum historical months must be at least 3")
        
        # Validate transport modes
        for mode_name, config in self.config.TRANSPORT_MODES.items():
            if config['base_consumption'] <= 0:
                errors.append(f"Transport mode {mode_name} must have positive base consumption")
        
        return errors
    
    def export_config(self, file_path: str = None) -> str:
        """
        Export current configuration to JSON file
        
        Args:
            file_path: Optional file path
            
        Returns:
            File path of exported config
        """
        import json
        from datetime import datetime
        
        if not file_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"config_export_{timestamp}.json"
        
        # Prepare config data (exclude non-serializable items)
        config_data = {
            'app_info': {
                'name': self.config.APP_NAME,
                'version': self.config.APP_VERSION,
                'export_timestamp': datetime.now().isoformat()
            },
            'user_overrides': self.user_overrides,
            'transport_modes': self.config.TRANSPORT_MODES,
            'bbm_types': self.config.BBM_TYPES,
            'arima_config': self.config.ARIMA_CONFIG,
            'time_config': {k: v for k, v in self.config.TIME_CONFIG.items() 
                          if k != 'default_start_date'},
            'validation_rules': self.config.VALIDATION_RULES
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        return file_path

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions for easy access
def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get(key_path, default)

def set_config(key_path: str, value: Any) -> None:
    """Set configuration value"""
    config_manager.set(key_path, value)

def get_transport_modes() -> Dict[str, Any]:
    """Get all transport mode configurations"""
    return config_manager.config.TRANSPORT_MODES

def get_bbm_types() -> Dict[str, Any]:
    """Get all BBM type configurations"""
    return config_manager.config.BBM_TYPES

def get_default_locations() -> List[str]:
    """Get default location list"""
    return config_manager.config.DEFAULT_LOCATIONS

def get_arima_defaults() -> Dict[str, Any]:
    """Get ARIMA default configuration"""
    return config_manager.config.ARIMA_CONFIG

def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration"""
    return config_manager.config.VISUALIZATION_CONFIG

# Environment-specific configurations
def load_environment_config(env: str = 'development') -> None:
    """
    Load environment-specific configuration
    
    Args:
        env: Environment name ('development', 'staging', 'production')
    """
    if env == 'development':
        set_config('LOGGING_CONFIG.level', 'DEBUG')
        set_config('UI_CONFIG.expander_defaults.expanded', True)
    
    elif env == 'staging':
        set_config('LOGGING_CONFIG.level', 'INFO')
        set_config('ARIMA_CONFIG.default_max_p', 2)  # Faster analysis
    
    elif env == 'production':
        set_config('LOGGING_CONFIG.level', 'WARNING')
        set_config('VALIDATION_RULES.consumption.max_value', 500000)  # More conservative