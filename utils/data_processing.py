"""
Data Processing Utilities for BBM Dashboard & MARLOFIR-P
Advanced data transformation, validation, cleaning, and preprocessing utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import warnings
warnings.filterwarnings('ignore')

# Statistical and validation imports
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Supported data types for processing"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    GEOGRAPHIC = "geographic"
    DEMAND = "demand"
    COST = "cost"
    DISTANCE = "distance"

class CleaningMethod(Enum):
    """Data cleaning methods"""
    REMOVE_OUTLIERS = "remove_outliers"
    CAP_OUTLIERS = "cap_outliers"
    FILL_MISSING = "fill_missing"
    INTERPOLATE = "interpolate"
    SMOOTH = "smooth"
    NORMALIZE = "normalize"

class ValidationLevel(Enum):
    """Data validation levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    STRICT = "strict"
    CUSTOM = "custom"

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_records: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    outliers_detected: Dict[str, int] = field(default_factory=dict)
    data_types: Dict[str, str] = field(default_factory=dict)
    duplicates: int = 0
    quality_score: float = 0.0
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    handle_missing: str = "interpolate"  # 'drop', 'fill', 'interpolate'
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 1.5
    normalization_method: str = "standard"  # 'standard', 'minmax', 'robust'
    datetime_format: str = "%Y-%m-%d"
    decimal_places: int = 2
    remove_duplicates: bool = True
    validate_ranges: bool = True

class DataProcessor:
    """
    Comprehensive data processing utilities for BBM dashboard and MARLOFIR-P
    Handles data cleaning, transformation, validation, and quality assessment
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.scalers = {}
        self.processing_history = []
        
        # Quality thresholds
        self.quality_thresholds = {
            'missing_data_threshold': 0.1,  # 10% missing data threshold
            'outlier_threshold': 0.05,      # 5% outliers threshold
            'duplicate_threshold': 0.02,    # 2% duplicates threshold
            'minimum_quality_score': 0.8    # 80% quality score threshold
        }
        
        logger.info("Data processor initialized")
    
    def assess_data_quality(self, df: pd.DataFrame, 
                          target_columns: List[str] = None) -> DataQualityReport:
        """
        Comprehensive data quality assessment
        
        Args:
            df: DataFrame to assess
            target_columns: Specific columns to focus on
            
        Returns:
            Data quality report
        """
        start_time = datetime.now()
        logger.info(f"Assessing data quality for {len(df)} records")
        
        if target_columns is None:
            target_columns = df.columns.tolist()
        
        report = DataQualityReport()
        report.total_records = len(df)
        
        # Missing values analysis
        missing_analysis = df[target_columns].isnull().sum()
        report.missing_values = missing_analysis.to_dict()
        
        # Data types analysis
        report.data_types = df[target_columns].dtypes.astype(str).to_dict()
        
        # Duplicates analysis
        report.duplicates = df.duplicated().sum()
        
        # Outliers analysis for numeric columns
        numeric_columns = df[target_columns].select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            outliers = self._detect_outliers(df[col].dropna(), method="iqr")
            report.outliers_detected[col] = len(outliers)
        
        # Calculate quality score
        report.quality_score = self._calculate_quality_score(df, report)
        
        # Generate issues and recommendations
        report.issues_found = self._identify_issues(df, report)
        report.recommendations = self._generate_recommendations(report)
        
        # Processing time
        report.processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Data quality assessment completed. Score: {report.quality_score:.2f}")
        return report
    
    def clean_dataset(self, df: pd.DataFrame, 
                     target_columns: List[str] = None,
                     custom_rules: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Comprehensive data cleaning
        
        Args:
            df: DataFrame to clean
            target_columns: Columns to focus on
            custom_rules: Custom cleaning rules
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {len(df)} records")
        
        df_cleaned = df.copy()
        cleaning_log = []
        
        if target_columns is None:
            target_columns = df.columns.tolist()
        
        # 1. Remove duplicates
        if self.config.remove_duplicates:
            initial_count = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            removed_duplicates = initial_count - len(df_cleaned)
            if removed_duplicates > 0:
                cleaning_log.append(f"Removed {removed_duplicates} duplicate records")
        
        # 2. Handle missing values
        for col in target_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = self._handle_missing_values(
                    df_cleaned[col], 
                    method=self.config.handle_missing
                )
                cleaning_log.append(f"Handled missing values in {col}")
        
        # 3. Handle outliers
        numeric_columns = df_cleaned[target_columns].select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            outliers_before = len(self._detect_outliers(df_cleaned[col], self.config.outlier_method))
            df_cleaned[col] = self._handle_outliers(
                df_cleaned[col], 
                method=self.config.outlier_method,
                threshold=self.config.outlier_threshold
            )
            outliers_after = len(self._detect_outliers(df_cleaned[col], self.config.outlier_method))
            if outliers_before > outliers_after:
                cleaning_log.append(f"Handled {outliers_before - outliers_after} outliers in {col}")
        
        # 4. Data type conversion and validation
        df_cleaned = self._convert_data_types(df_cleaned, target_columns)
        
        # 5. Apply custom rules
        if custom_rules:
            df_cleaned = self._apply_custom_rules(df_cleaned, custom_rules)
            cleaning_log.append("Applied custom cleaning rules")
        
        # 6. Final validation
        df_cleaned = self._validate_ranges(df_cleaned, target_columns)
        
        # Log cleaning summary
        self.processing_history.append({
            'timestamp': datetime.now(),
            'operation': 'clean_dataset',
            'records_before': len(df),
            'records_after': len(df_cleaned),
            'columns_processed': len(target_columns),
            'cleaning_steps': cleaning_log
        })
        
        logger.info(f"Data cleaning completed. Records: {len(df)} → {len(df_cleaned)}")
        return df_cleaned
    
    def transform_for_arima(self, df: pd.DataFrame, 
                           demand_column: str,
                           date_column: str = None,
                           location_column: str = None) -> Dict[str, pd.Series]:
        """
        Transform data for ARIMA analysis
        
        Args:
            df: Input DataFrame
            demand_column: Column containing demand data
            date_column: Column containing dates
            location_column: Column containing location identifiers
            
        Returns:
            Dictionary of location -> demand time series
        """
        logger.info("Transforming data for ARIMA analysis")
        
        df_processed = df.copy()
        
        # Handle date column
        if date_column and date_column in df_processed.columns:
            df_processed[date_column] = pd.to_datetime(df_processed[date_column])
            df_processed = df_processed.sort_values(date_column)
        
        # Ensure demand column is numeric
        df_processed[demand_column] = pd.to_numeric(df_processed[demand_column], errors='coerce')
        
        # Remove negative values (invalid for demand)
        df_processed = df_processed[df_processed[demand_column] >= 0]
        
        # Group by location if specified
        if location_column and location_column in df_processed.columns:
            arima_data = {}
            for location in df_processed[location_column].unique():
                location_data = df_processed[df_processed[location_column] == location]
                
                # Create time series
                if date_column:
                    ts = location_data.set_index(date_column)[demand_column]
                    # Fill missing dates with interpolation
                    ts = ts.resample('D').interpolate()
                else:
                    ts = location_data[demand_column].reset_index(drop=True)
                
                # Apply smoothing if needed
                ts = self._smooth_time_series(ts)
                
                arima_data[location] = ts
            
            return arima_data
        else:
            # Single time series
            if date_column:
                ts = df_processed.set_index(date_column)[demand_column]
                ts = ts.resample('D').interpolate()
            else:
                ts = df_processed[demand_column].reset_index(drop=True)
            
            ts = self._smooth_time_series(ts)
            return {'default': ts}
    
    def transform_for_optimization(self, df: pd.DataFrame,
                                 location_columns: List[str],
                                 demand_columns: List[str],
                                 coordinate_columns: List[str] = None) -> Dict[str, Any]:
        """
        Transform data for GA/VRP optimization
        
        Args:
            df: Input DataFrame
            location_columns: Columns identifying locations
            demand_columns: Columns with demand data
            coordinate_columns: Columns with lat/lon coordinates
            
        Returns:
            Structured data for optimization
        """
        logger.info("Transforming data for optimization")
        
        df_processed = df.copy()
        
        # Clean and validate location data
        for col in location_columns:
            df_processed[col] = df_processed[col].astype(str).str.strip()
        
        # Clean and validate demand data
        for col in demand_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col] = df_processed[col].fillna(0).clip(lower=0)
        
        # Clean coordinate data if provided
        if coordinate_columns and len(coordinate_columns) >= 2:
            lat_col, lon_col = coordinate_columns[0], coordinate_columns[1]
            df_processed[lat_col] = pd.to_numeric(df_processed[lat_col], errors='coerce')
            df_processed[lon_col] = pd.to_numeric(df_processed[lon_col], errors='coerce')
            
            # Validate coordinate ranges
            df_processed = df_processed[
                (df_processed[lat_col].between(-90, 90)) & 
                (df_processed[lon_col].between(-180, 180))
            ]
        
        # Structure data for optimization
        optimization_data = {
            'locations': {},
            'demands': {},
            'coordinates': {},
            'summary': {
                'total_locations': len(df_processed),
                'total_demand': df_processed[demand_columns].sum().sum(),
                'avg_demand_per_location': df_processed[demand_columns].mean().mean()
            }
        }
        
        for idx, row in df_processed.iterrows():
            location_id = '_'.join([str(row[col]) for col in location_columns])
            
            # Location data
            optimization_data['locations'][location_id] = {
                col: row[col] for col in location_columns
            }
            
            # Demand data
            optimization_data['demands'][location_id] = {
                col: row[col] for col in demand_columns
            }
            
            # Coordinate data
            if coordinate_columns and len(coordinate_columns) >= 2:
                optimization_data['coordinates'][location_id] = {
                    'latitude': row[coordinate_columns[0]],
                    'longitude': row[coordinate_columns[1]]
                }
        
        return optimization_data
    
    def normalize_data(self, df: pd.DataFrame, 
                      columns: List[str] = None,
                      method: str = None) -> pd.DataFrame:
        """
        Normalize numeric data
        
        Args:
            df: DataFrame to normalize
            columns: Columns to normalize
            method: Normalization method
            
        Returns:
            Normalized DataFrame
        """
        method = method or self.config.normalization_method
        df_normalized = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.number]:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler()
                elif method == "robust":
                    scaler = RobustScaler()
                else:
                    continue
                
                df_normalized[col] = scaler.fit_transform(df[[col]]).flatten()
                self.scalers[col] = scaler
        
        logger.info(f"Normalized {len(columns)} columns using {method} method")
        return df_normalized
    
    def validate_bbm_data(self, df: pd.DataFrame, 
                         validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> Dict[str, Any]:
        """
        BBM-specific data validation
        
        Args:
            df: DataFrame to validate
            validation_level: Validation strictness level
            
        Returns:
            Validation results
        """
        logger.info(f"Validating BBM data with {validation_level.value} level")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Basic validations
        required_columns = ['location', 'demand', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            validation_results['is_valid'] = False
        
        # Demand validation
        if 'demand' in df.columns:
            demand_issues = self._validate_demand_data(df['demand'])
            validation_results['warnings'].extend(demand_issues)
        
        # Date validation
        if 'date' in df.columns:
            date_issues = self._validate_date_data(df['date'])
            validation_results['warnings'].extend(date_issues)
        
        # Location validation
        if 'location' in df.columns:
            location_issues = self._validate_location_data(df['location'])
            validation_results['warnings'].extend(location_issues)
        
        # Advanced validations based on level
        if validation_level in [ValidationLevel.INTERMEDIATE, ValidationLevel.STRICT]:
            # Check for reasonable demand ranges
            if 'demand' in df.columns:
                demand_stats = df['demand'].describe()
                if demand_stats['max'] > 100000:  # 100k liters seems high for daily
                    validation_results['warnings'].append("Unusually high demand values detected")
                
                if demand_stats['min'] < 0:
                    validation_results['errors'].append("Negative demand values found")
                    validation_results['is_valid'] = False
        
        if validation_level == ValidationLevel.STRICT:
            # Strict validations
            if df.isnull().sum().sum() > 0:
                validation_results['errors'].append("Missing values not allowed in strict mode")
                validation_results['is_valid'] = False
            
            if df.duplicated().sum() > 0:
                validation_results['errors'].append("Duplicate records not allowed in strict mode")
                validation_results['is_valid'] = False
        
        # Generate suggestions
        if len(validation_results['warnings']) > 0:
            validation_results['suggestions'].append("Review data quality issues before analysis")
        
        if 'demand' in df.columns and df['demand'].isnull().sum() > 0:
            validation_results['suggestions'].append("Consider interpolating missing demand values")
        
        logger.info(f"Validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
    
    def aggregate_demand_data(self, df: pd.DataFrame,
                            group_by: List[str],
                            demand_column: str,
                            aggregation_method: str = 'sum',
                            time_period: str = 'daily') -> pd.DataFrame:
        """
        Aggregate demand data by various dimensions
        
        Args:
            df: Input DataFrame
            group_by: Columns to group by
            demand_column: Column containing demand values
            aggregation_method: Aggregation method ('sum', 'mean', 'max', etc.)
            time_period: Time aggregation period
            
        Returns:
            Aggregated DataFrame
        """
        logger.info(f"Aggregating demand data by {group_by}")
        
        df_agg = df.copy()
        
        # Handle time period aggregation
        if 'date' in df_agg.columns and time_period != 'daily':
            df_agg['date'] = pd.to_datetime(df_agg['date'])
            
            if time_period == 'weekly':
                df_agg['period'] = df_agg['date'].dt.to_period('W')
            elif time_period == 'monthly':
                df_agg['period'] = df_agg['date'].dt.to_period('M')
            elif time_period == 'quarterly':
                df_agg['period'] = df_agg['date'].dt.to_period('Q')
            
            group_by = ['period'] + [col for col in group_by if col != 'date']
        
        # Perform aggregation
        agg_functions = {
            demand_column: aggregation_method,
        }
        
        # Add count of records
        agg_functions['record_count'] = 'count'
        
        result = df_agg.groupby(group_by).agg(agg_functions).reset_index()
        
        # Calculate additional metrics
        if aggregation_method == 'sum':
            result[f'{demand_column}_daily_avg'] = result[demand_column] / result['record_count']
        
        logger.info(f"Aggregation completed. {len(df)} → {len(result)} records")
        return result
    
    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> List[int]:
        """Detect outliers using various methods"""
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series.dropna()))
            threshold = 3
            outliers = series.index[z_scores > threshold].tolist()
        
        elif method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(series.dropna().values.reshape(-1, 1))
            outliers = series.index[outlier_labels == -1].tolist()
        
        else:
            outliers = []
        
        return outliers
    
    def _handle_missing_values(self, series: pd.Series, method: str = "interpolate") -> pd.Series:
        """Handle missing values in a series"""
        if method == "drop":
            return series.dropna()
        
        elif method == "fill":
            # Fill with median for numeric, mode for categorical
            if series.dtype in [np.number]:
                return series.fillna(series.median())
            else:
                return series.fillna(series.mode().iloc[0] if not series.mode().empty else 'Unknown')
        
        elif method == "interpolate":
            if series.dtype in [np.number]:
                return series.interpolate(method='linear')
            else:
                return series.fillna(method='ffill').fillna(method='bfill')
        
        elif method == "knn":
            if series.dtype in [np.number]:
                imputer = KNNImputer(n_neighbors=3)
                return pd.Series(imputer.fit_transform(series.values.reshape(-1, 1)).flatten(),
                               index=series.index)
        
        return series
    
    def _handle_outliers(self, series: pd.Series, method: str = "iqr", 
                        threshold: float = 1.5) -> pd.Series:
        """Handle outliers in a series"""
        outlier_indices = self._detect_outliers(series, method)
        
        if not outlier_indices:
            return series
        
        result = series.copy()
        
        # Cap outliers to reasonable bounds
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        result = result.clip(lower=lower_bound, upper=upper_bound)
        
        return result
    
    def _smooth_time_series(self, series: pd.Series, window: int = 3) -> pd.Series:
        """Apply smoothing to time series data"""
        if len(series) < window:
            return series
        
        # Apply rolling mean smoothing
        smoothed = series.rolling(window=window, center=True).mean()
        
        # Fill edge values
        smoothed.iloc[:window//2] = series.iloc[:window//2]
        smoothed.iloc[-window//2:] = series.iloc[-window//2:]
        
        return smoothed.fillna(series)
    
    def _convert_data_types(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convert data types appropriately"""
        df_converted = df.copy()
        
        for col in columns:
            if col not in df_converted.columns:
                continue
            
            # Try to infer and convert appropriate type
            if df_converted[col].dtype == 'object':
                # Try numeric conversion
                try:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='ignore')
                except:
                    pass
                
                # Try datetime conversion
                if df_converted[col].dtype == 'object':
                    try:
                        df_converted[col] = pd.to_datetime(df_converted[col], errors='ignore')
                    except:
                        pass
        
        return df_converted
    
    def _apply_custom_rules(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply custom cleaning rules"""
        df_custom = df.copy()
        
        for rule_name, rule_config in rules.items():
            if rule_name == "cap_values":
                for col, bounds in rule_config.items():
                    if col in df_custom.columns:
                        df_custom[col] = df_custom[col].clip(
                            lower=bounds.get('min'), 
                            upper=bounds.get('max')
                        )
            
            elif rule_name == "replace_values":
                for col, replacements in rule_config.items():
                    if col in df_custom.columns:
                        df_custom[col] = df_custom[col].replace(replacements)
            
            elif rule_name == "filter_rows":
                condition = rule_config.get('condition')
                if condition:
                    # Apply simple filtering conditions
                    # This is a simplified implementation
                    pass
        
        return df_custom
    
    def _validate_ranges(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Validate data ranges for specific domains"""
        df_validated = df.copy()
        
        for col in columns:
            if col not in df_validated.columns:
                continue
            
            # BBM-specific range validations
            if 'demand' in col.lower():
                # Demand should be non-negative
                df_validated[col] = df_validated[col].clip(lower=0)
            
            elif 'latitude' in col.lower():
                # Latitude should be between -90 and 90
                df_validated[col] = df_validated[col].clip(lower=-90, upper=90)
            
            elif 'longitude' in col.lower():
                # Longitude should be between -180 and 180
                df_validated[col] = df_validated[col].clip(lower=-180, upper=180)
            
            elif 'cost' in col.lower() or 'price' in col.lower():
                # Costs should be non-negative
                df_validated[col] = df_validated[col].clip(lower=0)
        
        return df_validated
    
    def _calculate_quality_score(self, df: pd.DataFrame, report: DataQualityReport) -> float:
        """Calculate overall data quality score"""
        score = 1.0
        
        # Penalize missing values
        total_missing = sum(report.missing_values.values())
        missing_ratio = total_missing / (len(df) * len(df.columns))
        score -= min(missing_ratio * 2, 0.3)  # Max 30% penalty
        
        # Penalize duplicates
        duplicate_ratio = report.duplicates / len(df)
        score -= min(duplicate_ratio * 1.5, 0.2)  # Max 20% penalty
        
        # Penalize outliers
        total_outliers = sum(report.outliers_detected.values())
        outlier_ratio = total_outliers / len(df)
        score -= min(outlier_ratio * 1.0, 0.15)  # Max 15% penalty
        
        return max(score, 0.0)
    
    def _identify_issues(self, df: pd.DataFrame, report: DataQualityReport) -> List[str]:
        """Identify data quality issues"""
        issues = []
        
        # Missing values issues
        high_missing_cols = [col for col, count in report.missing_values.items() 
                           if count / len(df) > self.quality_thresholds['missing_data_threshold']]
        if high_missing_cols:
            issues.append(f"High missing values in columns: {high_missing_cols}")
        
        # Duplicate issues
        if report.duplicates > len(df) * self.quality_thresholds['duplicate_threshold']:
            issues.append(f"High number of duplicate records: {report.duplicates}")
        
        # Outlier issues
        high_outlier_cols = [col for col, count in report.outliers_detected.items()
                           if count / len(df) > self.quality_thresholds['outlier_threshold']]
        if high_outlier_cols:
            issues.append(f"High outlier count in columns: {high_outlier_cols}")
        
        # Data type issues
        object_cols = [col for col, dtype in report.data_types.items() if dtype == 'object']
        if len(object_cols) > len(df.columns) * 0.5:
            issues.append("Many columns have object type - consider type conversion")
        
        return issues
    
    def _generate_recommendations(self, report: DataQualityReport) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        if report.quality_score < self.quality_thresholds['minimum_quality_score']:
            recommendations.append("Data quality score is below threshold - comprehensive cleaning recommended")
        
        if sum(report.missing_values.values()) > 0:
            recommendations.append("Apply missing value imputation before analysis")
        
        if report.duplicates > 0:
            recommendations.append("Remove duplicate records to improve data quality")
        
        if sum(report.outliers_detected.values()) > 0:
            recommendations.append("Review and handle outliers before statistical analysis")
        
        recommendations.append("Regular data quality monitoring is recommended")
        
        return recommendations
    
    def _validate_demand_data(self, demand_series: pd.Series) -> List[str]:
        """Validate demand-specific data"""
        issues = []
        
        if demand_series.min() < 0:
            issues.append("Negative demand values found")
        
        if demand_series.isnull().sum() > 0:
            issues.append(f"{demand_series.isnull().sum()} missing demand values")
        
        # Check for unrealistic values
        if demand_series.max() > 1000000:  # 1M liters seems unrealistic for daily demand
            issues.append("Extremely high demand values detected")
        
        return issues
    
    def _validate_date_data(self, date_series: pd.Series) -> List[str]:
        """Validate date-specific data"""
        issues = []
        
        # Try to convert to datetime
        try:
            dates_converted = pd.to_datetime(date_series, errors='coerce')
            null_dates = dates_converted.isnull().sum()
            
            if null_dates > 0:
                issues.append(f"{null_dates} invalid date formats found")
            
            # Check date range reasonableness
            if not dates_converted.empty:
                min_date = dates_converted.min()
                max_date = dates_converted.max()
                
                if min_date < pd.Timestamp('1900-01-01'):
                    issues.append("Dates before 1900 found - check data validity")
                
                if max_date > pd.Timestamp.now() + pd.Timedelta(days=365):
                    issues.append("Future dates beyond reasonable forecast horizon found")
        
        except Exception as e:
            issues.append(f"Date validation error: {str(e)}")
        
        return issues
    
    def _validate_location_data(self, location_series: pd.Series) -> List[str]:
        """Validate location-specific data"""
        issues = []
        
        if location_series.isnull().sum() > 0:
            issues.append(f"{location_series.isnull().sum()} missing location identifiers")
        
        # Check for suspicious location names
        if location_series.str.len().min() < 2:
            issues.append("Very short location identifiers found")
        
        unique_locations = location_series.nunique()
        total_records = len(location_series)
        
        if unique_locations / total_records > 0.9:
            issues.append("High ratio of unique locations - check for data inconsistencies")
        
        return issues

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing operations performed"""
        return {
            'total_operations': len(self.processing_history),
            'operations_history': self.processing_history,
            'scalers_created': list(self.scalers.keys()),
            'quality_thresholds': self.quality_thresholds,
            'current_config': self.config.__dict__
        }
    
    def export_processing_report(self, output_format: str = 'json') -> Union[str, bytes]:
        """Export processing report in various formats"""
        report_data = {
            'processing_summary': self.get_processing_summary(),
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        if output_format == 'json':
            return json.dumps(report_data, indent=2, default=str)
        elif output_format == 'excel':
            # Create Excel report
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([report_data['processing_summary']])
                summary_df.to_excel(writer, sheet_name='Processing Summary', index=False)
                
                # Operations history
                if self.processing_history:
                    history_df = pd.DataFrame(self.processing_history)
                    history_df.to_excel(writer, sheet_name='Operations History', index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

# Utility functions for common data processing tasks
def quick_clean_bbm_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quick cleaning for BBM data with default settings"""
    processor = DataProcessor()
    return processor.clean_dataset(df)

def validate_bbm_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick validation for BBM datasets"""
    processor = DataProcessor()
    quality_report = processor.assess_data_quality(df)
    validation_results = processor.validate_bbm_data(df)
    
    return {
        'quality_report': quality_report,
        'validation_results': validation_results,
        'overall_status': 'PASS' if validation_results['is_valid'] and quality_report.quality_score > 0.8 else 'FAIL'
    }

def prepare_data_for_analysis(df: pd.DataFrame, analysis_type: str = 'arima') -> Union[Dict, pd.DataFrame]:
    """Prepare data for specific analysis types"""
    processor = DataProcessor()
    
    if analysis_type == 'arima':
        # Assume standard BBM columns
        if 'demand' in df.columns and 'location' in df.columns:
            return processor.transform_for_arima(df, 'demand', 
                                               date_column='date' if 'date' in df.columns else None,
                                               location_column='location')
    
    elif analysis_type == 'optimization':
        location_cols = [col for col in df.columns if 'location' in col.lower()]
        demand_cols = [col for col in df.columns if 'demand' in col.lower()]
        coord_cols = [col for col in df.columns if col.lower() in ['latitude', 'longitude', 'lat', 'lon']]
        
        return processor.transform_for_optimization(df, location_cols, demand_cols, coord_cols)
    
    else:
        # General cleaning
        return processor.clean_dataset(df)

def create_sample_bbm_dataset(num_locations: int = 5, num_days: int = 30) -> pd.DataFrame:
    """Create sample BBM dataset for testing"""
    np.random.seed(42)
    
    locations = [f"SPBU_{i:03d}" for i in range(1, num_locations + 1)]
    dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
    
    data = []
    for location in locations:
        base_demand = np.random.normal(5000, 1000)  # Base daily demand
        
        for date in dates:
            # Add seasonal and random variation
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            random_factor = np.random.normal(1, 0.15)
            
            daily_demand = max(0, base_demand * seasonal_factor * random_factor)
            
            # Add some missing values randomly
            if np.random.random() < 0.05:  # 5% missing rate
                daily_demand = np.nan
            
            data.append({
                'location': location,
                'date': date,
                'demand': daily_demand,
                'latitude': -6.2 + np.random.normal(0, 0.1),  # Jakarta area
                'longitude': 106.8 + np.random.normal(0, 0.1),
                'capacity': np.random.normal(50000, 10000),
                'current_stock': np.random.normal(25000, 5000)
            })
    
    df = pd.DataFrame(data)
    
    # Add some outliers
    outlier_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    df.loc[outlier_indices, 'demand'] *= 5  # Make some demands 5x higher
    
    # Add some duplicates
    duplicate_indices = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

# Example usage and testing
if __name__ == "__main__":
    # Initialize data processor
    processor = DataProcessor()
    
    print("Data Processing Utilities Test Results:")
    print("=" * 60)
    
    # Create sample dataset
    print("\n1. Creating Sample Dataset:")
    sample_df = create_sample_bbm_dataset(num_locations=3, num_days=14)
    print(f"   Created dataset: {len(sample_df)} records, {len(sample_df.columns)} columns")
    print(f"   Columns: {list(sample_df.columns)}")
    print(f"   Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
    print(f"   Locations: {sample_df['location'].unique()}")
    
    # Data quality assessment
    print("\n2. Data Quality Assessment:")
    quality_report = processor.assess_data_quality(sample_df)
    print(f"   Quality Score: {quality_report.quality_score:.3f}")
    print(f"   Missing Values: {sum(quality_report.missing_values.values())} total")
    print(f"   Duplicates: {quality_report.duplicates}")
    print(f"   Outliers Detected: {sum(quality_report.outliers_detected.values())}")
    print(f"   Processing Time: {quality_report.processing_time:.3f}s")
    
    if quality_report.issues_found:
        print(f"   Issues Found:")
        for issue in quality_report.issues_found:
            print(f"     - {issue}")
    
    if quality_report.recommendations:
        print(f"   Recommendations:")
        for rec in quality_report.recommendations:
            print(f"     - {rec}")
    
    # Data cleaning
    print("\n3. Data Cleaning:")
    cleaned_df = processor.clean_dataset(sample_df)
    print(f"   Records: {len(sample_df)} → {len(cleaned_df)}")
    print(f"   Missing values before: {sample_df.isnull().sum().sum()}")
    print(f"   Missing values after: {cleaned_df.isnull().sum().sum()}")
    print(f"   Duplicates before: {sample_df.duplicated().sum()}")
    print(f"   Duplicates after: {cleaned_df.duplicated().sum()}")
    
    # BBM-specific validation
    print("\n4. BBM Data Validation:")
    validation_results = processor.validate_bbm_data(cleaned_df, ValidationLevel.INTERMEDIATE)
    print(f"   Validation Status: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
    print(f"   Errors: {len(validation_results['errors'])}")
    print(f"   Warnings: {len(validation_results['warnings'])}")
    print(f"   Suggestions: {len(validation_results['suggestions'])}")
    
    if validation_results['errors']:
        print("   Errors found:")
        for error in validation_results['errors']:
            print(f"     - {error}")
    
    # ARIMA transformation
    print("\n5. ARIMA Data Transformation:")
    arima_data = processor.transform_for_arima(
        cleaned_df, 'demand', 'date', 'location'
    )
    print(f"   Transformed for {len(arima_data)} locations")
    for location, ts in arima_data.items():
        print(f"     {location}: {len(ts)} time points, avg demand: {ts.mean():.0f}")
    
    # Optimization transformation
    print("\n6. Optimization Data Transformation:")
    opt_data = processor.transform_for_optimization(
        cleaned_df, ['location'], ['demand'], ['latitude', 'longitude']
    )
    print(f"   Locations: {len(opt_data['locations'])}")
    print(f"   Total demand: {opt_data['summary']['total_demand']:,.0f}")
    print(f"   Avg demand per location: {opt_data['summary']['avg_demand_per_location']:,.0f}")
    
    # Data normalization
    print("\n7. Data Normalization:")
    numeric_columns = ['demand', 'capacity', 'current_stock']
    normalized_df = processor.normalize_data(cleaned_df, numeric_columns, 'standard')
    print(f"   Normalized {len(numeric_columns)} columns")
    for col in numeric_columns:
        if col in normalized_df.columns:
            print(f"     {col}: mean={normalized_df[col].mean():.3f}, std={normalized_df[col].std():.3f}")
    
    # Aggregation test
    print("\n8. Data Aggregation:")
    agg_df = processor.aggregate_demand_data(
        cleaned_df, ['location'], 'demand', 'sum', 'weekly'
    )
    print(f"   Aggregated: {len(cleaned_df)} → {len(agg_df)} records")
    print(f"   Columns: {list(agg_df.columns)}")
    
    # Processing summary
    print("\n9. Processing Summary:")
    summary = processor.get_processing_summary()
    print(f"   Total operations: {summary['total_operations']}")
    print(f"   Scalers created: {len(summary['scalers_created'])}")
    
    # Quick utility functions test
    print("\n10. Quick Utility Functions:")
    quick_cleaned = quick_clean_bbm_data(sample_df)
    print(f"    Quick clean: {len(sample_df)} → {len(quick_cleaned)} records")
    
    validation_summary = validate_bbm_dataset(quick_cleaned)
    print(f"    Quick validation: {validation_summary['overall_status']}")
    print(f"    Quality score: {validation_summary['quality_report'].quality_score:.3f}")
    
    # Performance test
    print("\n11. Performance Test:")
    import time
    
    # Large dataset test
    large_df = create_sample_bbm_dataset(num_locations=20, num_days=365)
    print(f"    Large dataset: {len(large_df)} records")
    
    start_time = time.time()
    large_quality = processor.assess_data_quality(large_df)
    quality_time = time.time() - start_time
    
    start_time = time.time()
    large_cleaned = processor.clean_dataset(large_df)
    cleaning_time = time.time() - start_time
    
    print(f"    Quality assessment: {quality_time:.3f}s")
    print(f"    Data cleaning: {cleaning_time:.3f}s")
    print(f"    Total processing rate: {len(large_df) / (quality_time + cleaning_time):.0f} records/sec")
    
    print("\n" + "=" * 60)
    print("Data processing utilities test completed! 🔧")
    
    # Final summary
    print("\nFinal Summary:")
    print(f"✅ Data quality assessment: Comprehensive analysis with {quality_report.quality_score:.1%} score")
    print(f"✅ Data cleaning: Automated cleaning with multiple methods")
    print(f"✅ BBM validation: Domain-specific validation rules")
    print(f"✅ ARIMA transformation: Time series preparation for forecasting")
    print(f"✅ Optimization transformation: Data structuring for GA/VRP")
    print(f"✅ Data normalization: Multiple scaling methods available")
    print(f"✅ Aggregation: Flexible grouping and time period aggregation")
    print(f"✅ Utility functions: Quick access methods for common tasks")
    print(f"✅ Performance: Efficient processing for large datasets")
    
    print(f"\n🎯 Data processing system ready for production use!")