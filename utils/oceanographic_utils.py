"""
Oceanographic Data Processing and Analysis Utilities
Wave forecasting, tidal calculations, and maritime condition analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import math
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeaState(Enum):
    CALM = 0          # 0-0.1m
    SMOOTH = 1        # 0.1-0.5m
    SLIGHT = 2        # 0.5-1.25m
    MODERATE = 3      # 1.25-2.5m
    ROUGH = 4         # 2.5-4m
    VERY_ROUGH = 5    # 4-6m
    HIGH = 6          # 6-9m
    VERY_HIGH = 7     # 9-14m
    PHENOMENAL = 8    # >14m

@dataclass
class OceanCondition:
    timestamp: datetime
    significant_wave_height: float
    peak_wave_period: float
    mean_wave_direction: float
    wind_speed: float
    wind_direction: float
    current_speed: float
    current_direction: float
    water_temperature: float
    visibility: float
    sea_state: SeaState
    
    @property
    def beaufort_scale(self) -> int:
        if self.wind_speed < 0.3:
            return 0
        elif self.wind_speed < 1.5:
            return 1
        elif self.wind_speed < 3.3:
            return 2
        elif self.wind_speed < 5.5:
            return 3
        elif self.wind_speed < 8.0:
            return 4
        elif self.wind_speed < 10.8:
            return 5
        elif self.wind_speed < 13.9:
            return 6
        elif self.wind_speed < 17.2:
            return 7
        elif self.wind_speed < 20.7:
            return 8
        elif self.wind_speed < 24.5:
            return 9
        elif self.wind_speed < 28.4:
            return 10
        elif self.wind_speed < 32.6:
            return 11
        else:
            return 12

@dataclass
class TidalData:
    timestamp: datetime
    tide_height: float
    tide_type: str  # "high", "low", "rising", "falling"
    tidal_range: float
    tidal_current: float
    moon_phase: float

class OceanographicProcessor:
    def __init__(self):
        self.wave_models = {}
        self.tidal_stations = {}
        self.historical_data = {}
        
        # Indonesian waters parameters
        self.java_sea_params = {
            'avg_depth': 46.0,
            'max_fetch': 600.0,
            'seasonal_wind_pattern': 'monsoon'
        }
        
        self.indian_ocean_params = {
            'avg_depth': 3890.0,
            'max_fetch': 2000.0,
            'seasonal_wind_pattern': 'monsoon'
        }
        
        logger.info("Oceanographic processor initialized")
    
    def classify_sea_state(self, wave_height: float) -> SeaState:
        if wave_height <= 0.1:
            return SeaState.CALM
        elif wave_height <= 0.5:
            return SeaState.SMOOTH
        elif wave_height <= 1.25:
            return SeaState.SLIGHT
        elif wave_height <= 2.5:
            return SeaState.MODERATE
        elif wave_height <= 4.0:
            return SeaState.ROUGH
        elif wave_height <= 6.0:
            return SeaState.VERY_ROUGH
        elif wave_height <= 9.0:
            return SeaState.HIGH
        elif wave_height <= 14.0:
            return SeaState.VERY_HIGH
        else:
            return SeaState.PHENOMENAL
    
    def calculate_significant_wave_height(self, wind_speed: float, fetch: float, duration: float) -> float:
        # SMB method for wave prediction
        g = 9.81  # gravity
        
        # Non-dimensional parameters
        U_star = wind_speed * 0.71  # friction velocity
        fetch_nd = (g * fetch) / (U_star ** 2)
        duration_nd = (g * duration) / U_star
        
        # Fetch-limited conditions
        if fetch_nd <= 2.2e4:
            H_s_nd = 1.6e-3 * (fetch_nd ** 0.5)
        else:
            H_s_nd = 2.433e-1 * (fetch_nd ** 0.25)
        
        # Duration-limited conditions
        if duration_nd <= 7.15e4:
            H_s_nd_duration = 4.13e-2 * (duration_nd ** 0.25)
        else:
            H_s_nd_duration = 2.433e-1
        
        # Take minimum (limiting factor)
        H_s_nd_final = min(H_s_nd, H_s_nd_duration)
        
        # Convert to dimensional
        significant_wave_height = H_s_nd_final * (U_star ** 2) / g
        
        return max(0.1, significant_wave_height)
    
    def calculate_wave_period(self, wave_height: float, wind_speed: float) -> float:
        # Empirical relationship for wave period
        g = 9.81
        
        if wind_speed > 0:
            # Steepness-based calculation
            steepness = 0.142 * np.tanh(0.0925 * (g * wave_height / (wind_speed ** 2)) ** 0.28)
            period = (2 * np.pi * wave_height / (g * steepness)) ** 0.5
        else:
            # Default period for given wave height
            period = 3.86 * (wave_height ** 0.4)
        
        return max(2.0, min(20.0, period))
    
    def predict_wave_conditions(self, wind_speed: float, wind_direction: float,
                               fetch: float, duration: float = 10800,
                               location_type: str = "java_sea") -> Dict[str, float]:
        
        # Get location parameters
        if location_type == "java_sea":
            params = self.java_sea_params
        else:
            params = self.indian_ocean_params
        
        # Adjust fetch based on location
        effective_fetch = min(fetch, params['max_fetch'])
        
        # Calculate wave height
        wave_height = self.calculate_significant_wave_height(wind_speed, effective_fetch, duration)
        
        # Seasonal adjustment for monsoon
        current_month = datetime.now().month
        if params['seasonal_wind_pattern'] == 'monsoon':
            if 12 <= current_month <= 3:  # Northeast monsoon
                wave_height *= 1.2
            elif 6 <= current_month <= 9:  # Southwest monsoon
                wave_height *= 1.4
        
        # Calculate wave period
        wave_period = self.calculate_wave_period(wave_height, wind_speed)
        
        # Wave direction (typically follows wind with some deviation)
        wave_direction = wind_direction + np.random.normal(0, 15)
        wave_direction = wave_direction % 360
        
        return {
            'significant_wave_height': wave_height,
            'peak_wave_period': wave_period,
            'mean_wave_direction': wave_direction,
            'sea_state': self.classify_sea_state(wave_height).value
        }
    
    def calculate_tidal_height(self, timestamp: datetime, latitude: float, longitude: float) -> TidalData:
        # Simplified tidal calculation for Indonesian waters
        
        # Principal tidal constituents for Indonesian waters
        M2_amp = 0.8   # Principal lunar semi-diurnal
        S2_amp = 0.3   # Principal solar semi-diurnal
        K1_amp = 0.6   # Lunar diurnal
        O1_amp = 0.4   # Lunar diurnal
        
        # Angular frequencies (degrees per hour)
        M2_freq = 28.9841042  # 12.42 hour period
        S2_freq = 30.0000000  # 12.00 hour period
        K1_freq = 15.0410686  # 23.93 hour period
        O1_freq = 13.9430356  # 25.82 hour period
        
        # Calculate hours since reference
        ref_time = datetime(2024, 1, 1, 0, 0, 0)
        hours_since_ref = (timestamp - ref_time).total_seconds() / 3600
        
        # Calculate tidal components
        M2_component = M2_amp * math.cos(math.radians(M2_freq * hours_since_ref))
        S2_component = S2_amp * math.cos(math.radians(S2_freq * hours_since_ref))
        K1_component = K1_amp * math.cos(math.radians(K1_freq * hours_since_ref))
        O1_component = O1_amp * math.cos(math.radians(O1_freq * hours_since_ref))
        
        # Sum components
        tide_height = M2_component + S2_component + K1_component + O1_component
        
        # Add mean sea level and location adjustments
        mean_sea_level = 1.2  # meters above chart datum
        tide_height += mean_sea_level
        
        # Determine tide type
        previous_height = self.calculate_simple_tide(timestamp - timedelta(hours=1), latitude, longitude)
        next_height = self.calculate_simple_tide(timestamp + timedelta(hours=1), latitude, longitude)
        
        if tide_height > previous_height and tide_height > next_height:
            tide_type = "high"
        elif tide_height < previous_height and tide_height < next_height:
            tide_type = "low"
        elif tide_height > previous_height:
            tide_type = "rising"
        else:
            tide_type = "falling"
        
        # Calculate tidal range and current
        tidal_range = (M2_amp + S2_amp + K1_amp + O1_amp) * 2
        tidal_current = abs(next_height - previous_height) / 2 * 0.5  # Approximate current
        
        # Moon phase calculation (simplified)
        days_since_new_moon = (timestamp - datetime(2024, 1, 11)).days % 29.53
        moon_phase = days_since_new_moon / 29.53
        
        return TidalData(
            timestamp=timestamp,
            tide_height=tide_height,
            tide_type=tide_type,
            tidal_range=tidal_range,
            tidal_current=tidal_current,
            moon_phase=moon_phase
        )
    
    def calculate_simple_tide(self, timestamp: datetime, latitude: float, longitude: float) -> float:
        # Simplified version for internal calculations
        ref_time = datetime(2024, 1, 1, 0, 0, 0)
        hours_since_ref = (timestamp - ref_time).total_seconds() / 3600
        
        M2_component = 0.8 * math.cos(math.radians(28.9841042 * hours_since_ref))
        S2_component = 0.3 * math.cos(math.radians(30.0000000 * hours_since_ref))
        
        return M2_component + S2_component + 1.2
    
    def generate_wave_forecast(self, location: Tuple[float, float], 
                              forecast_hours: int = 72) -> List[OceanCondition]:
        
        latitude, longitude = location
        forecast_data = []
        
        # Determine location characteristics
        if -8 < latitude < -2 and 105 < longitude < 115:
            location_type = "java_sea"
            base_wind = 8.0
        else:
            location_type = "indian_ocean"
            base_wind = 12.0
        
        # Generate forecast
        for hour in range(forecast_hours):
            timestamp = datetime.now() + timedelta(hours=hour)
            
            # Simulate wind conditions with daily and seasonal patterns
            daily_factor = 1 + 0.3 * math.sin(2 * math.pi * hour / 24)
            seasonal_factor = 1 + 0.2 * math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 365)
            random_factor = 1 + np.random.normal(0, 0.2)
            
            wind_speed = base_wind * daily_factor * seasonal_factor * random_factor
            wind_speed = max(0, wind_speed)
            
            wind_direction = 180 + 30 * math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 365)
            wind_direction += np.random.normal(0, 20)
            wind_direction = wind_direction % 360
            
            # Calculate wave conditions
            fetch = 300 if location_type == "java_sea" else 800
            wave_pred = self.predict_wave_conditions(wind_speed, wind_direction, fetch, 
                                                   duration=3600, location_type=location_type)
            
            # Calculate tidal data
            tidal_data = self.calculate_tidal_height(timestamp, latitude, longitude)
            
            # Current conditions (simplified)
            current_speed = tidal_data.tidal_current + 0.2 * wind_speed * 0.03
            current_direction = wind_direction + np.random.normal(0, 30)
            current_direction = current_direction % 360
            
            # Water temperature (seasonal variation)
            base_temp = 28.0  # Indonesian waters
            seasonal_temp_var = 2 * math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 365)
            water_temp = base_temp + seasonal_temp_var + np.random.normal(0, 0.5)
            
            # Visibility (weather dependent)
            visibility = 10.0 - (wind_speed - 10) * 0.5 if wind_speed > 10 else 10.0
            visibility = max(1.0, visibility + np.random.normal(0, 1))
            
            condition = OceanCondition(
                timestamp=timestamp,
                significant_wave_height=wave_pred['significant_wave_height'],
                peak_wave_period=wave_pred['peak_wave_period'],
                mean_wave_direction=wave_pred['mean_wave_direction'],
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                current_speed=current_speed,
                current_direction=current_direction,
                water_temperature=water_temp,
                visibility=visibility,
                sea_state=SeaState(wave_pred['sea_state'])
            )
            
            forecast_data.append(condition)
        
        return forecast_data
    
    def analyze_maritime_safety(self, conditions: List[OceanCondition],
                               vessel_limitations: Dict[str, float] = None) -> Dict[str, Any]:
        
        if not vessel_limitations:
            vessel_limitations = {
                'max_wave_height': 3.0,
                'max_wind_speed': 20.0,
                'min_visibility': 3.0
            }
        
        safe_periods = []
        risk_periods = []
        
        for condition in conditions:
            safety_score = self.calculate_safety_score(condition, vessel_limitations)
            
            period_data = {
                'timestamp': condition.timestamp,
                'safety_score': safety_score,
                'wave_height': condition.significant_wave_height,
                'wind_speed': condition.wind_speed,
                'visibility': condition.visibility,
                'sea_state': condition.sea_state.value
            }
            
            if safety_score >= 0.7:
                safe_periods.append(period_data)
            elif safety_score < 0.4:
                risk_periods.append(period_data)
        
        return {
            'total_forecast_hours': len(conditions),
            'safe_periods': safe_periods,
            'risk_periods': risk_periods,
            'safety_percentage': len(safe_periods) / len(conditions) * 100,
            'avg_wave_height': np.mean([c.significant_wave_height for c in conditions]),
            'max_wave_height': max([c.significant_wave_height for c in conditions]),
            'avg_wind_speed': np.mean([c.wind_speed for c in conditions]),
            'storm_hours': len([c for c in conditions if c.beaufort_scale >= 8]),
            'calm_hours': len([c for c in conditions if c.sea_state.value <= 2])
        }
    
    def calculate_safety_score(self, condition: OceanCondition, limitations: Dict[str, float]) -> float:
        score = 1.0
        
        # Wave height penalty
        if condition.significant_wave_height > limitations['max_wave_height']:
            wave_penalty = (condition.significant_wave_height - limitations['max_wave_height']) / limitations['max_wave_height']
            score -= min(wave_penalty * 0.5, 0.6)
        
        # Wind speed penalty
        if condition.wind_speed > limitations['max_wind_speed']:
            wind_penalty = (condition.wind_speed - limitations['max_wind_speed']) / limitations['max_wind_speed']
            score -= min(wind_penalty * 0.3, 0.4)
        
        # Visibility penalty
        if condition.visibility < limitations['min_visibility']:
            vis_penalty = (limitations['min_visibility'] - condition.visibility) / limitations['min_visibility']
            score -= min(vis_penalty * 0.4, 0.5)
        
        # Sea state penalty
        if condition.sea_state.value >= 5:  # Very rough or worse
            score -= 0.3
        elif condition.sea_state.value >= 4:  # Rough
            score -= 0.2
        
        return max(0.0, score)
    
    def find_optimal_departure_windows(self, forecast: List[OceanCondition],
                                     min_duration_hours: float = 4.0,
                                     vessel_limitations: Dict[str, float] = None) -> List[Dict[str, Any]]:
        
        safety_analysis = self.analyze_maritime_safety(forecast, vessel_limitations)
        safe_periods = safety_analysis['safe_periods']
        
        # Group consecutive safe periods
        windows = []
        current_window = []
        
        for i, period in enumerate(safe_periods):
            if not current_window:
                current_window = [period]
            else:
                # Check if consecutive (within 2 hours)
                last_time = current_window[-1]['timestamp']
                current_time = period['timestamp']
                
                if (current_time - last_time).total_seconds() <= 7200:  # 2 hours
                    current_window.append(period)
                else:
                    # Process completed window
                    if len(current_window) >= min_duration_hours:
                        windows.append(self.create_departure_window(current_window))
                    current_window = [period]
        
        # Process final window
        if len(current_window) >= min_duration_hours:
            windows.append(self.create_departure_window(current_window))
        
        # Sort by quality score
        windows.sort(key=lambda w: w['quality_score'], reverse=True)
        
        return windows
    
    def create_departure_window(self, periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = periods[0]['timestamp']
        end_time = periods[-1]['timestamp']
        duration = (end_time - start_time).total_seconds() / 3600
        
        avg_safety = np.mean([p['safety_score'] for p in periods])
        avg_wave_height = np.mean([p['wave_height'] for p in periods])
        avg_wind_speed = np.mean([p['wind_speed'] for p in periods])
        max_wave_height = max([p['wave_height'] for p in periods])
        
        # Quality score combines safety, duration, and conditions
        quality_score = (avg_safety * 0.4 + 
                        min(duration / 12, 1.0) * 0.3 +  # Prefer longer windows
                        (1 - max_wave_height / 6) * 0.3)  # Prefer calmer seas
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration_hours': duration,
            'avg_safety_score': avg_safety,
            'avg_wave_height': avg_wave_height,
            'max_wave_height': max_wave_height,
            'avg_wind_speed': avg_wind_speed,
            'quality_score': quality_score,
            'periods_count': len(periods)
        }
    
    def export_forecast_data(self, forecast: List[OceanCondition]) -> pd.DataFrame:
        data = []
        
        for condition in forecast:
            data.append({
                'timestamp': condition.timestamp,
                'wave_height_m': condition.significant_wave_height,
                'wave_period_s': condition.peak_wave_period,
                'wave_direction_deg': condition.mean_wave_direction,
                'wind_speed_ms': condition.wind_speed,
                'wind_direction_deg': condition.wind_direction,
                'current_speed_ms': condition.current_speed,
                'current_direction_deg': condition.current_direction,
                'water_temp_c': condition.water_temperature,
                'visibility_km': condition.visibility,
                'sea_state': condition.sea_state.value,
                'beaufort_scale': condition.beaufort_scale
            })
        
        return pd.DataFrame(data)
    
    def calculate_wave_energy(self, wave_height: float, wave_period: float) -> float:
        # Wave energy calculation (kW/m)
        rho = 1025  # seawater density kg/m³
        g = 9.81    # gravity m/s²
        
        energy = (rho * g**2 / (64 * np.pi)) * wave_height**2 * wave_period
        return energy / 1000  # Convert to kW/m
    
    def assess_vessel_operability(self, forecast: List[OceanCondition],
                                 vessel_specs: Dict[str, Any]) -> Dict[str, Any]:
        
        operable_hours = 0
        restricted_hours = 0
        dangerous_hours = 0
        
        max_wave = vessel_specs.get('max_wave_height', 2.5)
        max_wind = vessel_specs.get('max_wind_speed', 15.0)
        
        for condition in forecast:
            if (condition.significant_wave_height <= max_wave and 
                condition.wind_speed <= max_wind and
                condition.visibility >= 2.0):
                operable_hours += 1
            elif (condition.significant_wave_height <= max_wave * 1.5 and
                  condition.wind_speed <= max_wind * 1.3):
                restricted_hours += 1
            else:
                dangerous_hours += 1
        
        total_hours = len(forecast)
        
        return {
            'operable_percentage': operable_hours / total_hours * 100,
            'restricted_percentage': restricted_hours / total_hours * 100,
            'dangerous_percentage': dangerous_hours / total_hours * 100,
            'total_forecast_hours': total_hours,
            'recommended_departure_windows': self.find_optimal_departure_windows(
                forecast, vessel_limitations={'max_wave_height': max_wave, 'max_wind_speed': max_wind, 'min_visibility': 2.0}
            )
        }

def create_indonesian_forecast(location_name: str = "Jakarta_Bay") -> List[OceanCondition]:
    processor = OceanographicProcessor()
    
    # Indonesian maritime locations
    locations = {
        "Jakarta_Bay": (-6.1, 106.8),
        "Surabaya_Coast": (-7.3, 112.7),
        "Balikpapan_Waters": (-1.3, 116.8),
        "Makassar_Strait": (-5.1, 119.4),
        "Banda_Sea": (-4.5, 129.9)
    }
    
    location_coords = locations.get(location_name, (-6.1, 106.8))
    return processor.generate_wave_forecast(location_coords, 72)

# Example usage
if __name__ == "__main__":
    processor = OceanographicProcessor()
    
    print("Oceanographic Utilities Test:")
    print("=" * 40)
    
    # Generate forecast for Jakarta Bay
    forecast = create_indonesian_forecast("Jakarta_Bay")
    print(f"Generated {len(forecast)} hour forecast for Jakarta Bay")
    
    # Analyze conditions
    safety_analysis = processor.analyze_maritime_safety(forecast)
    print(f"\nSafety Analysis:")
    print(f"Safe periods: {len(safety_analysis['safe_periods'])}")
    print(f"Risk periods: {len(safety_analysis['risk_periods'])}")
    print(f"Safety percentage: {safety_analysis['safety_percentage']:.1f}%")
    print(f"Average wave height: {safety_analysis['avg_wave_height']:.2f}m")
    print(f"Maximum wave height: {safety_analysis['max_wave_height']:.2f}m")
    
    # Find departure windows
    departure_windows = processor.find_optimal_departure_windows(forecast, min_duration_hours=4)
    print(f"\nOptimal Departure Windows: {len(departure_windows)}")
    
    if departure_windows:
        best_window = departure_windows[0]
        print(f"Best window: {best_window['start_time'].strftime('%Y-%m-%d %H:%M')} - {best_window['end_time'].strftime('%Y-%m-%d %H:%M')}")
        print(f"Duration: {best_window['duration_hours']:.1f} hours")
        print(f"Quality score: {best_window['quality_score']:.2f}")
        print(f"Average wave height: {best_window['avg_wave_height']:.2f}m")
    
    # Vessel operability assessment
    vessel_specs = {
        'max_wave_height': 2.5,
        'max_wind_speed': 15.0,
        'vessel_type': 'coastal_tanker'
    }
    
    operability = processor.assess_vessel_operability(forecast, vessel_specs)
    print(f"\nVessel Operability (Coastal Tanker):")
    print(f"Operable: {operability['operable_percentage']:.1f}%")
    print(f"Restricted: {operability['restricted_percentage']:.1f}%")
    print(f"Dangerous: {operability['dangerous_percentage']:.1f}%")
    
    # Export data
    forecast_df = processor.export_forecast_data(forecast[:24])  # First 24 hours
    print(f"\nExported forecast data: {len(forecast_df)} rows")
    print(f"Columns: {list(forecast_df.columns)}")
    
    # Sample conditions
    print(f"\nSample Conditions (First 3 hours):")
    for i in range(3):
        condition = forecast[i]
        print(f"  {condition.timestamp.strftime('%H:%M')}: Wave {condition.significant_wave_height:.1f}m, "
              f"Wind {condition.wind_speed:.1f}m/s, {condition.sea_state.name}")
    
    print("\nOceanographic utilities test completed! 🌊")