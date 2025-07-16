"""
Wave-influenced BBM Delivery Scheduling Model
Maritime scheduling with oceanographic constraints and vessel management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VesselType(Enum):
    COASTAL_TANKER = "coastal_tanker"
    OFFSHORE_BARGE = "offshore_barge"
    SMALL_VESSEL = "small_vessel"
    HEAVY_TANKER = "heavy_tanker"

class PortType(Enum):
    MAIN_PORT = "main_port"
    COASTAL_TERMINAL = "coastal_terminal"
    OFFSHORE_PLATFORM = "offshore_platform"
    FLOATING_STATION = "floating_station"

class SchedulePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class MaritimeLocation:
    location_id: str
    name: str
    port_type: PortType
    latitude: float
    longitude: float
    water_depth: float
    max_vessel_size: VesselType
    tidal_range: float
    shelter_factor: float
    fuel_capacity: float
    current_stock: float
    daily_demand: float
    priority: SchedulePriority
    operating_hours: Tuple[int, int] = (6, 18)
    weather_restrictions: Dict[str, float] = field(default_factory=dict)

@dataclass
class Vessel:
    vessel_id: str
    vessel_type: VesselType
    fuel_capacity: float
    max_wave_height: float
    min_water_depth: float
    speed_knots: float
    fuel_consumption_rate: float
    crew_cost_per_hour: float
    maintenance_cost_per_day: float
    weather_capability: Dict[str, float] = field(default_factory=dict)
    current_location: str = "base_port"
    availability_start: datetime = None
    availability_end: datetime = None

@dataclass
class WeatherWindow:
    start_time: datetime
    end_time: datetime
    wave_height: float
    wind_speed: float
    visibility: float
    tide_level: float
    weather_score: float
    suitable_vessels: List[VesselType] = field(default_factory=list)

@dataclass
class DeliveryTask:
    task_id: str
    destination: str
    fuel_amount: float
    priority: SchedulePriority
    earliest_start: datetime
    latest_finish: datetime
    estimated_duration: float
    assigned_vessel: str = None
    scheduled_time: datetime = None
    weather_window: WeatherWindow = None

class WaveSchedulingModel:
    def __init__(self):
        self.maritime_locations: Dict[str, MaritimeLocation] = {}
        self.vessels: Dict[str, Vessel] = {}
        self.delivery_tasks: List[DeliveryTask] = []
        self.weather_windows: List[WeatherWindow] = []
        self.distance_matrix: Dict[Tuple[str, str], float] = {}
        
        # Scheduling constraints
        self.max_scheduling_horizon = 7  # days
        self.min_weather_window = 4  # hours
        self.safety_buffer = 1.5  # safety multiplier
        self.fuel_reserve_ratio = 0.1  # 10% fuel reserve
        
        # Cost parameters
        self.base_fuel_cost = 12000  # IDR per liter
        self.weather_delay_penalty = 100000  # IDR per hour delay
        self.vessel_standby_cost = 50000  # IDR per hour
        self.priority_multipliers = {
            SchedulePriority.CRITICAL: 3.0,
            SchedulePriority.HIGH: 2.0,
            SchedulePriority.NORMAL: 1.0,
            SchedulePriority.LOW: 0.5
        }
        
        logger.info("Wave scheduling model initialized")
    
    def add_maritime_location(self, location: MaritimeLocation) -> None:
        self.maritime_locations[location.location_id] = location
        logger.info(f"Added maritime location: {location.name}")
    
    def add_vessel(self, vessel: Vessel) -> None:
        self.vessels[vessel.vessel_id] = vessel
        logger.info(f"Added vessel: {vessel.vessel_id} ({vessel.vessel_type.value})")
    
    def add_delivery_task(self, task: DeliveryTask) -> None:
        self.delivery_tasks.append(task)
        logger.info(f"Added delivery task: {task.task_id} to {task.destination}")
    
    def generate_weather_windows(self, wave_data: Dict[datetime, Any], 
                                forecast_hours: int = 168) -> None:
        logger.info(f"Generating weather windows for {forecast_hours} hours")
        
        self.weather_windows = []
        sorted_times = sorted(wave_data.keys())
        
        for i in range(len(sorted_times) - 1):
            start_time = sorted_times[i]
            end_time = sorted_times[i + 1]
            
            wave_info = wave_data[start_time]
            
            # Calculate weather score
            weather_score = self.calculate_weather_score(wave_info)
            
            # Determine suitable vessels
            suitable_vessels = self.get_suitable_vessels_for_weather(wave_info)
            
            window = WeatherWindow(
                start_time=start_time,
                end_time=end_time,
                wave_height=getattr(wave_info, 'wave_height', 0),
                wind_speed=getattr(wave_info, 'wind_speed', 0),
                visibility=10.0,  # Default visibility
                tide_level=getattr(wave_info, 'tide_level', 0),
                weather_score=weather_score,
                suitable_vessels=suitable_vessels
            )
            
            self.weather_windows.append(window)
    
    def calculate_weather_score(self, wave_info: Any) -> float:
        wave_height = getattr(wave_info, 'wave_height', 0)
        wind_speed = getattr(wave_info, 'wind_speed', 0)
        
        # Weather score based on conditions (0-1, higher is better)
        wave_score = max(0, 1 - (wave_height / 4.0))  # Max 4m waves
        wind_score = max(0, 1 - (wind_speed / 25.0))   # Max 25 m/s wind
        
        return (wave_score * 0.7 + wind_score * 0.3)
    
    def get_suitable_vessels_for_weather(self, wave_info: Any) -> List[VesselType]:
        suitable = []
        wave_height = getattr(wave_info, 'wave_height', 0)
        
        for vessel in self.vessels.values():
            if wave_height <= vessel.max_wave_height:
                suitable.append(vessel.vessel_type)
        
        return list(set(suitable))
    
    def calculate_distance(self, from_location: str, to_location: str) -> float:
        if (from_location, to_location) in self.distance_matrix:
            return self.distance_matrix[(from_location, to_location)]
        
        # Calculate if not cached
        if from_location in self.maritime_locations and to_location in self.maritime_locations:
            loc1 = self.maritime_locations[from_location]
            loc2 = self.maritime_locations[to_location]
            
            # Haversine distance in nautical miles
            lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
            lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            distance = 3440.065 * c  # Earth radius in nautical miles
            self.distance_matrix[(from_location, to_location)] = distance
            return distance
        
        return 0.0
    
    def calculate_travel_time(self, from_location: str, to_location: str, 
                            vessel_id: str) -> float:
        distance = self.calculate_distance(from_location, to_location)
        
        if vessel_id in self.vessels:
            vessel = self.vessels[vessel_id]
            travel_time = distance / vessel.speed_knots  # hours
            return travel_time
        
        return 0.0
    
    def find_optimal_weather_windows(self, task: DeliveryTask, 
                                   vessel_id: str) -> List[WeatherWindow]:
        suitable_windows = []
        
        if vessel_id not in self.vessels:
            return suitable_windows
        
        vessel = self.vessels[vessel_id]
        
        for window in self.weather_windows:
            # Check time constraints
            if (window.start_time >= task.earliest_start and 
                window.end_time <= task.latest_finish):
                
                # Check weather suitability
                if (vessel.vessel_type in window.suitable_vessels and
                    window.wave_height <= vessel.max_wave_height):
                    
                    # Check window duration
                    window_duration = (window.end_time - window.start_time).total_seconds() / 3600
                    if window_duration >= self.min_weather_window:
                        suitable_windows.append(window)
        
        # Sort by weather score (best conditions first)
        suitable_windows.sort(key=lambda w: w.weather_score, reverse=True)
        return suitable_windows
    
    def assign_vessel_to_task(self, task: DeliveryTask) -> Optional[str]:
        best_vessel = None
        best_score = float('-inf')
        
        destination = self.maritime_locations.get(task.destination)
        if not destination:
            return None
        
        for vessel_id, vessel in self.vessels.items():
            # Check vessel capacity
            if vessel.fuel_capacity < task.fuel_amount:
                continue
            
            # Check vessel size compatibility
            if not self.is_vessel_compatible(vessel, destination):
                continue
            
            # Calculate suitability score
            score = self.calculate_vessel_suitability(vessel, task, destination)
            
            if score > best_score:
                best_score = score
                best_vessel = vessel_id
        
        return best_vessel
    
    def is_vessel_compatible(self, vessel: Vessel, destination: MaritimeLocation) -> bool:
        # Check water depth
        if vessel.min_water_depth > destination.water_depth:
            return False
        
        # Check vessel size vs port capacity
        vessel_sizes = {
            VesselType.SMALL_VESSEL: 1,
            VesselType.COASTAL_TANKER: 2,
            VesselType.OFFSHORE_BARGE: 3,
            VesselType.HEAVY_TANKER: 4
        }
        
        port_capacity = {
            PortType.FLOATING_STATION: 2,
            PortType.COASTAL_TERMINAL: 3,
            PortType.OFFSHORE_PLATFORM: 2,
            PortType.MAIN_PORT: 4
        }
        
        vessel_size = vessel_sizes.get(vessel.vessel_type, 1)
        max_size = port_capacity.get(destination.port_type, 1)
        
        return vessel_size <= max_size
    
    def calculate_vessel_suitability(self, vessel: Vessel, task: DeliveryTask, 
                                   destination: MaritimeLocation) -> float:
        score = 0.0
        
        # Capacity utilization (prefer efficient use)
        capacity_ratio = task.fuel_amount / vessel.fuel_capacity
        if 0.3 <= capacity_ratio <= 0.9:
            score += 2.0
        else:
            score += 1.0 - abs(capacity_ratio - 0.6)
        
        # Weather capability
        weather_bonus = 1.0
        if vessel.max_wave_height > 2.5:
            weather_bonus = 1.5
        score += weather_bonus
        
        # Operating cost (lower is better)
        cost_factor = 1.0 / (vessel.crew_cost_per_hour + vessel.maintenance_cost_per_day/24)
        score += cost_factor * 0.1
        
        # Priority matching
        priority_bonus = self.priority_multipliers.get(task.priority, 1.0)
        score *= priority_bonus
        
        return score
    
    def create_delivery_schedule(self, optimization_results: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info("Creating wave-optimized delivery schedule...")
        
        scheduled_tasks = []
        unscheduled_tasks = []
        total_cost = 0.0
        
        # Sort tasks by priority and deadline
        sorted_tasks = sorted(self.delivery_tasks, 
                            key=lambda t: (t.priority.value, t.latest_finish))
        
        for task in sorted_tasks:
            # Assign vessel
            assigned_vessel = self.assign_vessel_to_task(task)
            
            if not assigned_vessel:
                unscheduled_tasks.append(task)
                continue
            
            # Find optimal weather window
            weather_windows = self.find_optimal_weather_windows(task, assigned_vessel)
            
            if not weather_windows:
                unscheduled_tasks.append(task)
                continue
            
            # Select best window
            best_window = weather_windows[0]
            
            # Calculate scheduling details
            vessel = self.vessels[assigned_vessel]
            travel_time = self.calculate_travel_time(vessel.current_location, 
                                                   task.destination, assigned_vessel)
            
            # Schedule within weather window
            departure_time = best_window.start_time
            arrival_time = departure_time + timedelta(hours=travel_time)
            completion_time = arrival_time + timedelta(hours=task.estimated_duration)
            
            # Check if fits in weather window
            if completion_time <= best_window.end_time:
                task.assigned_vessel = assigned_vessel
                task.scheduled_time = departure_time
                task.weather_window = best_window
                
                # Calculate cost
                task_cost = self.calculate_delivery_cost(task, vessel, travel_time)
                total_cost += task_cost
                
                scheduled_tasks.append(task)
                
                # Update vessel availability
                vessel.current_location = task.destination
                vessel.availability_start = completion_time
            else:
                unscheduled_tasks.append(task)
        
        # Generate schedule summary
        schedule_summary = {
            'scheduled_deliveries': len(scheduled_tasks),
            'unscheduled_deliveries': len(unscheduled_tasks),
            'total_cost': total_cost,
            'schedule_efficiency': len(scheduled_tasks) / len(self.delivery_tasks),
            'avg_weather_score': np.mean([t.weather_window.weather_score for t in scheduled_tasks]),
            'critical_deliveries_scheduled': sum(1 for t in scheduled_tasks 
                                               if t.priority == SchedulePriority.CRITICAL)
        }
        
        return {
            'scheduled_tasks': self.format_scheduled_tasks(scheduled_tasks),
            'unscheduled_tasks': [t.task_id for t in unscheduled_tasks],
            'summary': schedule_summary,
            'vessel_utilization': self.calculate_vessel_utilization(),
            'weather_analysis': self.analyze_weather_impact(scheduled_tasks)
        }
    
    def calculate_delivery_cost(self, task: DeliveryTask, vessel: Vessel, 
                              travel_time: float) -> float:
        # Base fuel cost
        fuel_cost = task.fuel_amount * self.base_fuel_cost
        
        # Vessel operating cost
        total_time = travel_time + task.estimated_duration
        operating_cost = (vessel.crew_cost_per_hour * total_time + 
                         vessel.maintenance_cost_per_day * (total_time / 24))
        
        # Weather risk premium
        weather_premium = 0.0
        if task.weather_window:
            risk_factor = 1.0 - task.weather_window.weather_score
            weather_premium = fuel_cost * risk_factor * 0.1
        
        # Priority premium
        priority_premium = fuel_cost * (self.priority_multipliers[task.priority] - 1.0) * 0.05
        
        return fuel_cost + operating_cost + weather_premium + priority_premium
    
    def format_scheduled_tasks(self, scheduled_tasks: List[DeliveryTask]) -> List[Dict]:
        formatted = []
        
        for task in sorted(scheduled_tasks, key=lambda t: t.scheduled_time):
            vessel = self.vessels[task.assigned_vessel]
            destination = self.maritime_locations[task.destination]
            
            formatted.append({
                'task_id': task.task_id,
                'destination': destination.name,
                'destination_type': destination.port_type.value,
                'fuel_amount': task.fuel_amount,
                'priority': task.priority.value,
                'assigned_vessel': task.assigned_vessel,
                'vessel_type': vessel.vessel_type.value,
                'scheduled_departure': task.scheduled_time.strftime('%Y-%m-%d %H:%M'),
                'weather_conditions': {
                    'wave_height': task.weather_window.wave_height,
                    'wind_speed': task.weather_window.wind_speed,
                    'weather_score': task.weather_window.weather_score
                },
                'estimated_duration': task.estimated_duration,
                'travel_time': self.calculate_travel_time(vessel.current_location, 
                                                        task.destination, task.assigned_vessel)
            })
        
        return formatted
    
    def calculate_vessel_utilization(self) -> Dict[str, float]:
        utilization = {}
        
        scheduled_tasks_by_vessel = {}
        for task in self.delivery_tasks:
            if task.assigned_vessel:
                if task.assigned_vessel not in scheduled_tasks_by_vessel:
                    scheduled_tasks_by_vessel[task.assigned_vessel] = []
                scheduled_tasks_by_vessel[task.assigned_vessel].append(task)
        
        for vessel_id, vessel in self.vessels.items():
            if vessel_id in scheduled_tasks_by_vessel:
                tasks = scheduled_tasks_by_vessel[vessel_id]
                total_time = sum(t.estimated_duration for t in tasks)
                utilization[vessel_id] = min(1.0, total_time / (24 * self.max_scheduling_horizon))
            else:
                utilization[vessel_id] = 0.0
        
        return utilization
    
    def analyze_weather_impact(self, scheduled_tasks: List[DeliveryTask]) -> Dict[str, Any]:
        if not scheduled_tasks:
            return {}
        
        weather_scores = [t.weather_window.weather_score for t in scheduled_tasks]
        wave_heights = [t.weather_window.wave_height for t in scheduled_tasks]
        
        return {
            'avg_weather_score': np.mean(weather_scores),
            'min_weather_score': np.min(weather_scores),
            'avg_wave_height': np.mean(wave_heights),
            'max_wave_height': np.max(wave_heights),
            'good_weather_percentage': sum(1 for s in weather_scores if s > 0.7) / len(weather_scores),
            'weather_delay_risk': sum(1 for s in weather_scores if s < 0.4) / len(weather_scores)
        }
    
    def optimize_schedule_with_waves(self, memetic_results: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Integrating memetic algorithm results with wave scheduling...")
        
        # Extract memetic schedule
        memetic_schedule = memetic_results.get('best_schedule', [])
        
        # Update delivery tasks with memetic optimization
        for schedule_item in memetic_schedule:
            location = schedule_item['location']
            delivery_time = datetime.strptime(schedule_item['delivery_time'], '%Y-%m-%d %H:%M')
            fuel_amount = schedule_item['fuel_amount']
            
            # Find or create corresponding task
            task = next((t for t in self.delivery_tasks if t.destination == location), None)
            
            if task:
                task.scheduled_time = delivery_time
                task.fuel_amount = fuel_amount
            else:
                # Create new task from memetic results
                new_task = DeliveryTask(
                    task_id=f"memetic_{location}",
                    destination=location,
                    fuel_amount=fuel_amount,
                    priority=SchedulePriority.NORMAL,
                    earliest_start=delivery_time,
                    latest_finish=delivery_time + timedelta(hours=12),
                    estimated_duration=2.0
                )
                self.delivery_tasks.append(new_task)
        
        # Create optimized schedule
        optimized_schedule = self.create_delivery_schedule()
        
        # Add memetic integration metrics
        optimized_schedule['memetic_integration'] = {
            'memetic_fitness': memetic_results.get('best_fitness', 0),
            'memetic_cost': memetic_results.get('total_cost', 0),
            'wave_optimized_cost': optimized_schedule['summary']['total_cost'],
            'integration_efficiency': optimized_schedule['summary']['schedule_efficiency']
        }
        
        return optimized_schedule
    
    def export_schedule_data(self) -> Dict[str, Any]:
        return {
            'maritime_locations': {
                loc_id: {
                    'name': loc.name,
                    'type': loc.port_type.value,
                    'coordinates': [loc.latitude, loc.longitude],
                    'capacity': loc.fuel_capacity,
                    'demand': loc.daily_demand,
                    'priority': loc.priority.value
                } for loc_id, loc in self.maritime_locations.items()
            },
            'vessels': {
                vessel_id: {
                    'type': vessel.vessel_type.value,
                    'capacity': vessel.fuel_capacity,
                    'max_wave_height': vessel.max_wave_height,
                    'speed': vessel.speed_knots,
                    'location': vessel.current_location
                } for vessel_id, vessel in self.vessels.items()
            },
            'delivery_tasks': [
                {
                    'task_id': task.task_id,
                    'destination': task.destination,
                    'fuel_amount': task.fuel_amount,
                    'priority': task.priority.value,
                    'scheduled_time': task.scheduled_time.isoformat() if task.scheduled_time else None
                } for task in self.delivery_tasks
            ],
            'weather_windows': len(self.weather_windows),
            'scheduling_constraints': {
                'max_horizon_days': self.max_scheduling_horizon,
                'min_weather_window_hours': self.min_weather_window,
                'safety_buffer': self.safety_buffer
            }
        }

def create_sample_maritime_network() -> WaveSchedulingModel:
    model = WaveSchedulingModel()
    
    # Add maritime locations
    locations = [
        MaritimeLocation("MAIN_PORT", "Jakarta Port", PortType.MAIN_PORT, 
                        -6.1167, 106.8833, 15.0, VesselType.HEAVY_TANKER, 2.5, 0.9, 
                        500000, 400000, 50000, SchedulePriority.HIGH),
        MaritimeLocation("COASTAL_01", "Tanjung Priok Terminal", PortType.COASTAL_TERMINAL,
                        -6.1075, 106.8800, 12.0, VesselType.COASTAL_TANKER, 2.0, 0.8,
                        100000, 60000, 15000, SchedulePriority.NORMAL),
        MaritimeLocation("OFFSHORE_01", "Offshore Platform A", PortType.OFFSHORE_PLATFORM,
                        -5.8500, 107.2000, 25.0, VesselType.OFFSHORE_BARGE, 1.5, 0.6,
                        50000, 20000, 8000, SchedulePriority.CRITICAL),
        MaritimeLocation("FLOATING_01", "Floating Station B", PortType.FLOATING_STATION,
                        -6.3000, 106.5000, 8.0, VesselType.SMALL_VESSEL, 3.0, 0.4,
                        20000, 5000, 3000, SchedulePriority.LOW)
    ]
    
    for loc in locations:
        model.add_maritime_location(loc)
    
    # Add vessels
    vessels = [
        Vessel("TANKER_01", VesselType.HEAVY_TANKER, 50000, 3.0, 10.0, 12.0, 
               150.0, 200000, 500000),
        Vessel("COASTAL_01", VesselType.COASTAL_TANKER, 25000, 2.5, 8.0, 15.0,
               80.0, 150000, 300000),
        Vessel("BARGE_01", VesselType.OFFSHORE_BARGE, 15000, 2.0, 5.0, 10.0,
               60.0, 100000, 200000),
        Vessel("SMALL_01", VesselType.SMALL_VESSEL, 5000, 1.5, 3.0, 20.0,
               25.0, 75000, 100000)
    ]
    
    for vessel in vessels:
        model.add_vessel(vessel)
    
    # Add delivery tasks
    tasks = [
        DeliveryTask("TASK_001", "COASTAL_01", 15000, SchedulePriority.HIGH,
                    datetime.now(), datetime.now() + timedelta(days=2), 3.0),
        DeliveryTask("TASK_002", "OFFSHORE_01", 8000, SchedulePriority.CRITICAL,
                    datetime.now() + timedelta(hours=6), datetime.now() + timedelta(days=1), 4.0),
        DeliveryTask("TASK_003", "FLOATING_01", 3000, SchedulePriority.NORMAL,
                    datetime.now() + timedelta(hours=12), datetime.now() + timedelta(days=3), 2.0)
    ]
    
    for task in tasks:
        model.add_delivery_task(task)
    
    return model

# Example usage
if __name__ == "__main__":
    model = create_sample_maritime_network()
    
    print("Wave Scheduling Model Test:")
    print(f"Maritime locations: {len(model.maritime_locations)}")
    print(f"Vessels: {len(model.vessels)}")
    print(f"Delivery tasks: {len(model.delivery_tasks)}")
    
    # Generate sample weather data
    from algorithms.memetic_algorithm import create_sample_wave_data
    wave_data = create_sample_wave_data(datetime.now(), days=3)
    model.generate_weather_windows(wave_data, 72)
    
    print(f"Weather windows: {len(model.weather_windows)}")
    
    # Create schedule
    schedule = model.create_delivery_schedule()
    
    print(f"\nScheduling Results:")
    print(f"Scheduled deliveries: {schedule['summary']['scheduled_deliveries']}")
    print(f"Total cost: IDR {schedule['summary']['total_cost']:,.0f}")
    print(f"Schedule efficiency: {schedule['summary']['schedule_efficiency']:.1%}")
    print(f"Average weather score: {schedule['summary']['avg_weather_score']:.2f}")
    
    if schedule['scheduled_tasks']:
        print(f"\nFirst scheduled delivery:")
        first_task = schedule['scheduled_tasks'][0]
        print(f"  {first_task['destination']}: {first_task['scheduled_departure']}")
        print(f"  Vessel: {first_task['vessel_type']}")
        print(f"  Wave height: {first_task['weather_conditions']['wave_height']:.1f}m")
    
    print("\nWave scheduling model test completed! ⚓")