"""
BBM Distribution Network Model
Modeling distribution infrastructure, constraints, and logistics for MARLOFIR-P optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import math
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationType(Enum):
    """Types of locations in distribution network"""
    DEPOT = "depot"
    SPBU = "spbu"
    WAREHOUSE = "warehouse"
    DISTRIBUTION_CENTER = "distribution_center"

class VehicleType(Enum):
    """Types of vehicles for fuel transport"""
    TANKER_SMALL = "tanker_small"    # 5,000L
    TANKER_MEDIUM = "tanker_medium"  # 10,000L
    TANKER_LARGE = "tanker_large"    # 20,000L
    PIPELINE = "pipeline"            # Fixed infrastructure

@dataclass
class Location:
    """Location in distribution network"""
    id: str
    name: str
    type: LocationType
    latitude: float
    longitude: float
    address: str = ""
    capacity: float = 0.0  # Storage capacity in liters
    current_stock: float = 0.0  # Current fuel stock
    daily_demand: float = 0.0  # Average daily demand
    min_stock_level: float = 0.0  # Minimum stock threshold
    max_stock_level: float = 0.0  # Maximum stock threshold
    time_windows: List[Tuple[str, str]] = field(default_factory=list)  # Operating hours
    service_time: float = 30.0  # Service time in minutes
    priority: int = 1  # Delivery priority (1=highest)
    
    def __post_init__(self):
        if not self.max_stock_level:
            self.max_stock_level = self.capacity * 0.9
        if not self.min_stock_level:
            self.min_stock_level = self.capacity * 0.2

@dataclass
class Vehicle:
    """Vehicle for fuel transport"""
    id: str
    type: VehicleType
    capacity: float  # Fuel capacity in liters
    fuel_efficiency: float  # km per liter
    speed: float  # Average speed km/h
    operating_cost_per_km: float  # Cost per kilometer
    maintenance_cost_per_day: float  # Daily maintenance cost
    driver_cost_per_hour: float  # Driver cost per hour
    max_working_hours: float = 8.0  # Maximum working hours per day
    available_from: str = "06:00"  # Available from time
    available_until: str = "18:00"  # Available until time
    current_location: str = ""  # Current location ID
    
    @property
    def total_operating_cost_per_km(self) -> float:
        """Calculate total operating cost per kilometer"""
        return self.operating_cost_per_km + (self.driver_cost_per_hour / self.speed)

@dataclass
class Route:
    """Route between two locations"""
    from_location: str
    to_location: str
    distance: float  # Distance in kilometers
    travel_time: float  # Travel time in minutes
    fuel_cost: float  # Fuel cost for the route
    road_condition: str = "good"  # Road condition
    traffic_factor: float = 1.0  # Traffic multiplier
    toll_cost: float = 0.0  # Toll charges
    
    @property
    def total_cost(self) -> float:
        """Calculate total route cost"""
        return self.fuel_cost + self.toll_cost
    
    @property
    def effective_travel_time(self) -> float:
        """Calculate effective travel time with traffic"""
        return self.travel_time * self.traffic_factor

class DistributionNetwork:
    """
    Complete distribution network model for BBM logistics
    Manages locations, vehicles, routes, and constraints
    """
    
    def __init__(self, network_name: str = "BBM_Distribution_Network"):
        self.network_name = network_name
        self.locations: Dict[str, Location] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.routes: Dict[Tuple[str, str], Route] = {}
        self.distance_matrix: Dict[Tuple[str, str], float] = {}
        self.time_matrix: Dict[Tuple[str, str], float] = {}
        self.cost_matrix: Dict[Tuple[str, str], float] = {}
        
        # Network parameters
        self.fuel_price_per_liter = 10000  # IDR per liter
        self.default_fuel_efficiency = 3.5  # km/liter for trucks
        self.working_days_per_week = 6
        self.shifts_per_day = 1
        
        logger.info(f"Distribution network '{network_name}' initialized")
    
    def add_location(self, location: Location) -> None:
        """Add location to network"""
        self.locations[location.id] = location
        logger.info(f"Added location: {location.name} ({location.type.value})")
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add vehicle to fleet"""
        self.vehicles[vehicle.id] = vehicle
        logger.info(f"Added vehicle: {vehicle.id} ({vehicle.type.value})")
    
    def add_route(self, route: Route) -> None:
        """Add route to network"""
        key = (route.from_location, route.to_location)
        self.routes[key] = route
        
        # Update matrices
        self.distance_matrix[key] = route.distance
        self.time_matrix[key] = route.effective_travel_time
        self.cost_matrix[key] = route.total_cost
        
        logger.debug(f"Added route: {route.from_location} -> {route.to_location}")
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two coordinates
        
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def generate_distance_matrix(self) -> None:
        """Generate distance matrix for all location pairs"""
        logger.info("Generating distance matrix...")
        
        location_ids = list(self.locations.keys())
        
        for i, loc1_id in enumerate(location_ids):
            for j, loc2_id in enumerate(location_ids):
                if i != j:
                    loc1 = self.locations[loc1_id]
                    loc2 = self.locations[loc2_id]
                    
                    # Check if route already exists
                    if (loc1_id, loc2_id) not in self.routes:
                        # Calculate distance
                        distance = self.calculate_distance(
                            loc1.latitude, loc1.longitude,
                            loc2.latitude, loc2.longitude
                        )
                        
                        # Estimate travel time (assuming average speed 40 km/h)
                        travel_time = (distance / 40) * 60  # minutes
                        
                        # Calculate fuel cost
                        fuel_needed = distance / self.default_fuel_efficiency
                        fuel_cost = fuel_needed * self.fuel_price_per_liter
                        
                        # Create route
                        route = Route(
                            from_location=loc1_id,
                            to_location=loc2_id,
                            distance=distance,
                            travel_time=travel_time,
                            fuel_cost=fuel_cost
                        )
                        
                        self.add_route(route)
        
        logger.info(f"Distance matrix generated for {len(location_ids)} locations")
    
    def get_depot_locations(self) -> List[Location]:
        """Get all depot locations"""
        return [loc for loc in self.locations.values() if loc.type == LocationType.DEPOT]
    
    def get_spbu_locations(self) -> List[Location]:
        """Get all SPBU locations"""
        return [loc for loc in self.locations.values() if loc.type == LocationType.SPBU]
    
    def get_available_vehicles(self, location_id: str = None) -> List[Vehicle]:
        """Get available vehicles at specific location or all"""
        if location_id:
            return [v for v in self.vehicles.values() if v.current_location == location_id]
        return list(self.vehicles.values())
    
    def calculate_demand_urgency(self, location_id: str) -> float:
        """
        Calculate demand urgency for a location
        
        Returns:
            Urgency score (0-1, 1 = most urgent)
        """
        if location_id not in self.locations:
            return 0.0
        
        location = self.locations[location_id]
        
        # Stock level ratio
        stock_ratio = location.current_stock / location.capacity if location.capacity > 0 else 0
        
        # Demand rate (daily demand / capacity)
        demand_rate = location.daily_demand / location.capacity if location.capacity > 0 else 0
        
        # Days until empty
        days_until_empty = location.current_stock / location.daily_demand if location.daily_demand > 0 else float('inf')
        
        # Calculate urgency
        if days_until_empty <= 1:
            urgency = 1.0  # Critical
        elif days_until_empty <= 2:
            urgency = 0.8  # High
        elif days_until_empty <= 3:
            urgency = 0.6  # Medium
        elif stock_ratio < 0.3:
            urgency = 0.4  # Low
        else:
            urgency = 0.2  # Normal
        
        return urgency
    
    def get_route_cost(self, from_location: str, to_location: str, vehicle_id: str) -> float:
        """
        Calculate total route cost for specific vehicle
        
        Args:
            from_location: Starting location ID
            to_location: Destination location ID
            vehicle_id: Vehicle ID
            
        Returns:
            Total cost for the route
        """
        route_key = (from_location, to_location)
        
        if route_key not in self.routes or vehicle_id not in self.vehicles:
            return float('inf')
        
        route = self.routes[route_key]
        vehicle = self.vehicles[vehicle_id]
        
        # Base route cost
        base_cost = route.total_cost
        
        # Vehicle operating cost
        vehicle_cost = vehicle.total_operating_cost_per_km * route.distance
        
        # Time-based cost
        travel_hours = route.effective_travel_time / 60
        time_cost = vehicle.driver_cost_per_hour * travel_hours
        
        return base_cost + vehicle_cost + time_cost
    
    def validate_vehicle_capacity(self, vehicle_id: str, demand: float) -> bool:
        """Check if vehicle can handle the demand"""
        if vehicle_id not in self.vehicles:
            return False
        
        vehicle = self.vehicles[vehicle_id]
        return vehicle.capacity >= demand
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get network summary statistics"""
        total_locations = len(self.locations)
        depot_count = len(self.get_depot_locations())
        spbu_count = len(self.get_spbu_locations())
        vehicle_count = len(self.vehicles)
        route_count = len(self.routes)
        
        # Calculate total capacity and demand
        total_capacity = sum(loc.capacity for loc in self.locations.values())
        total_demand = sum(loc.daily_demand for loc in self.locations.values())
        total_stock = sum(loc.current_stock for loc in self.locations.values())
        
        # Vehicle fleet summary
        fleet_capacity = sum(v.capacity for v in self.vehicles.values())
        
        return {
            'network_name': self.network_name,
            'total_locations': total_locations,
            'depot_count': depot_count,
            'spbu_count': spbu_count,
            'vehicle_count': vehicle_count,
            'route_count': route_count,
            'total_storage_capacity': total_capacity,
            'total_daily_demand': total_demand,
            'total_current_stock': total_stock,
            'fleet_capacity': fleet_capacity,
            'stock_ratio': total_stock / total_capacity if total_capacity > 0 else 0,
            'demand_coverage_days': total_stock / total_demand if total_demand > 0 else float('inf')
        }
    
    def export_network_data(self) -> Dict[str, Any]:
        """Export complete network data for GA optimization"""
        # Location data
        location_data = {}
        for loc_id, location in self.locations.items():
            location_data[loc_id] = {
                'name': location.name,
                'type': location.type.value,
                'coordinates': [location.latitude, location.longitude],
                'capacity': location.capacity,
                'current_stock': location.current_stock,
                'daily_demand': location.daily_demand,
                'urgency': self.calculate_demand_urgency(loc_id),
                'priority': location.priority
            }
        
        # Vehicle data
        vehicle_data = {}
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle_data[vehicle_id] = {
                'type': vehicle.type.value,
                'capacity': vehicle.capacity,
                'operating_cost_per_km': vehicle.total_operating_cost_per_km,
                'current_location': vehicle.current_location
            }
        
        # Distance and cost matrices
        matrices = {
            'distances': self.distance_matrix,
            'travel_times': self.time_matrix,
            'costs': self.cost_matrix
        }
        
        return {
            'locations': location_data,
            'vehicles': vehicle_data,
            'matrices': matrices,
            'network_summary': self.get_network_summary()
        }

# Factory functions for creating common network configurations
def create_jakarta_bbm_network() -> DistributionNetwork:
    """Create sample Jakarta BBM distribution network"""
    network = DistributionNetwork("Jakarta_BBM_Network")
    
    # Add depot
    depot = Location(
        id="DEPOT_01",
        name="Depot Plumpang",
        type=LocationType.DEPOT,
        latitude=-6.1167,
        longitude=106.8833,
        capacity=100000000,  # 100 million liters
        current_stock=80000000,
        address="Plumpang, Jakarta Utara"
    )
    network.add_location(depot)
    
    # Add SPBU locations
    spbu_data = [
        ("SPBU_001", "SPBU Sudirman", -6.2088, 106.8456, 50000, "Jl. Sudirman, Jakarta"),
        ("SPBU_002", "SPBU Gatot Subroto", -6.2297, 106.8206, 45000, "Jl. Gatot Subroto, Jakarta"),
        ("SPBU_003", "SPBU Kuningan", -6.2383, 106.8306, 40000, "Jl. Kuningan, Jakarta"),
        ("SPBU_004", "SPBU Senayan", -6.2297, 106.8019, 55000, "Jl. Senayan, Jakarta"),
        ("SPBU_005", "SPBU Kemang", -6.2614, 106.8147, 35000, "Jl. Kemang, Jakarta"),
    ]
    
    for spbu_id, name, lat, lon, capacity, address in spbu_data:
        spbu = Location(
            id=spbu_id,
            name=name,
            type=LocationType.SPBU,
            latitude=lat,
            longitude=lon,
            capacity=capacity,
            current_stock=capacity * 0.4,  # 40% stock
            daily_demand=capacity * 0.15,  # 15% daily turnover
            address=address
        )
        network.add_location(spbu)
    
    # Add vehicles
    vehicle_types = [
        ("TRUCK_001", VehicleType.TANKER_MEDIUM, 10000, 3.5),
        ("TRUCK_002", VehicleType.TANKER_MEDIUM, 10000, 3.5),
        ("TRUCK_003", VehicleType.TANKER_LARGE, 20000, 3.0),
        ("TRUCK_004", VehicleType.TANKER_SMALL, 5000, 4.0),
    ]
    
    for vehicle_id, vtype, capacity, efficiency in vehicle_types:
        vehicle = Vehicle(
            id=vehicle_id,
            type=vtype,
            capacity=capacity,
            fuel_efficiency=efficiency,
            speed=40,  # 40 km/h average
            operating_cost_per_km=2000,  # IDR 2000 per km
            maintenance_cost_per_day=50000,  # IDR 50k per day
            driver_cost_per_hour=25000,  # IDR 25k per hour
            current_location="DEPOT_01"
        )
        network.add_vehicle(vehicle)
    
    # Generate distance matrix
    network.generate_distance_matrix()
    
    logger.info("Jakarta BBM network created successfully")
    return network

# Example usage and testing
if __name__ == "__main__":
    # Create sample network
    network = create_jakarta_bbm_network()
    
    # Print network summary
    summary = network.get_network_summary()
    print("Network Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test urgency calculation
    print("\nLocation Urgency Scores:")
    for loc_id in network.locations:
        urgency = network.calculate_demand_urgency(loc_id)
        print(f"  {network.locations[loc_id].name}: {urgency:.2f}")
    
    # Export data for GA
    export_data = network.export_network_data()
    print(f"\nExport data keys: {list(export_data.keys())}")