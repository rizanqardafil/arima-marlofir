"""
Vehicle Routing Problem (VRP) Solver for BBM Distribution
Specialized algorithms for fuel delivery optimization with capacity and time constraints
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import copy
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from models.distribution_model import DistributionNetwork, Location, Vehicle, LocationType
    from utils.geographic_utils import GeographicCalculator, Coordinate
except ImportError:
    # For standalone testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VRPType(Enum):
    """Types of Vehicle Routing Problems"""
    CVRP = "capacitated_vrp"           # Capacitated VRP
    VRPTW = "vrp_time_windows"         # VRP with Time Windows
    MDVRP = "multi_depot_vrp"          # Multi-Depot VRP
    VRPB = "vrp_backhauls"             # VRP with Backhauls
    VRPPD = "vrp_pickup_delivery"      # VRP with Pickup and Delivery
    HFVRP = "heterogeneous_fleet_vrp"  # Heterogeneous Fleet VRP

class SolutionMethod(Enum):
    """VRP solution methods"""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    SAVINGS_ALGORITHM = "savings_algorithm"
    SWEEP_ALGORITHM = "sweep_algorithm"
    TWO_OPT = "two_opt"
    THREE_OPT = "three_opt"
    OR_OPT = "or_opt"
    SIMULATED_ANNEALING = "simulated_annealing"
    TABU_SEARCH = "tabu_search"

@dataclass
class VRPSolution:
    """VRP solution representation"""
    routes: List[List[str]]  # List of routes (each route is list of location IDs)
    vehicle_assignments: Dict[str, str]  # Vehicle ID -> Route mapping
    total_cost: float = 0.0
    total_distance: float = 0.0
    total_time: float = 0.0
    num_vehicles_used: int = 0
    capacity_utilization: float = 0.0
    solution_quality: float = 0.0
    computation_time: float = 0.0
    method_used: str = ""
    feasible: bool = True
    violations: List[str] = field(default_factory=list)

@dataclass
class VRPConstraints:
    """VRP constraints configuration"""
    max_route_time: float = 480.0  # 8 hours in minutes
    max_route_distance: float = 500.0  # Maximum route distance in km
    time_windows_enabled: bool = False
    capacity_constraints: bool = True
    driver_break_time: float = 60.0  # Break time in minutes
    service_time_per_stop: float = 30.0  # Service time per location in minutes
    fuel_consumption_rate: float = 0.3  # Liters per km
    overtime_penalty: float = 100.0  # Penalty per minute of overtime

class VehicleRoutingProblem:
    """
    Comprehensive Vehicle Routing Problem solver for BBM distribution
    Supports multiple VRP variants and solution methods
    """
    
    def __init__(self, distribution_network: DistributionNetwork):
        self.network = distribution_network
        self.geo_calculator = GeographicCalculator()
        self.constraints = VRPConstraints()
        self.solution_cache = {}
        
        # Performance tracking
        self.performance_stats = {
            'solutions_computed': 0,
            'avg_computation_time': 0.0,
            'best_solution_quality': 0.0,
            'cache_hits': 0
        }
        
        logger.info("VRP solver initialized")
    
    def set_constraints(self, constraints: VRPConstraints) -> None:
        """Set VRP constraints"""
        self.constraints = constraints
        logger.info("VRP constraints updated")
    
    def solve_cvrp(self, depot_id: str, customer_locations: List[str],
                   demands: Dict[str, float], vehicle_capacity: float,
                   method: SolutionMethod = SolutionMethod.SAVINGS_ALGORITHM) -> VRPSolution:
        """
        Solve Capacitated Vehicle Routing Problem (CVRP)
        
        Args:
            depot_id: Depot location ID
            customer_locations: List of customer location IDs
            demands: Demand for each customer location
            vehicle_capacity: Vehicle capacity constraint
            method: Solution method
            
        Returns:
            VRP solution
        """
        start_time = time.time()
        
        logger.info(f"Solving CVRP with {len(customer_locations)} customers using {method.value}")
        
        # Cache key for solution caching
        cache_key = f"cvrp_{depot_id}_{len(customer_locations)}_{vehicle_capacity}_{method.value}"
        
        if cache_key in self.solution_cache:
            self.performance_stats['cache_hits'] += 1
            logger.info("Solution found in cache")
            return self.solution_cache[cache_key]
        
        try:
            if method == SolutionMethod.NEAREST_NEIGHBOR:
                solution = self._solve_nearest_neighbor(depot_id, customer_locations, demands, vehicle_capacity)
            elif method == SolutionMethod.SAVINGS_ALGORITHM:
                solution = self._solve_savings_algorithm(depot_id, customer_locations, demands, vehicle_capacity)
            elif method == SolutionMethod.SWEEP_ALGORITHM:
                solution = self._solve_sweep_algorithm(depot_id, customer_locations, demands, vehicle_capacity)
            else:
                # Default to savings algorithm
                solution = self._solve_savings_algorithm(depot_id, customer_locations, demands, vehicle_capacity)
            
            # Post-process solution
            solution = self._post_process_solution(solution, depot_id)
            
            # Apply local search improvements
            solution = self._apply_local_search(solution, method)
            
            # Calculate final metrics
            solution.computation_time = time.time() - start_time
            solution.method_used = method.value
            
            # Cache solution
            self.solution_cache[cache_key] = solution
            
            # Update performance stats
            self.performance_stats['solutions_computed'] += 1
            self.performance_stats['avg_computation_time'] = (
                (self.performance_stats['avg_computation_time'] * 
                 (self.performance_stats['solutions_computed'] - 1) + solution.computation_time) /
                self.performance_stats['solutions_computed']
            )
            
            if solution.solution_quality > self.performance_stats['best_solution_quality']:
                self.performance_stats['best_solution_quality'] = solution.solution_quality
            
            logger.info(f"CVRP solved in {solution.computation_time:.2f}s with {solution.num_vehicles_used} vehicles")
            return solution
        
        except Exception as e:
            logger.error(f"CVRP solving failed: {str(e)}")
            return self._create_fallback_solution(depot_id, customer_locations, str(e))
    
    def _solve_nearest_neighbor(self, depot_id: str, customers: List[str],
                               demands: Dict[str, float], capacity: float) -> VRPSolution:
        """Solve using Nearest Neighbor heuristic"""
        routes = []
        unvisited = customers.copy()
        
        while unvisited:
            # Start new route
            current_route = [depot_id]
            current_location = depot_id
            current_load = 0.0
            route_distance = 0.0
            
            while unvisited:
                # Find feasible nearest neighbor
                best_customer = None
                best_distance = float('inf')
                
                for customer in unvisited:
                    # Check capacity constraint
                    customer_demand = demands.get(customer, 0)
                    if current_load + customer_demand > capacity:
                        continue
                    
                    # Calculate distance
                    distance = self._get_distance(current_location, customer)
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer
                
                if best_customer is None:
                    break  # No feasible customer found
                
                # Add customer to route
                current_route.append(best_customer)
                current_location = best_customer
                current_load += demands.get(best_customer, 0)
                route_distance += best_distance
                unvisited.remove(best_customer)
            
            # Return to depot
            current_route.append(depot_id)
            route_distance += self._get_distance(current_location, depot_id)
            
            routes.append(current_route)
        
        return self._create_solution_from_routes(routes, demands)
    
    def _solve_savings_algorithm(self, depot_id: str, customers: List[str],
                                demands: Dict[str, float], capacity: float) -> VRPSolution:
        """Solve using Clarke-Wright Savings Algorithm"""
        # Calculate savings matrix
        savings = {}
        
        for i, customer_i in enumerate(customers):
            for j, customer_j in enumerate(customers[i+1:], i+1):
                # Savings = distance(depot, i) + distance(depot, j) - distance(i, j)
                dist_depot_i = self._get_distance(depot_id, customer_i)
                dist_depot_j = self._get_distance(depot_id, customer_j)
                dist_i_j = self._get_distance(customer_i, customer_j)
                
                saving = dist_depot_i + dist_depot_j - dist_i_j
                savings[(customer_i, customer_j)] = saving
        
        # Sort savings in descending order
        sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize routes (each customer in separate route)
        routes = {}
        route_loads = {}
        customer_route_map = {}
        
        for i, customer in enumerate(customers):
            route_id = i
            routes[route_id] = [depot_id, customer, depot_id]
            route_loads[route_id] = demands.get(customer, 0)
            customer_route_map[customer] = route_id
        
        # Process savings
        for (customer_i, customer_j), saving in sorted_savings:
            route_i = customer_route_map.get(customer_i)
            route_j = customer_route_map.get(customer_j)
            
            # Skip if customers are in same route
            if route_i == route_j:
                continue
            
            # Check if routes can be merged
            if route_i is not None and route_j is not None:
                combined_load = route_loads[route_i] + route_loads[route_j]
                
                if combined_load <= capacity:
                    # Merge routes
                    route_i_customers = routes[route_i][1:-1]  # Exclude depot
                    route_j_customers = routes[route_j][1:-1]
                    
                    # Determine merge order
                    if self._can_merge_routes(routes[route_i], routes[route_j], customer_i, customer_j):
                        # Merge route_j into route_i
                        merged_route = self._merge_routes(routes[route_i], routes[route_j], customer_i, customer_j)
                        
                        # Update data structures
                        routes[route_i] = merged_route
                        route_loads[route_i] = combined_load
                        
                        # Update customer mappings
                        for customer in route_j_customers:
                            customer_route_map[customer] = route_i
                        
                        # Remove merged route
                        del routes[route_j]
                        del route_loads[route_j]
        
        # Convert to final format
        final_routes = list(routes.values())
        return self._create_solution_from_routes(final_routes, demands)
    
    def _solve_sweep_algorithm(self, depot_id: str, customers: List[str],
                              demands: Dict[str, float], capacity: float) -> VRPSolution:
        """Solve using Sweep Algorithm"""
        # Get depot coordinates
        depot_location = self.network.locations[depot_id]
        depot_coord = Coordinate(depot_location.latitude, depot_location.longitude)
        
        # Calculate angles for all customers relative to depot
        customer_angles = []
        for customer in customers:
            customer_location = self.network.locations[customer]
            customer_coord = Coordinate(customer_location.latitude, customer_location.longitude)
            
            # Calculate bearing (angle)
            bearing = self.geo_calculator.calculate_bearing(depot_coord, customer_coord)
            customer_angles.append((customer, bearing, demands.get(customer, 0)))
        
        # Sort customers by angle
        customer_angles.sort(key=lambda x: x[1])
        
        # Create routes using sweep
        routes = []
        current_route = [depot_id]
        current_load = 0.0
        
        for customer, angle, demand in customer_angles:
            if current_load + demand <= capacity:
                # Add to current route
                current_route.append(customer)
                current_load += demand
            else:
                # Start new route
                current_route.append(depot_id)  # Return to depot
                routes.append(current_route)
                
                # Start new route
                current_route = [depot_id, customer]
                current_load = demand
        
        # Add final route
        if len(current_route) > 1:
            current_route.append(depot_id)
            routes.append(current_route)
        
        return self._create_solution_from_routes(routes, demands)
    
    def _can_merge_routes(self, route1: List[str], route2: List[str],
                         customer_i: str, customer_j: str) -> bool:
        """Check if two routes can be merged at specified customers"""
        # Check if customers are at the ends of their respective routes
        route1_customers = route1[1:-1]  # Exclude depot
        route2_customers = route2[1:-1]
        
        # Customer_i should be at end of route1, customer_j at start of route2
        # or vice versa
        return ((route1_customers[-1] == customer_i and route2_customers[0] == customer_j) or
                (route1_customers[-1] == customer_j and route2_customers[0] == customer_i) or
                (route1_customers[0] == customer_i and route2_customers[-1] == customer_j) or
                (route1_customers[0] == customer_j and route2_customers[-1] == customer_i))
    
    def _merge_routes(self, route1: List[str], route2: List[str],
                     customer_i: str, customer_j: str) -> List[str]:
        """Merge two routes at specified connection points"""
        route1_customers = route1[1:-1]
        route2_customers = route2[1:-1]
        
        # Determine merge order
        if route1_customers[-1] == customer_i and route2_customers[0] == customer_j:
            merged = [route1[0]] + route1_customers + route2_customers + [route1[0]]
        elif route1_customers[-1] == customer_j and route2_customers[0] == customer_i:
            merged = [route1[0]] + route1_customers + route2_customers + [route1[0]]
        elif route1_customers[0] == customer_i and route2_customers[-1] == customer_j:
            merged = [route1[0]] + route2_customers + route1_customers + [route1[0]]
        elif route1_customers[0] == customer_j and route2_customers[-1] == customer_i:
            merged = [route1[0]] + route2_customers + route1_customers + [route1[0]]
        else:
            # Default: concatenate route1 + route2
            merged = [route1[0]] + route1_customers + route2_customers + [route1[0]]
        
        return merged
    
    def _apply_local_search(self, solution: VRPSolution, method: SolutionMethod) -> VRPSolution:
        """Apply local search improvements to solution"""
        improved_solution = copy.deepcopy(solution)
        
        # Apply 2-opt improvement
        improved_solution = self._apply_two_opt(improved_solution)
        
        # Apply Or-opt improvement
        improved_solution = self._apply_or_opt(improved_solution)
        
        # Recalculate metrics
        improved_solution = self._calculate_solution_metrics(improved_solution)
        
        return improved_solution
    
    def _apply_two_opt(self, solution: VRPSolution) -> VRPSolution:
        """Apply 2-opt local search to each route"""
        improved = False
        
        for route_idx, route in enumerate(solution.routes):
            if len(route) <= 4:  # Skip short routes (depot-customer-depot)
                continue
            
            best_route = route.copy()
            best_distance = self._calculate_route_distance(route)
            
            # Try all 2-opt swaps
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Create 2-opt swap
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_distance = self._calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
            
            solution.routes[route_idx] = best_route
        
        if improved:
            logger.debug("2-opt improvement applied")
        
        return solution
    
    def _apply_or_opt(self, solution: VRPSolution) -> VRPSolution:
        """Apply Or-opt local search (relocate sequence of customers)"""
        improved = False
        
        for route_idx, route in enumerate(solution.routes):
            if len(route) <= 4:
                continue
            
            best_route = route.copy()
            best_distance = self._calculate_route_distance(route)
            
            # Try relocating sequences of 1, 2, or 3 customers
            for seq_len in [1, 2, 3]:
                for i in range(1, len(route) - seq_len):
                    if i + seq_len >= len(route) - 1:
                        break
                    
                    # Extract sequence
                    sequence = route[i:i+seq_len]
                    remaining = route[:i] + route[i+seq_len:]
                    
                    # Try inserting sequence at different positions
                    for j in range(1, len(remaining)):
                        new_route = remaining[:j] + sequence + remaining[j:]
                        new_distance = self._calculate_route_distance(new_route)
                        
                        if new_distance < best_distance:
                            best_route = new_route
                            best_distance = new_distance
                            improved = True
            
            solution.routes[route_idx] = best_route
        
        if improved:
            logger.debug("Or-opt improvement applied")
        
        return solution
    
    def _get_distance(self, location1: str, location2: str) -> float:
        """Get distance between two locations"""
        if (location1, location2) in self.network.distance_matrix:
            return self.network.distance_matrix[(location1, location2)]
        elif (location2, location1) in self.network.distance_matrix:
            return self.network.distance_matrix[(location2, location1)]
        else:
            # Calculate on-the-fly if not in matrix
            loc1 = self.network.locations[location1]
            loc2 = self.network.locations[location2]
            coord1 = Coordinate(loc1.latitude, loc1.longitude)
            coord2 = Coordinate(loc2.latitude, loc2.longitude)
            return self.geo_calculator.calculate_distance(coord1, coord2)
    
    def _calculate_route_distance(self, route: List[str]) -> float:
        """Calculate total distance for a route"""
        total_distance = 0.0
        
        for i in range(len(route) - 1):
            total_distance += self._get_distance(route[i], route[i + 1])
        
        return total_distance
    
    def _calculate_route_time(self, route: List[str]) -> float:
        """Calculate total time for a route including service time"""
        total_time = 0.0
        
        # Travel time
        for i in range(len(route) - 1):
            distance = self._get_distance(route[i], route[i + 1])
            # Assume average speed of 40 km/h
            travel_time = (distance / 40) * 60  # Convert to minutes
            total_time += travel_time
        
        # Service time (exclude depot)
        service_stops = len([loc for loc in route if loc in self.network.locations 
                           and self.network.locations[loc].type != LocationType.DEPOT])
        total_time += service_stops * self.constraints.service_time_per_stop
        
        return total_time
    
    def _create_solution_from_routes(self, routes: List[List[str]], 
                                   demands: Dict[str, float]) -> VRPSolution:
        """Create VRPSolution object from routes"""
        solution = VRPSolution(routes=routes)
        
        # Calculate metrics
        solution = self._calculate_solution_metrics(solution, demands)
        
        return solution
    
    def _calculate_solution_metrics(self, solution: VRPSolution, 
                                  demands: Dict[str, float] = None) -> VRPSolution:
        """Calculate all solution metrics"""
        total_distance = 0.0
        total_time = 0.0
        total_demand_served = 0.0
        total_capacity_used = 0.0
        
        for route in solution.routes:
            # Route distance and time
            route_distance = self._calculate_route_distance(route)
            route_time = self._calculate_route_time(route)
            
            total_distance += route_distance
            total_time += route_time
            
            # Route demand
            if demands:
                route_demand = sum(demands.get(loc, 0) for loc in route 
                                 if loc in self.network.locations 
                                 and self.network.locations[loc].type != LocationType.DEPOT)
                total_demand_served += route_demand
        
        # Update solution metrics
        solution.total_distance = total_distance
        solution.total_time = total_time
        solution.num_vehicles_used = len(solution.routes)
        
        # Calculate cost (simplified)
        fuel_cost = total_distance * self.constraints.fuel_consumption_rate * self.network.fuel_price_per_liter / 1000
        time_cost = total_time * 500  # IDR 500 per minute
        solution.total_cost = fuel_cost + time_cost
        
        # Calculate capacity utilization
        if self.network.vehicles:
            avg_vehicle_capacity = np.mean([v.capacity for v in self.network.vehicles.values()])
            solution.capacity_utilization = total_demand_served / (solution.num_vehicles_used * avg_vehicle_capacity)
        
        # Calculate solution quality (higher is better)
        if solution.num_vehicles_used > 0:
            solution.solution_quality = (total_demand_served / (solution.total_cost + 1)) * 1000
        
        # Check feasibility
        solution.feasible = self._check_solution_feasibility(solution)
        
        return solution
    
    def _check_solution_feasibility(self, solution: VRPSolution) -> bool:
        """Check if solution satisfies all constraints"""
        violations = []
        
        for i, route in enumerate(solution.routes):
            # Check route time constraint
            route_time = self._calculate_route_time(route)
            if route_time > self.constraints.max_route_time:
                violations.append(f"Route {i}: Time constraint violation ({route_time:.1f} > {self.constraints.max_route_time})")
            
            # Check route distance constraint
            route_distance = self._calculate_route_distance(route)
            if route_distance > self.constraints.max_route_distance:
                violations.append(f"Route {i}: Distance constraint violation ({route_distance:.1f} > {self.constraints.max_route_distance})")
        
        solution.violations = violations
        return len(violations) == 0
    
    def _post_process_solution(self, solution: VRPSolution, depot_id: str) -> VRPSolution:
        """Post-process solution to ensure consistency"""
        # Ensure all routes start and end at depot
        for i, route in enumerate(solution.routes):
            if not route:
                continue
            
            if route[0] != depot_id:
                route.insert(0, depot_id)
            
            if route[-1] != depot_id:
                route.append(depot_id)
            
            solution.routes[i] = route
        
        # Remove empty routes
        solution.routes = [route for route in solution.routes if len(route) > 2]
        
        return solution
    
    def _create_fallback_solution(self, depot_id: str, customers: List[str], 
                                 error_msg: str) -> VRPSolution:
        """Create fallback solution when optimization fails"""
        # Create simple one-customer-per-route solution
        routes = []
        for customer in customers:
            routes.append([depot_id, customer, depot_id])
        
        solution = VRPSolution(
            routes=routes,
            feasible=False,
            violations=[f"Optimization failed: {error_msg}"]
        )
        
        return self._calculate_solution_metrics(solution)
    
    def solve_vrptw(self, depot_id: str, customers: List[str], demands: Dict[str, float],
                    vehicle_capacity: float, time_windows: Dict[str, Tuple[float, float]]) -> VRPSolution:
        """
        Solve VRP with Time Windows
        
        Args:
            depot_id: Depot location
            customers: Customer locations
            demands: Customer demands
            vehicle_capacity: Vehicle capacity
            time_windows: Time windows for each location (start_time, end_time)
            
        Returns:
            VRP solution with time window compliance
        """
        logger.info(f"Solving VRPTW with {len(customers)} customers")
        
        # Enable time window constraints
        old_constraints = self.constraints
        self.constraints.time_windows_enabled = True
        
        try:
            # Use savings algorithm with time window checks
            solution = self._solve_savings_algorithm_with_tw(depot_id, customers, demands, 
                                                           vehicle_capacity, time_windows)
            solution.method_used = "savings_with_time_windows"
            
            return solution
        
        finally:
            # Restore original constraints
            self.constraints = old_constraints
    
    def _solve_savings_algorithm_with_tw(self, depot_id: str, customers: List[str],
                                       demands: Dict[str, float], capacity: float,
                                       time_windows: Dict[str, Tuple[float, float]]) -> VRPSolution:
        """Savings algorithm with time window constraints"""
        # Start with basic savings solution
        basic_solution = self._solve_savings_algorithm(depot_id, customers, demands, capacity)
        
        # Adjust routes to satisfy time windows
        adjusted_routes = []
        
        for route in basic_solution.routes:
            # Check time window feasibility
            if self._is_route_time_feasible(route, time_windows):
                adjusted_routes.append(route)
            else:
                # Split route to satisfy time windows
                split_routes = self._split_route_for_time_windows(route, time_windows, capacity, demands)
                adjusted_routes.extend(split_routes)
        
        solution = VRPSolution(routes=adjusted_routes)
        return self._calculate_solution_metrics(solution, demands)
    
    def _is_route_time_feasible(self, route: List[str], 
                               time_windows: Dict[str, Tuple[float, float]]) -> bool:
        """Check if route satisfies time window constraints"""
        current_time = 0.0  # Start at time 0
        
        for i in range(len(route) - 1):
            current_loc = route[i]
            next_loc = route[i + 1]
            
            # Travel time
            distance = self._get_distance(current_loc, next_loc)
            travel_time = (distance / 40) * 60  # minutes
            current_time += travel_time
            
            # Service time
            if next_loc in time_windows:
                tw_start, tw_end = time_windows[next_loc]
                
                # Check if we can arrive within time window
                if current_time > tw_end:
                    return False
                
                # Wait if we arrive too early
                if current_time < tw_start:
                    current_time = tw_start
                
                # Add service time
                current_time += self.constraints.service_time_per_stop
        
        return True
    
    def _split_route_for_time_windows(self, route: List[str], 
                                    time_windows: Dict[str, Tuple[float, float]],
                                    capacity: float, demands: Dict[str, float]) -> List[List[str]]:
        """Split route to satisfy time window constraints"""
        if len(route) <= 3:  # depot-customer-depot
            return [route]
        
        depot = route[0]
        customers = route[1:-1]
        split_routes = []
        current_route = [depot]
        current_time = 0.0
        current_load = 0.0
        
        for customer in customers:
            # Calculate arrival time at customer
            if current_route[-1] != depot:
                distance = self._get_distance(current_route[-1], customer)
                travel_time = (distance / 40) * 60
                arrival_time = current_time + travel_time
            else:
                distance = self._get_distance(depot, customer)
                travel_time = (distance / 40) * 60
                arrival_time = travel_time
            
            # Check time window and capacity
            customer_demand = demands.get(customer, 0)
            tw_start, tw_end = time_windows.get(customer, (0, float('inf')))
            
            if (arrival_time <= tw_end and 
                current_load + customer_demand <= capacity):
                # Add to current route
                current_route.append(customer)
                current_time = max(arrival_time, tw_start) + self.constraints.service_time_per_stop
                current_load += customer_demand
            else:
                # Start new route
                current_route.append(depot)
                split_routes.append(current_route)
                
                # Begin new route with this customer
                current_route = [depot, customer]
                distance = self._get_distance(depot, customer)
                travel_time = (distance / 40) * 60
                current_time = max(travel_time, tw_start) + self.constraints.service_time_per_stop
                current_load = customer_demand
        
        # Add final route
        if len(current_route) > 1:
            current_route.append(depot)
            split_routes.append(current_route)
        
        return split_routes
    
    def solve_mdvrp(self, depot_ids: List[str], customers: List[str],
                    demands: Dict[str, float], vehicle_capacities: Dict[str, float]) -> VRPSolution:
        """
        Solve Multi-Depot Vehicle Routing Problem
        
        Args:
            depot_ids: List of depot location IDs
            customers: Customer location IDs
            demands: Customer demands
            vehicle_capacities: Vehicle capacity per depot
            
        Returns:
            Multi-depot VRP solution
        """
        logger.info(f"Solving MDVRP with {len(depot_ids)} depots and {len(customers)} customers")
        
        # Assign customers to nearest depots
        customer_depot_assignment = {}
        for customer in customers:
            best_depot = None
            best_distance = float('inf')
            
            for depot in depot_ids:
                distance = self._get_distance(depot, customer)
                if distance < best_distance:
                    best_distance = distance
                    best_depot = depot
            
            customer_depot_assignment[customer] = best_depot
        
        # Solve CVRP for each depot
        all_routes = []
        total_cost = 0.0
        total_distance = 0.0
        total_time = 0.0
        
        for depot in depot_ids:
            # Get customers assigned to this depot
            depot_customers = [c for c, d in customer_depot_assignment.items() if d == depot]
            
            if not depot_customers:
                continue
            
            # Solve CVRP for this depot
            depot_capacity = vehicle_capacities.get(depot, 10000)
            depot_solution = self.solve_cvrp(depot, depot_customers, demands, depot_capacity)
            
            # Add routes to overall solution
            all_routes.extend(depot_solution.routes)
            total_cost += depot_solution.total_cost
            total_distance += depot_solution.total_distance
            total_time += depot_solution.total_time
        
        # Create combined solution
        solution = VRPSolution(
            routes=all_routes,
            total_cost=total_cost,
            total_distance=total_distance,
            total_time=total_time,
            num_vehicles_used=len(all_routes),
            method_used="multi_depot_vrp"
        )
        
        return self._calculate_solution_metrics(solution, demands)
    
    def solve_hfvrp(self, depot_id: str, customers: List[str], demands: Dict[str, float],
                    available_vehicles: Dict[str, Dict]) -> VRPSolution:
        """
        Solve Heterogeneous Fleet Vehicle Routing Problem
        
        Args:
            depot_id: Depot location
            customers: Customer locations
            demands: Customer demands
            available_vehicles: Dict of vehicle_id -> {capacity, cost_per_km, etc.}
            
        Returns:
            Heterogeneous fleet VRP solution
        """
        logger.info(f"Solving HFVRP with {len(available_vehicles)} vehicle types")
        
        # Sort vehicles by efficiency (capacity/cost ratio)
        vehicles_by_efficiency = sorted(
            available_vehicles.items(),
            key=lambda x: x[1]['capacity'] / x[1].get('cost_per_km', 1),
            reverse=True
        )
        
        routes = []
        vehicle_assignments = {}
        unassigned_customers = customers.copy()
        
        # Assign vehicles efficiently
        for vehicle_id, vehicle_info in vehicles_by_efficiency:
            if not unassigned_customers:
                break
            
            vehicle_capacity = vehicle_info['capacity']
            
            # Solve for this vehicle's capacity
            if len(unassigned_customers) <= 10:  # Use exact method for small problems
                vehicle_customers = self._select_customers_for_vehicle(
                    depot_id, unassigned_customers, demands, vehicle_capacity
                )
            else:  # Use heuristic for larger problems
                vehicle_customers = self._greedy_customer_selection(
                    depot_id, unassigned_customers, demands, vehicle_capacity
                )
            
            if vehicle_customers:
                # Create route for this vehicle
                vehicle_route = self._create_vehicle_route(depot_id, vehicle_customers)
                routes.append(vehicle_route)
                vehicle_assignments[vehicle_id] = vehicle_route
                
                # Remove assigned customers
                for customer in vehicle_customers:
                    if customer in unassigned_customers:
                        unassigned_customers.remove(customer)
        
        # Handle any remaining unassigned customers with default vehicle
        if unassigned_customers:
            default_capacity = max(v['capacity'] for v in available_vehicles.values())
            remaining_solution = self.solve_cvrp(depot_id, unassigned_customers, demands, default_capacity)
            routes.extend(remaining_solution.routes)
        
        solution = VRPSolution(
            routes=routes,
            vehicle_assignments=vehicle_assignments,
            method_used="heterogeneous_fleet_vrp"
        )
        
        return self._calculate_solution_metrics(solution, demands)
    
    def _select_customers_for_vehicle(self, depot_id: str, customers: List[str],
                                    demands: Dict[str, float], capacity: float) -> List[str]:
        """Select optimal set of customers for a vehicle using value density"""
        # Calculate value density (demand/distance ratio) for each customer
        customer_values = []
        
        for customer in customers:
            demand = demands.get(customer, 0)
            distance = self._get_distance(depot_id, customer)
            value_density = demand / (distance + 1) if distance > 0 else demand
            customer_values.append((customer, demand, value_density))
        
        # Sort by value density (highest first)
        customer_values.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy selection within capacity
        selected_customers = []
        current_load = 0.0
        
        for customer, demand, _ in customer_values:
            if current_load + demand <= capacity:
                selected_customers.append(customer)
                current_load += demand
        
        return selected_customers
    
    def _greedy_customer_selection(self, depot_id: str, customers: List[str],
                                 demands: Dict[str, float], capacity: float) -> List[str]:
        """Greedy customer selection for vehicle assignment"""
        selected = []
        current_load = 0.0
        remaining = customers.copy()
        current_location = depot_id
        
        while remaining:
            best_customer = None
            best_score = -1
            
            for customer in remaining:
                demand = demands.get(customer, 0)
                if current_load + demand > capacity:
                    continue
                
                distance = self._get_distance(current_location, customer)
                # Score combines proximity and demand efficiency
                score = demand / (distance + 1)
                
                if score > best_score:
                    best_score = score
                    best_customer = customer
            
            if best_customer is None:
                break
            
            selected.append(best_customer)
            current_load += demands.get(best_customer, 0)
            current_location = best_customer
            remaining.remove(best_customer)
        
        return selected
    
    def _create_vehicle_route(self, depot_id: str, customers: List[str]) -> List[str]:
        """Create optimized route for given customers"""
        if not customers:
            return [depot_id]
        
        # Use nearest neighbor for route ordering
        route = [depot_id]
        remaining = customers.copy()
        current_location = depot_id
        
        while remaining:
            nearest_customer = min(remaining, 
                                 key=lambda c: self._get_distance(current_location, c))
            route.append(nearest_customer)
            current_location = nearest_customer
            remaining.remove(nearest_customer)
        
        route.append(depot_id)
        return route
    
    def analyze_solution(self, solution: VRPSolution) -> Dict[str, Any]:
        """
        Comprehensive solution analysis
        
        Args:
            solution: VRP solution to analyze
            
        Returns:
            Detailed analysis results
        """
        analysis = {
            'basic_metrics': {
                'num_routes': len(solution.routes),
                'num_vehicles_used': solution.num_vehicles_used,
                'total_distance_km': solution.total_distance,
                'total_time_hours': solution.total_time / 60,
                'total_cost': solution.total_cost,
                'avg_route_distance': solution.total_distance / len(solution.routes) if solution.routes else 0,
                'capacity_utilization': solution.capacity_utilization,
                'solution_quality': solution.solution_quality
            },
            'route_analysis': [],
            'efficiency_metrics': {},
            'constraint_compliance': {
                'feasible': solution.feasible,
                'violations': solution.violations
            },
            'recommendations': []
        }
        
        # Analyze each route
        for i, route in enumerate(solution.routes):
            route_metrics = {
                'route_id': i,
                'stops': len(route) - 2,  # Exclude depot start/end
                'distance_km': self._calculate_route_distance(route),
                'time_minutes': self._calculate_route_time(route),
                'efficiency_ratio': 0.0,
                'locations': route
            }
            
            # Calculate route efficiency
            if len(route) > 2:
                direct_distance = self._get_distance(route[0], route[-1])
                route_metrics['efficiency_ratio'] = direct_distance / route_metrics['distance_km'] if route_metrics['distance_km'] > 0 else 0
            
            analysis['route_analysis'].append(route_metrics)
        
        # Calculate efficiency metrics
        if solution.routes:
            distances = [r['distance_km'] for r in analysis['route_analysis']]
            times = [r['time_minutes'] for r in analysis['route_analysis']]
            
            analysis['efficiency_metrics'] = {
                'distance_std_dev': np.std(distances),
                'time_std_dev': np.std(times),
                'route_balance_score': 1 - (np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0,
                'avg_efficiency_ratio': np.mean([r['efficiency_ratio'] for r in analysis['route_analysis']]),
                'vehicles_saved': max(0, len(solution.routes) - self._calculate_minimum_vehicles_needed(solution))
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_solution_recommendations(analysis)
        
        return analysis
    
    def _calculate_minimum_vehicles_needed(self, solution: VRPSolution) -> int:
        """Calculate theoretical minimum number of vehicles needed"""
        if not self.network.vehicles:
            return len(solution.routes)
        
        # Use largest vehicle capacity
        max_capacity = max(v.capacity for v in self.network.vehicles.values())
        
        # Calculate total demand from all routes
        total_demand = 0.0
        for route in solution.routes:
            for location_id in route:
                if (location_id in self.network.locations and 
                    self.network.locations[location_id].type != LocationType.DEPOT):
                    total_demand += self.network.locations[location_id].daily_demand
        
        return max(1, math.ceil(total_demand / max_capacity))
    
    def _generate_solution_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on solution analysis"""
        recommendations = []
        
        basic_metrics = analysis['basic_metrics']
        efficiency_metrics = analysis.get('efficiency_metrics', {})
        
        # Distance-based recommendations
        if basic_metrics['avg_route_distance'] > 200:
            recommendations.append("Consider adding more depots to reduce average route distance")
        
        # Efficiency recommendations
        avg_efficiency = efficiency_metrics.get('avg_efficiency_ratio', 0)
        if avg_efficiency < 0.6:
            recommendations.append("Route efficiency is low. Consider route optimization or clustering customers")
        
        # Balance recommendations
        balance_score = efficiency_metrics.get('route_balance_score', 1)
        if balance_score < 0.7:
            recommendations.append("Routes are unbalanced. Consider redistributing customers among vehicles")
        
        # Capacity utilization
        if basic_metrics['capacity_utilization'] < 0.7:
            recommendations.append("Vehicle capacity utilization is low. Consider using smaller vehicles or consolidating routes")
        elif basic_metrics['capacity_utilization'] > 0.95:
            recommendations.append("Vehicle capacity utilization is very high. Consider adding more vehicles to improve service")
        
        # Time-based recommendations
        if basic_metrics['total_time_hours'] > 8 * basic_metrics['num_routes']:
            recommendations.append("Some routes exceed 8-hour working time. Consider adding more vehicles or depots")
        
        # Cost optimization
        vehicles_saved = efficiency_metrics.get('vehicles_saved', 0)
        if vehicles_saved > 0:
            recommendations.append(f"Solution uses {vehicles_saved} fewer vehicles than basic assignment")
        
        # Constraint compliance
        if not analysis['constraint_compliance']['feasible']:
            recommendations.append("Solution violates constraints. Review capacity and time window requirements")
        
        return recommendations
    
    def export_solution(self, solution: VRPSolution, format: str = "json") -> Union[str, Dict]:
        """
        Export solution in various formats
        
        Args:
            solution: VRP solution to export
            format: Export format ("json", "csv", "excel")
            
        Returns:
            Exported solution data
        """
        if format == "json":
            return self._export_solution_json(solution)
        elif format == "csv":
            return self._export_solution_csv(solution)
        elif format == "excel":
            return self._export_solution_excel(solution)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_solution_json(self, solution: VRPSolution) -> str:
        """Export solution as JSON"""
        export_data = {
            'solution_summary': {
                'num_routes': len(solution.routes),
                'total_distance_km': solution.total_distance,
                'total_cost': solution.total_cost,
                'total_time_minutes': solution.total_time,
                'vehicles_used': solution.num_vehicles_used,
                'method_used': solution.method_used,
                'computation_time_seconds': solution.computation_time,
                'feasible': solution.feasible
            },
            'routes': [],
            'vehicle_assignments': solution.vehicle_assignments,
            'violations': solution.violations
        }
        
        # Add detailed route information
        for i, route in enumerate(solution.routes):
            route_info = {
                'route_id': i,
                'sequence': route,
                'distance_km': self._calculate_route_distance(route),
                'time_minutes': self._calculate_route_time(route),
                'stops': len(route) - 2
            }
            
            # Add location details
            route_info['location_details'] = []
            for location_id in route:
                if location_id in self.network.locations:
                    location = self.network.locations[location_id]
                    route_info['location_details'].append({
                        'location_id': location_id,
                        'name': location.name,
                        'type': location.type.value,
                        'coordinates': [location.latitude, location.longitude]
                    })
            
            export_data['routes'].append(route_info)
        
        return json.dumps(export_data, indent=2)
    
    def _export_solution_csv(self, solution: VRPSolution) -> str:
        """Export solution as CSV"""
        # Create route data for CSV
        csv_data = []
        
        for i, route in enumerate(solution.routes):
            for j, location_id in enumerate(route):
                csv_data.append({
                    'route_id': i,
                    'stop_sequence': j,
                    'location_id': location_id,
                    'location_name': self.network.locations[location_id].name if location_id in self.network.locations else location_id,
                    'location_type': self.network.locations[location_id].type.value if location_id in self.network.locations else 'unknown'
                })
        
        # Convert to CSV string
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    
    def _export_solution_excel(self, solution: VRPSolution) -> bytes:
        """Export solution as Excel file"""
        from io import BytesIO
        
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Routes', 'Total Distance (km)', 'Total Cost', 'Total Time (hours)', 'Vehicles Used', 'Feasible'],
                'Value': [len(solution.routes), solution.total_distance, solution.total_cost, 
                         solution.total_time/60, solution.num_vehicles_used, solution.feasible]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Routes sheet
            route_data = []
            for i, route in enumerate(solution.routes):
                for j, location_id in enumerate(route):
                    route_data.append({
                        'Route ID': i,
                        'Stop Sequence': j,
                        'Location ID': location_id,
                        'Location Name': self.network.locations[location_id].name if location_id in self.network.locations else location_id
                    })
            
            pd.DataFrame(route_data).to_excel(writer, sheet_name='Routes', index=False)
        
        buffer.seek(0)
        return buffer.getvalue()

# Example usage and testing
if __name__ == "__main__":
    # Create sample network for testing
    from models.distribution_model import create_jakarta_bbm_network
    
    network = create_jakarta_bbm_network()
    vrp_solver = VehicleRoutingProblem(network)
    
    print("VRP Solver Test Results:")
    print("=" * 60)
    
    # Test data
    depot_id = "DEPOT_01"
    customers = [loc_id for loc_id, loc in network.locations.items() 
                if loc.type == LocationType.SPBU]
    demands = {loc_id: loc.daily_demand for loc_id, loc in network.locations.items() 
              if loc.type == LocationType.SPBU}
    vehicle_capacity = 10000  # 10,000 liters
    
    print(f"Test setup: {len(customers)} customers, capacity: {vehicle_capacity:,}L")
    print(f"Total demand: {sum(demands.values()):,.0f}L")
    
    # Test different VRP methods
    methods_to_test = [
        SolutionMethod.NEAREST_NEIGHBOR,
        SolutionMethod.SAVINGS_ALGORITHM,
        SolutionMethod.SWEEP_ALGORITHM
    ]
    
    results = {}
    
    for method in methods_to_test:
        print(f"\n--- Testing {method.value} ---")
        start_time = time.time()
        
        try:
            solution = vrp_solver.solve_cvrp(depot_id, customers, demands, vehicle_capacity, method)
            
            print(f"✅ Solution found:")
            print(f"   Routes: {solution.num_vehicles_used}")
            print(f"   Total distance: {solution.total_distance:.1f} km")
            print(f"   Total cost: IDR {solution.total_cost:,.0f}")
            print(f"   Computation time: {solution.computation_time:.3f}s")
            print(f"   Feasible: {solution.feasible}")
            print(f"   Solution quality: {solution.solution_quality:.1f}")
            
            if solution.violations:
                print(f"   Violations: {len(solution.violations)}")
            
            results[method.value] = solution
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    # Compare methods
    if len(results) > 1:
        print(f"\n--- Method Comparison ---")
        comparison_data = []
        
        for method_name, solution in results.items():
            comparison_data.append({
                'Method': method_name,
                'Routes': solution.num_vehicles_used,
                'Distance (km)': f"{solution.total_distance:.1f}",
                'Cost (IDR)': f"{solution.total_cost:,.0f}",
                'Time (s)': f"{solution.computation_time:.3f}",
                'Quality': f"{solution.solution_quality:.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best solution
        best_method = max(results.keys(), key=lambda x: results[x].solution_quality)
        print(f"\n🏆 Best method: {best_method}")
    
    # Test solution analysis
    if results:
        best_solution = max(results.values(), key=lambda x: x.solution_quality)
        print(f"\n--- Solution Analysis ---")
        
        analysis = vrp_solver.analyze_solution(best_solution)
        
        print(f"Basic metrics:")
        for key, value in analysis['basic_metrics'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\nEfficiency metrics:")
        for key, value in analysis['efficiency_metrics'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # Test export functionality
    if results:
        print(f"\n--- Export Test ---")
        best_solution = max(results.values(), key=lambda x: x.solution_quality)
        
        try:
            json_export = vrp_solver.export_solution(best_solution, "json")
            json_data = json.loads(json_export)
            print(f"✅ JSON export: {len(json_data['routes'])} routes exported")
            
            csv_export = vrp_solver.export_solution(best_solution, "csv")
            print(f"✅ CSV export: {len(csv_export.split('\\n')) - 1} rows exported")
            
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
    
    # Test performance stats
    print(f"\n--- Performance Statistics ---")
    stats = vrp_solver.performance_stats
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n" + "=" * 60)
    print("VRP solver test completed! 🚛")