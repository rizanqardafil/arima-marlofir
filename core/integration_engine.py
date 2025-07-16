"""
BBM Dashboard - Integration Engine
Bridge between ARIMA forecasting and MARLOFIR-P Genetic Algorithm optimization
Converts ARIMA results into GA-compatible optimization parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import from our modules
try:
    from algorithms.genetic_algorithm import GeneticAlgorithm, GAParameters, Individual
    from models.distribution_model import DistributionNetwork, Location, Vehicle, LocationType
except ImportError:
    # For standalone testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAResult:
    """Container for ARIMA forecasting results"""
    
    def __init__(self, location_id: str, forecasts: pd.Series, 
                 confidence_intervals: Dict, model_metrics: Dict):
        self.location_id = location_id
        self.forecasts = forecasts
        self.confidence_intervals = confidence_intervals
        self.model_metrics = model_metrics
        self.forecast_period = len(forecasts)
        
    def get_daily_demand(self, day: int) -> float:
        """Get forecasted demand for specific day"""
        if 0 <= day < len(self.forecasts):
            return max(0, self.forecasts.iloc[day])
        return 0.0
    
    def get_weekly_demand(self) -> float:
        """Get total weekly demand"""
        return self.forecasts.sum()
    
    def get_confidence_bound(self, day: int, bound_type: str = 'upper') -> float:
        """Get confidence bound for specific day"""
        if bound_type in self.confidence_intervals and day < len(self.confidence_intervals[bound_type]):
            return max(0, self.confidence_intervals[bound_type][day])
        return self.get_daily_demand(day)

class IntegrationEngine:
    """
    Integration layer between ARIMA forecasting and Genetic Algorithm optimization
    Manages the complete data flow from forecasting to optimized distribution planning
    """
    
    def __init__(self, distribution_network: DistributionNetwork):
        self.network = distribution_network
        self.arima_results: Dict[str, ARIMAResult] = {}
        self.optimization_scenarios = []
        self.integration_config = {
            'forecast_horizon_days': 7,
            'safety_stock_factor': 1.2,
            'demand_uncertainty_buffer': 0.15,
            'priority_weight_factor': 2.0,
            'cost_optimization_weight': 0.6,
            'time_optimization_weight': 0.4
        }
        
        logger.info("Integration Engine initialized")
    
    def set_integration_config(self, config: Dict[str, Any]) -> None:
        """Update integration configuration"""
        self.integration_config.update(config)
        logger.info("Integration configuration updated")
    
    def add_arima_results(self, location_id: str, arima_result: ARIMAResult) -> None:
        """Add ARIMA forecasting results for a location"""
        self.arima_results[location_id] = arima_result
        logger.info(f"ARIMA results added for location: {location_id}")
    
    def process_arima_forecasts(self, arima_data: Dict[str, Any]) -> Dict[str, ARIMAResult]:
        """
        Process raw ARIMA data into structured results
        
        Args:
            arima_data: Dictionary containing ARIMA forecasting results
            
        Returns:
            Dictionary of processed ARIMA results
        """
        logger.info("Processing ARIMA forecasts for GA integration...")
        
        processed_results = {}
        
        for location_id, forecast_data in arima_data.items():
            try:
                # Extract forecast series
                if isinstance(forecast_data, dict):
                    forecasts = pd.Series(forecast_data.get('forecast', []))
                    confidence_intervals = forecast_data.get('confidence_intervals', {})
                    model_metrics = forecast_data.get('metrics', {})
                else:
                    # Assume it's already a pandas Series
                    forecasts = pd.Series(forecast_data)
                    confidence_intervals = {}
                    model_metrics = {}
                
                # Create ARIMA result object
                arima_result = ARIMAResult(
                    location_id=location_id,
                    forecasts=forecasts,
                    confidence_intervals=confidence_intervals,
                    model_metrics=model_metrics
                )
                
                processed_results[location_id] = arima_result
                self.add_arima_results(location_id, arima_result)
                
            except Exception as e:
                logger.error(f"Error processing ARIMA data for {location_id}: {str(e)}")
                continue
        
        logger.info(f"Processed ARIMA results for {len(processed_results)} locations")
        return processed_results
    
    def calculate_demand_requirements(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate demand requirements based on ARIMA forecasts
        
        Returns:
            Dictionary with demand requirements per location per day
        """
        demand_requirements = {}
        
        for location_id, arima_result in self.arima_results.items():
            if location_id not in self.network.locations:
                logger.warning(f"Location {location_id} not found in network")
                continue
            
            location = self.network.locations[location_id]
            requirements = {}
            
            # Calculate requirements for forecast horizon
            for day in range(self.integration_config['forecast_horizon_days']):
                # Base demand from ARIMA forecast
                base_demand = arima_result.get_daily_demand(day)
                
                # Add safety buffer for uncertainty
                uncertainty_buffer = base_demand * self.integration_config['demand_uncertainty_buffer']
                
                # Consider confidence intervals if available
                upper_bound = arima_result.get_confidence_bound(day, 'upper')
                if upper_bound > base_demand:
                    demand_with_uncertainty = min(upper_bound, base_demand + uncertainty_buffer)
                else:
                    demand_with_uncertainty = base_demand + uncertainty_buffer
                
                # Apply safety stock factor
                final_demand = demand_with_uncertainty * self.integration_config['safety_stock_factor']
                
                # Ensure demand doesn't exceed location capacity
                max_deliverable = location.capacity - location.current_stock
                final_demand = min(final_demand, max_deliverable)
                
                requirements[f'day_{day}'] = max(0, final_demand)
            
            demand_requirements[location_id] = requirements
        
        return demand_requirements
    
    def generate_optimization_constraints(self) -> Dict[str, Any]:
        """
        Generate constraints for GA optimization based on ARIMA forecasts and network
        
        Returns:
            Dictionary containing GA optimization constraints
        """
        logger.info("Generating optimization constraints...")
        
        # Get demand requirements
        demand_requirements = self.calculate_demand_requirements()
        
        # Location constraints
        location_constraints = {}
        for location_id, location in self.network.locations.items():
            if location.type == LocationType.SPBU:
                # Calculate total weekly demand
                weekly_demand = 0
                if location_id in demand_requirements:
                    weekly_demand = sum(demand_requirements[location_id].values())
                
                # Calculate urgency score
                urgency = self.network.calculate_demand_urgency(location_id)
                
                # Priority weight based on urgency and location priority
                priority_weight = (urgency * self.integration_config['priority_weight_factor'] + 
                                 location.priority) / 2
                
                location_constraints[location_id] = {
                    'weekly_demand': weekly_demand,
                    'current_stock': location.current_stock,
                    'capacity': location.capacity,
                    'urgency_score': urgency,
                    'priority_weight': priority_weight,
                    'min_delivery': max(0, location.min_stock_level - location.current_stock),
                    'max_delivery': location.capacity - location.current_stock,
                    'service_time': location.service_time
                }
        
        # Vehicle constraints
        vehicle_constraints = {}
        for vehicle_id, vehicle in self.network.vehicles.items():
            vehicle_constraints[vehicle_id] = {
                'capacity': vehicle.capacity,
                'operating_cost_per_km': vehicle.total_operating_cost_per_km,
                'max_working_hours': vehicle.max_working_hours,
                'current_location': vehicle.current_location,
                'speed': vehicle.speed
            }
        
        # Network constraints
        network_constraints = {
            'distance_matrix': dict(self.network.distance_matrix),
            'cost_matrix': dict(self.network.cost_matrix),
            'time_matrix': dict(self.network.time_matrix)
        }
        
        # Optimization objectives
        objectives = {
            'minimize_total_cost': self.integration_config['cost_optimization_weight'],
            'minimize_total_time': self.integration_config['time_optimization_weight'],
            'maximize_service_level': 1.0 - max(self.integration_config['cost_optimization_weight'],
                                               self.integration_config['time_optimization_weight'])
        }
        
        return {
            'locations': location_constraints,
            'vehicles': vehicle_constraints,
            'network': network_constraints,
            'objectives': objectives,
            'forecast_horizon': self.integration_config['forecast_horizon_days'],
            'demand_requirements': demand_requirements
        }
    
    def create_ga_parameters(self, complexity_factor: float = 1.0) -> GAParameters:
        """
        Create GA parameters based on problem complexity
        
        Args:
            complexity_factor: Factor to adjust GA parameters based on problem size
            
        Returns:
            Configured GA parameters
        """
        # Base parameters
        base_population = 100
        base_generations = 100
        
        # Adjust based on network size
        num_locations = len([loc for loc in self.network.locations.values() 
                           if loc.type == LocationType.SPBU])
        
        # Scale parameters with problem complexity
        population_size = int(base_population * complexity_factor * min(2.0, num_locations / 10))
        generations = int(base_generations * complexity_factor)
        
        return GAParameters(
            population_size=max(50, min(200, population_size)),
            generations=max(50, min(300, generations)),
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_rate=0.15,
            tournament_size=max(3, int(population_size * 0.05)),
            adaptive_mutation=True,
            local_search=True
        )
    
    def setup_ga_optimization(self, complexity_factor: float = 1.0) -> Tuple[GeneticAlgorithm, Dict]:
        """
        Setup GA optimization with ARIMA-derived constraints
        
        Args:
            complexity_factor: Complexity adjustment factor
            
        Returns:
            Tuple of (configured GA instance, constraints dictionary)
        """
        logger.info("Setting up GA optimization...")
        
        # Generate constraints
        constraints = self.generate_optimization_constraints()
        
        # Create GA parameters
        ga_params = self.create_ga_parameters(complexity_factor)
        
        # Initialize GA
        ga = GeneticAlgorithm(ga_params)
        
        # Setup problem constraints for GA
        locations = list(self.network.locations.keys())
        
        # Convert constraints to GA format
        distances = {}
        demands = {}
        
        for (from_loc, to_loc), distance in self.network.distance_matrix.items():
            # Convert location IDs to indices
            try:
                from_idx = locations.index(from_loc)
                to_idx = locations.index(to_loc)
                distances[(from_idx, to_idx)] = distance
            except ValueError:
                continue
        
        for location_id, location_data in constraints['locations'].items():
            if location_id in locations:
                demands[location_id] = location_data['weekly_demand']
        
        # Get largest vehicle capacity as default
        max_capacity = max([v.capacity for v in self.network.vehicles.values()], default=10000)
        
        # Set GA constraints
        ga.set_problem_constraints(
            locations=locations,
            distances=distances,
            demands=demands,
            vehicle_capacity=max_capacity,
            fuel_cost=self.network.fuel_price_per_liter / 1000  # Convert to manageable scale
        )
        
        logger.info("GA optimization setup completed")
        return ga, constraints
    
    def run_integrated_optimization(self, complexity_factor: float = 1.0) -> Dict[str, Any]:
        """
        Run complete integrated optimization (ARIMA → GA)
        
        Args:
            complexity_factor: Problem complexity adjustment
            
        Returns:
            Complete optimization results
        """
        logger.info("Starting integrated ARIMA-GA optimization...")
        
        # Validate inputs
        if not self.arima_results:
            raise ValueError("No ARIMA results available. Add ARIMA forecasts first.")
        
        if not self.network.locations:
            raise ValueError("Distribution network is empty. Add locations first.")
        
        # Setup and run GA optimization
        ga, constraints = self.setup_ga_optimization(complexity_factor)
        ga_results = ga.optimize()
        
        # Post-process results
        optimized_results = self.post_process_results(ga_results, constraints)
        
        logger.info("Integrated optimization completed successfully")
        return optimized_results
    
    def post_process_results(self, ga_results: Dict, constraints: Dict) -> Dict[str, Any]:
        """
        Post-process GA results with business intelligence
        
        Args:
            ga_results: Raw GA optimization results
            constraints: Original constraints used
            
        Returns:
            Enhanced results with business insights
        """
        logger.info("Post-processing optimization results...")
        
        # Extract best solution
        best_solution = ga_results['best_solution']
        
        # Convert route indices back to location names
        locations = list(self.network.locations.keys())
        route_names = []
        for idx in best_solution['route']:
            if 0 <= idx < len(locations):
                route_names.append(locations[idx])
            else:
                route_names.append(f"unknown_{idx}")
        
        # Calculate detailed metrics
        total_demand_served = 0
        service_level_achieved = 0
        location_details = []
        
        for location_id, location_data in constraints['locations'].items():
            demand = location_data['weekly_demand']
            total_demand_served += demand
            
            # Service level calculation (simplified)
            if demand > 0:
                service_level = min(1.0, demand / max(1, location_data['capacity']))
                service_level_achieved += service_level
            
            location_details.append({
                'location_id': location_id,
                'location_name': self.network.locations[location_id].name,
                'forecasted_demand': demand,
                'current_stock': location_data['current_stock'],
                'urgency_score': location_data['urgency_score'],
                'priority_weight': location_data['priority_weight']
            })
        
        # Calculate overall performance metrics
        avg_service_level = service_level_achieved / len(constraints['locations']) if constraints['locations'] else 0
        
        # Route analysis
        route_analysis = {
            'total_locations': len(route_names),
            'total_distance': best_solution['distance'],
            'total_cost': best_solution['cost'],
            'fuel_efficiency': best_solution['fuel_efficiency'],
            'route_sequence': route_names
        }
        
        # Convergence analysis
        convergence_data = []
        for gen_data in ga_results['convergence_history']:
            convergence_data.append({
                'generation': gen_data['generation'],
                'best_fitness': gen_data['best_fitness'],
                'avg_fitness': gen_data['avg_fitness'],
                'diversity': gen_data['diversity']
            })
        
        return {
            'optimization_summary': {
                'total_demand_served': total_demand_served,
                'service_level_achieved': avg_service_level,
                'optimization_fitness': best_solution['fitness'],
                'forecast_horizon_days': constraints['forecast_horizon']
            },
            'route_optimization': route_analysis,
            'location_analysis': location_details,
            'convergence_history': convergence_data,
            'arima_integration': {
                'locations_forecasted': len(self.arima_results),
                'forecast_accuracy': self.calculate_forecast_metrics(),
                'demand_scenarios': list(constraints['demand_requirements'].keys())
            },
            'recommendations': self.generate_recommendations(best_solution, constraints),
            'raw_ga_results': ga_results
        }
    
    def calculate_forecast_metrics(self) -> Dict[str, float]:
        """Calculate ARIMA forecast quality metrics"""
        if not self.arima_results:
            return {}
        
        metrics = {
            'avg_forecast_horizon': 0,
            'locations_with_confidence_intervals': 0,
            'total_forecasted_demand': 0
        }
        
        total_horizon = 0
        ci_count = 0
        total_demand = 0
        
        for location_id, arima_result in self.arima_results.items():
            total_horizon += arima_result.forecast_period
            if arima_result.confidence_intervals:
                ci_count += 1
            total_demand += arima_result.get_weekly_demand()
        
        num_locations = len(self.arima_results)
        metrics['avg_forecast_horizon'] = total_horizon / num_locations if num_locations > 0 else 0
        metrics['locations_with_confidence_intervals'] = ci_count
        metrics['total_forecasted_demand'] = total_demand
        
        return metrics
    
    def generate_recommendations(self, best_solution: Dict, constraints: Dict) -> List[str]:
        """Generate business recommendations based on optimization results"""
        recommendations = []
        
        # Cost analysis
        if best_solution['cost'] > 1000000:  # High cost threshold
            recommendations.append("Consider optimizing vehicle routes to reduce transportation costs")
        
        # Service level analysis
        urgent_locations = [loc for loc in constraints['locations'].values() 
                          if loc['urgency_score'] > 0.7]
        if urgent_locations:
            recommendations.append(f"Prioritize delivery to {len(urgent_locations)} high-urgency locations")
        
        # Capacity utilization
        underutilized_locations = [loc for loc in constraints['locations'].values()
                                 if loc['current_stock'] / loc['capacity'] < 0.3]
        if underutilized_locations:
            recommendations.append(f"Consider inventory optimization for {len(underutilized_locations)} underutilized locations")
        
        # Efficiency recommendations
        if best_solution['fuel_efficiency'] < 5:
            recommendations.append("Implement fuel efficiency improvements in vehicle operations")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Create sample network
    from models.distribution_model import create_jakarta_bbm_network
    
    network = create_jakarta_bbm_network()
    integration_engine = IntegrationEngine(network)
    
    # Create sample ARIMA data
    sample_arima_data = {}
    spbu_locations = [loc_id for loc_id, loc in network.locations.items() 
                     if loc.type == LocationType.SPBU]
    
    for location_id in spbu_locations:
        # Generate sample forecast data
        np.random.seed(42)
        forecast_values = np.random.normal(5000, 1000, 7)  # 7 days forecast
        forecast_values = np.maximum(forecast_values, 0)  # Ensure positive
        
        sample_arima_data[location_id] = {
            'forecast': forecast_values.tolist(),
            'confidence_intervals': {
                'upper': (forecast_values * 1.2).tolist(),
                'lower': (forecast_values * 0.8).tolist()
            },
            'metrics': {
                'mape': 5.2,
                'rmse': 234.5
            }
        }
    
    # Process ARIMA results
    integration_engine.process_arima_forecasts(sample_arima_data)
    
    # Run integrated optimization
    try:
        results = integration_engine.run_integrated_optimization()
        print("Integration Engine Test Results:")
        print(f"Total demand served: {results['optimization_summary']['total_demand_served']:.2f}")
        print(f"Service level: {results['optimization_summary']['service_level_achieved']:.2%}")
        print(f"Route cost: {results['route_optimization']['total_cost']:.2f}")
        print(f"Recommendations: {len(results['recommendations'])}")
        
    except Exception as e:
        print(f"Error during integration test: {str(e)}")