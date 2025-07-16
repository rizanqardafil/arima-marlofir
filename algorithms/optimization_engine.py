"""
MARLOFIR-P Multi-Objective Optimization Engine
High-level orchestration for BBM distribution optimization
Combines ARIMA forecasting, GA optimization, and business intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import time
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import from our modules
try:
    from algorithms.genetic_algorithm import GeneticAlgorithm, GAParameters, Individual
    from models.distribution_model import DistributionNetwork, Location, Vehicle, LocationType
    from core.integration_engine import IntegrationEngine, ARIMAResult
except ImportError:
    # For standalone testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives for multi-objective optimization"""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_DISTANCE = "minimize_distance"
    MAXIMIZE_SERVICE_LEVEL = "maximize_service_level"
    MAXIMIZE_FUEL_EFFICIENCY = "maximize_fuel_efficiency"
    MINIMIZE_CARBON_FOOTPRINT = "minimize_carbon_footprint"

class OptimizationStrategy(Enum):
    """Different optimization strategies"""
    COST_FOCUSED = "cost_focused"           # Prioritize cost minimization
    TIME_FOCUSED = "time_focused"           # Prioritize delivery time
    BALANCED = "balanced"                   # Balance all objectives
    SERVICE_FOCUSED = "service_focused"     # Prioritize service level
    ECO_FRIENDLY = "eco_friendly"          # Prioritize environmental impact

@dataclass
class OptimizationObjectives:
    """Multi-objective optimization configuration"""
    cost_weight: float = 0.3
    time_weight: float = 0.2
    distance_weight: float = 0.15
    service_level_weight: float = 0.25
    fuel_efficiency_weight: float = 0.1
    carbon_footprint_weight: float = 0.0
    
    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total = sum([self.cost_weight, self.time_weight, self.distance_weight,
                    self.service_level_weight, self.fuel_efficiency_weight,
                    self.carbon_footprint_weight])
        if total > 0:
            self.cost_weight /= total
            self.time_weight /= total
            self.distance_weight /= total
            self.service_level_weight /= total
            self.fuel_efficiency_weight /= total
            self.carbon_footprint_weight /= total

@dataclass
class OptimizationScenario:
    """Single optimization scenario configuration"""
    scenario_id: str
    name: str
    description: str
    objectives: OptimizationObjectives
    ga_parameters: GAParameters
    constraints: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

class OptimizationEngine:
    """
    High-level optimization engine for MARLOFIR-P
    Orchestrates multi-objective optimization with different strategies
    """
    
    def __init__(self, distribution_network: DistributionNetwork):
        self.network = distribution_network
        self.integration_engine = IntegrationEngine(distribution_network)
        self.optimization_history = []
        self.scenarios = {}
        self.current_results = {}
        
        # Default configuration
        self.config = {
            'max_optimization_time_minutes': 30,
            'parallel_scenarios': False,
            'save_intermediate_results': True,
            'auto_parameter_tuning': True,
            'convergence_threshold': 0.001,
            'max_stagnation_generations': 20
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_optimizations': 0,
            'avg_optimization_time': 0,
            'best_fitness_achieved': 0,
            'scenarios_executed': 0
        }
        
        logger.info("Optimization Engine initialized")
    
    def set_configuration(self, config: Dict[str, Any]) -> None:
        """Update optimization engine configuration"""
        self.config.update(config)
        logger.info("Optimization configuration updated")
    
    def create_predefined_scenarios(self) -> Dict[str, OptimizationScenario]:
        """Create predefined optimization scenarios"""
        scenarios = {}
        
        # Cost-focused scenario
        cost_objectives = OptimizationObjectives(
            cost_weight=0.5, time_weight=0.2, service_level_weight=0.2, 
            fuel_efficiency_weight=0.1
        )
        scenarios['cost_focused'] = OptimizationScenario(
            scenario_id='cost_focused',
            name='Cost Optimization',
            description='Minimize total distribution costs',
            objectives=cost_objectives,
            ga_parameters=GAParameters(population_size=100, generations=150, mutation_rate=0.1)
        )
        
        # Time-focused scenario
        time_objectives = OptimizationObjectives(
            time_weight=0.5, cost_weight=0.2, service_level_weight=0.2,
            fuel_efficiency_weight=0.1
        )
        scenarios['time_focused'] = OptimizationScenario(
            scenario_id='time_focused',
            name='Time Optimization',
            description='Minimize delivery time and maximize speed',
            objectives=time_objectives,
            ga_parameters=GAParameters(population_size=80, generations=100, mutation_rate=0.15)
        )
        
        # Balanced scenario
        balanced_objectives = OptimizationObjectives(
            cost_weight=0.25, time_weight=0.25, service_level_weight=0.25,
            fuel_efficiency_weight=0.15, distance_weight=0.1
        )
        scenarios['balanced'] = OptimizationScenario(
            scenario_id='balanced',
            name='Balanced Optimization',
            description='Balance all optimization objectives',
            objectives=balanced_objectives,
            ga_parameters=GAParameters(population_size=120, generations=200, mutation_rate=0.12)
        )
        
        # Service-focused scenario
        service_objectives = OptimizationObjectives(
            service_level_weight=0.4, cost_weight=0.2, time_weight=0.2,
            fuel_efficiency_weight=0.2
        )
        scenarios['service_focused'] = OptimizationScenario(
            scenario_id='service_focused',
            name='Service Level Optimization',
            description='Maximize customer service level',
            objectives=service_objectives,
            ga_parameters=GAParameters(population_size=100, generations=180, mutation_rate=0.08)
        )
        
        # Eco-friendly scenario
        eco_objectives = OptimizationObjectives(
            fuel_efficiency_weight=0.35, carbon_footprint_weight=0.25,
            distance_weight=0.2, cost_weight=0.2
        )
        scenarios['eco_friendly'] = OptimizationScenario(
            scenario_id='eco_friendly',
            name='Environmental Optimization',
            description='Minimize environmental impact',
            objectives=eco_objectives,
            ga_parameters=GAParameters(population_size=90, generations=160, mutation_rate=0.1)
        )
        
        return scenarios
    
    def add_scenario(self, scenario: OptimizationScenario) -> None:
        """Add optimization scenario"""
        self.scenarios[scenario.scenario_id] = scenario
        logger.info(f"Added optimization scenario: {scenario.name}")
    
    def load_predefined_scenarios(self) -> None:
        """Load all predefined scenarios"""
        predefined = self.create_predefined_scenarios()
        self.scenarios.update(predefined)
        logger.info(f"Loaded {len(predefined)} predefined scenarios")
    
    def calculate_multi_objective_fitness(self, individual: Individual, 
                                        objectives: OptimizationObjectives,
                                        constraints: Dict[str, Any]) -> float:
        """
        Calculate multi-objective fitness score
        
        Args:
            individual: GA individual to evaluate
            objectives: Objective weights
            constraints: Problem constraints
            
        Returns:
            Combined fitness score
        """
        # Base metrics from individual
        cost_score = 1000 / (individual.cost + 1) if individual.cost > 0 else 0
        distance_score = 1000 / (individual.distance + 1) if individual.distance > 0 else 0
        efficiency_score = individual.fuel_efficiency
        
        # Calculate service level score
        service_score = self.calculate_service_level_score(individual, constraints)
        
        # Calculate time score (inverse of travel time)
        time_score = self.calculate_time_score(individual, constraints)
        
        # Calculate carbon footprint score
        carbon_score = self.calculate_carbon_score(individual)
        
        # Combine objectives with weights
        combined_fitness = (
            objectives.cost_weight * cost_score +
            objectives.time_weight * time_score +
            objectives.distance_weight * distance_score +
            objectives.service_level_weight * service_score +
            objectives.fuel_efficiency_weight * efficiency_score +
            objectives.carbon_footprint_weight * carbon_score
        )
        
        return max(combined_fitness, 0.1)
    
    def calculate_service_level_score(self, individual: Individual, 
                                    constraints: Dict[str, Any]) -> float:
        """Calculate service level score for individual"""
        if 'locations' not in constraints:
            return 0.0
        
        total_score = 0
        total_locations = 0
        
        # Check each location in route
        for loc_idx in individual.chromosome:
            if loc_idx < len(self.network.locations):
                location_id = list(self.network.locations.keys())[loc_idx]
                if location_id in constraints['locations']:
                    loc_data = constraints['locations'][location_id]
                    # Service score based on urgency and priority
                    urgency = loc_data.get('urgency_score', 0)
                    priority = loc_data.get('priority_weight', 1)
                    score = (urgency + priority) / 2
                    total_score += score
                    total_locations += 1
        
        return total_score / total_locations if total_locations > 0 else 0
    
    def calculate_time_score(self, individual: Individual, 
                           constraints: Dict[str, Any]) -> float:
        """Calculate time efficiency score"""
        if 'network' not in constraints or 'time_matrix' not in constraints['network']:
            return 0.0
        
        total_time = 0
        time_matrix = constraints['network']['time_matrix']
        chromosome = individual.chromosome
        
        for i in range(len(chromosome) - 1):
            from_idx = chromosome[i]
            to_idx = chromosome[i + 1]
            
            # Convert indices to location IDs
            locations = list(self.network.locations.keys())
            if from_idx < len(locations) and to_idx < len(locations):
                from_loc = locations[from_idx]
                to_loc = locations[to_idx]
                
                if (from_loc, to_loc) in time_matrix:
                    total_time += time_matrix[(from_loc, to_loc)]
        
        # Return inverse time score (lower time = higher score)
        return 1000 / (total_time + 1) if total_time > 0 else 0
    
    def calculate_carbon_score(self, individual: Individual) -> float:
        """Calculate carbon footprint score"""
        # Simplified carbon calculation based on distance and fuel efficiency
        if individual.distance > 0 and individual.fuel_efficiency > 0:
            # Carbon emission factor (kg CO2 per liter of fuel)
            carbon_factor = 2.3  # kg CO2 per liter diesel
            fuel_consumption = individual.distance / individual.fuel_efficiency
            carbon_emissions = fuel_consumption * carbon_factor
            
            # Return inverse carbon score (lower emissions = higher score)
            return 1000 / (carbon_emissions + 1)
        
        return 0.0
    
    def auto_tune_parameters(self, scenario: OptimizationScenario, 
                           problem_complexity: float) -> GAParameters:
        """
        Auto-tune GA parameters based on problem complexity and scenario
        
        Args:
            scenario: Optimization scenario
            problem_complexity: Problem complexity factor
            
        Returns:
            Tuned GA parameters
        """
        if not self.config['auto_parameter_tuning']:
            return scenario.ga_parameters
        
        base_params = scenario.ga_parameters
        
        # Adjust based on complexity
        complexity_factor = min(2.0, max(0.5, problem_complexity))
        
        tuned_params = GAParameters(
            population_size=int(base_params.population_size * complexity_factor),
            generations=int(base_params.generations * complexity_factor),
            mutation_rate=base_params.mutation_rate,
            crossover_rate=base_params.crossover_rate,
            elitism_rate=base_params.elitism_rate,
            tournament_size=max(3, int(base_params.tournament_size * complexity_factor)),
            adaptive_mutation=True,
            local_search=True
        )
        
        # Ensure reasonable bounds
        tuned_params.population_size = max(50, min(300, tuned_params.population_size))
        tuned_params.generations = max(50, min(500, tuned_params.generations))
        
        return tuned_params
    
    def optimize_scenario(self, scenario_id: str, arima_data: Dict[str, Any],
                         complexity_factor: float = 1.0) -> Dict[str, Any]:
        """
        Optimize single scenario
        
        Args:
            scenario_id: ID of scenario to optimize
            arima_data: ARIMA forecasting data
            complexity_factor: Problem complexity adjustment
            
        Returns:
            Optimization results for the scenario
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        scenario = self.scenarios[scenario_id]
        if not scenario.enabled:
            logger.warning(f"Scenario {scenario_id} is disabled")
            return {}
        
        logger.info(f"Optimizing scenario: {scenario.name}")
        start_time = time.time()
        
        # Process ARIMA data
        self.integration_engine.process_arima_forecasts(arima_data)
        
        # Generate constraints
        constraints = self.integration_engine.generate_optimization_constraints()
        
        # Auto-tune parameters
        tuned_params = self.auto_tune_parameters(scenario, complexity_factor)
        
        # Create custom GA with multi-objective fitness
        ga = GeneticAlgorithm(tuned_params)
        
        # Setup GA constraints
        locations = list(self.network.locations.keys())
        distances = dict(self.network.distance_matrix)
        demands = {loc_id: data['weekly_demand'] 
                  for loc_id, data in constraints['locations'].items()}
        
        max_capacity = max([v.capacity for v in self.network.vehicles.values()], default=10000)
        
        ga.set_problem_constraints(
            locations=locations,
            distances=distances,
            demands=demands,
            vehicle_capacity=max_capacity,
            fuel_cost=self.network.fuel_price_per_liter / 1000
        )
        
        # Override fitness function with multi-objective
        original_evaluate = ga.evaluate_fitness
        
        def multi_objective_fitness(individual):
            # Get base fitness
            base_fitness = original_evaluate(individual)
            individual.fitness = base_fitness
            
            # Calculate multi-objective fitness
            mo_fitness = self.calculate_multi_objective_fitness(
                individual, scenario.objectives, constraints
            )
            
            return mo_fitness
        
        ga.evaluate_fitness = multi_objective_fitness
        
        # Run optimization with timeout
        max_time = self.config['max_optimization_time_minutes'] * 60
        ga_start = time.time()
        
        try:
            ga_results = ga.optimize()
            
            # Check for timeout
            elapsed = time.time() - ga_start
            if elapsed > max_time:
                logger.warning(f"Optimization timeout after {elapsed:.1f}s")
        
        except Exception as e:
            logger.error(f"Optimization error for scenario {scenario_id}: {str(e)}")
            return {'error': str(e)}
        
        # Post-process results
        optimization_time = time.time() - start_time
        enhanced_results = self.integration_engine.post_process_results(ga_results, constraints)
        
        # Add scenario-specific information
        enhanced_results['scenario_info'] = {
            'scenario_id': scenario_id,
            'scenario_name': scenario.name,
            'objectives': {
                'cost_weight': scenario.objectives.cost_weight,
                'time_weight': scenario.objectives.time_weight,
                'service_weight': scenario.objectives.service_level_weight,
                'efficiency_weight': scenario.objectives.fuel_efficiency_weight
            },
            'optimization_time_seconds': optimization_time,
            'parameters_used': {
                'population_size': tuned_params.population_size,
                'generations': tuned_params.generations,
                'mutation_rate': tuned_params.mutation_rate
            }
        }
        
        # Update performance metrics
        self.performance_metrics['total_optimizations'] += 1
        self.performance_metrics['avg_optimization_time'] = (
            (self.performance_metrics['avg_optimization_time'] * 
             (self.performance_metrics['total_optimizations'] - 1) + optimization_time) /
            self.performance_metrics['total_optimizations']
        )
        
        if enhanced_results['optimization_summary']['optimization_fitness'] > self.performance_metrics['best_fitness_achieved']:
            self.performance_metrics['best_fitness_achieved'] = enhanced_results['optimization_summary']['optimization_fitness']
        
        logger.info(f"Scenario {scenario_id} completed in {optimization_time:.2f}s")
        return enhanced_results
    
    def optimize_multiple_scenarios(self, scenario_ids: List[str], 
                                  arima_data: Dict[str, Any],
                                  complexity_factor: float = 1.0) -> Dict[str, Any]:
        """
        Optimize multiple scenarios and compare results
        
        Args:
            scenario_ids: List of scenario IDs to optimize
            arima_data: ARIMA forecasting data
            complexity_factor: Problem complexity factor
            
        Returns:
            Combined results from all scenarios
        """
        logger.info(f"Optimizing {len(scenario_ids)} scenarios")
        
        scenario_results = {}
        comparison_metrics = {}
        
        for scenario_id in scenario_ids:
            try:
                result = self.optimize_scenario(scenario_id, arima_data, complexity_factor)
                scenario_results[scenario_id] = result
                
                # Extract key metrics for comparison
                if 'optimization_summary' in result:
                    comparison_metrics[scenario_id] = {
                        'fitness': result['optimization_summary']['optimization_fitness'],
                        'cost': result['route_optimization']['total_cost'],
                        'distance': result['route_optimization']['total_distance'],
                        'service_level': result['optimization_summary']['service_level_achieved'],
                        'optimization_time': result['scenario_info']['optimization_time_seconds']
                    }
            
            except Exception as e:
                logger.error(f"Failed to optimize scenario {scenario_id}: {str(e)}")
                scenario_results[scenario_id] = {'error': str(e)}
        
        # Find best scenario for each metric
        best_scenarios = {}
        if comparison_metrics:
            best_scenarios = {
                'best_fitness': max(comparison_metrics.keys(), 
                                  key=lambda x: comparison_metrics[x]['fitness']),
                'lowest_cost': min(comparison_metrics.keys(), 
                                 key=lambda x: comparison_metrics[x]['cost']),
                'shortest_distance': min(comparison_metrics.keys(), 
                                       key=lambda x: comparison_metrics[x]['distance']),
                'best_service': max(comparison_metrics.keys(), 
                                  key=lambda x: comparison_metrics[x]['service_level']),
                'fastest_optimization': min(comparison_metrics.keys(), 
                                          key=lambda x: comparison_metrics[x]['optimization_time'])
            }
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report(scenario_results, comparison_metrics)
        
        return {
            'scenario_results': scenario_results,
            'comparison_metrics': comparison_metrics,
            'best_scenarios': best_scenarios,
            'comparison_report': comparison_report,
            'execution_summary': {
                'total_scenarios': len(scenario_ids),
                'successful_optimizations': len(comparison_metrics),
                'failed_optimizations': len(scenario_ids) - len(comparison_metrics)
            }
        }
    
    def generate_comparison_report(self, scenario_results: Dict, 
                                 comparison_metrics: Dict) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        if not comparison_metrics:
            return {}
        
        # Calculate statistics
        metrics_df = pd.DataFrame(comparison_metrics).T
        
        report = {
            'summary_statistics': {
                'fitness': {
                    'mean': metrics_df['fitness'].mean(),
                    'std': metrics_df['fitness'].std(),
                    'min': metrics_df['fitness'].min(),
                    'max': metrics_df['fitness'].max()
                },
                'cost': {
                    'mean': metrics_df['cost'].mean(),
                    'std': metrics_df['cost'].std(),
                    'min': metrics_df['cost'].min(),
                    'max': metrics_df['cost'].max()
                },
                'service_level': {
                    'mean': metrics_df['service_level'].mean(),
                    'std': metrics_df['service_level'].std(),
                    'min': metrics_df['service_level'].min(),
                    'max': metrics_df['service_level'].max()
                }
            },
            'recommendations': self.generate_scenario_recommendations(comparison_metrics),
            'pareto_analysis': self.calculate_pareto_efficiency(comparison_metrics)
        }
        
        return report
    
    def generate_scenario_recommendations(self, comparison_metrics: Dict) -> List[str]:
        """Generate recommendations based on scenario comparison"""
        recommendations = []
        
        if not comparison_metrics:
            return recommendations
        
        # Find best performing scenarios
        best_fitness = max(comparison_metrics.keys(), 
                          key=lambda x: comparison_metrics[x]['fitness'])
        best_cost = min(comparison_metrics.keys(), 
                       key=lambda x: comparison_metrics[x]['cost'])
        best_service = max(comparison_metrics.keys(), 
                          key=lambda x: comparison_metrics[x]['service_level'])
        
        # Generate recommendations
        if best_fitness in self.scenarios:
            recommendations.append(f"For overall performance, use '{self.scenarios[best_fitness].name}' scenario")
        
        if best_cost in self.scenarios:
            recommendations.append(f"For cost efficiency, use '{self.scenarios[best_cost].name}' scenario")
        
        if best_service in self.scenarios:
            recommendations.append(f"For service quality, use '{self.scenarios[best_service].name}' scenario")
        
        # Performance analysis
        cost_range = max(comparison_metrics[x]['cost'] for x in comparison_metrics) - \
                    min(comparison_metrics[x]['cost'] for x in comparison_metrics)
        
        if cost_range > 100000:  # Significant cost difference
            recommendations.append("Consider cost-focused optimization due to significant cost variations")
        
        return recommendations
    
    def calculate_pareto_efficiency(self, comparison_metrics: Dict) -> Dict[str, Any]:
        """Calculate Pareto efficiency analysis"""
        if len(comparison_metrics) < 2:
            return {}
        
        pareto_efficient = []
        scenarios = list(comparison_metrics.keys())
        
        for i, scenario in enumerate(scenarios):
            is_pareto = True
            current_metrics = comparison_metrics[scenario]
            
            for j, other_scenario in enumerate(scenarios):
                if i != j:
                    other_metrics = comparison_metrics[other_scenario]
                    
                    # Check if other dominates current
                    # (better or equal in all objectives, better in at least one)
                    if (other_metrics['fitness'] >= current_metrics['fitness'] and
                        other_metrics['service_level'] >= current_metrics['service_level'] and
                        other_metrics['cost'] <= current_metrics['cost'] and
                        (other_metrics['fitness'] > current_metrics['fitness'] or
                         other_metrics['service_level'] > current_metrics['service_level'] or
                         other_metrics['cost'] < current_metrics['cost'])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_efficient.append(scenario)
        
        return {
            'pareto_efficient_scenarios': pareto_efficient,
            'pareto_percentage': len(pareto_efficient) / len(scenarios) * 100,
            'dominated_scenarios': [s for s in scenarios if s not in pareto_efficient]
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get overall optimization engine summary"""
        return {
            'performance_metrics': self.performance_metrics,
            'available_scenarios': {sid: scenario.name for sid, scenario in self.scenarios.items()},
            'network_summary': self.network.get_network_summary(),
            'configuration': self.config
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample network and optimization engine
    from models.distribution_model import create_jakarta_bbm_network
    
    network = create_jakarta_bbm_network()
    optimization_engine = OptimizationEngine(network)
    
    # Load predefined scenarios
    optimization_engine.load_predefined_scenarios()
    
    # Create sample ARIMA data
    sample_arima_data = {}
    spbu_locations = [loc_id for loc_id, loc in network.locations.items() 
                     if loc.type == LocationType.SPBU]
    
    for location_id in spbu_locations:
        np.random.seed(42)
        forecast_values = np.random.normal(5000, 1000, 7)
        forecast_values = np.maximum(forecast_values, 0)
        
        sample_arima_data[location_id] = {
            'forecast': forecast_values.tolist(),
            'confidence_intervals': {
                'upper': (forecast_values * 1.2).tolist(),
                'lower': (forecast_values * 0.8).tolist()
            }
        }
    
    # Test single scenario optimization
    try:
        print("Testing single scenario optimization...")
        result = optimization_engine.optimize_scenario('balanced', sample_arima_data)
        print(f"Single scenario result: Fitness = {result['optimization_summary']['optimization_fitness']:.2f}")
        
        # Test multiple scenario optimization
        print("\nTesting multiple scenario optimization...")
        multi_results = optimization_engine.optimize_multiple_scenarios(
            ['cost_focused', 'time_focused', 'balanced'], 
            sample_arima_data
        )
        print(f"Multi-scenario results: {len(multi_results['scenario_results'])} scenarios completed")
        print(f"Best scenarios: {multi_results['best_scenarios']}")
        
    except Exception as e:
        print(f"Error during optimization test: {str(e)}")