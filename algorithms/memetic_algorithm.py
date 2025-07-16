"""
Memetic Algorithm for Wave-influenced BBM Delivery Scheduling
Combines genetic algorithm with local search and oceanographic knowledge
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import copy
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaveCondition(Enum):
    CALM = "calm"
    MODERATE = "moderate"
    ROUGH = "rough"
    SEVERE = "severe"

@dataclass
class WaveData:
    timestamp: datetime
    wave_height: float  # meters
    wave_period: float  # seconds
    wind_speed: float   # m/s
    tide_level: float   # meters
    condition: WaveCondition
    
    @property
    def delivery_feasibility(self) -> float:
        if self.condition == WaveCondition.CALM:
            return 1.0
        elif self.condition == WaveCondition.MODERATE:
            return 0.8
        elif self.condition == WaveCondition.ROUGH:
            return 0.4
        else:
            return 0.1

@dataclass
class ScheduleGene:
    location_id: str
    delivery_time: datetime
    wave_condition: WaveData
    fuel_amount: float
    vessel_id: str = "default"

class MemeticIndividual:
    def __init__(self, schedule: List[ScheduleGene]):
        self.schedule = schedule
        self.fitness = 0.0
        self.wave_penalty = 0.0
        self.cost_penalty = 0.0
        self.time_penalty = 0.0
        self.total_cost = 0.0
        self.feasible = True
        
    def __len__(self):
        return len(self.schedule)
    
    def copy(self):
        return copy.deepcopy(self)

class MemeticAlgorithm:
    def __init__(self, wave_data: Dict[datetime, WaveData], locations: List[str],
                 demands: Dict[str, float], population_size: int = 50, 
                 generations: int = 100):
        self.wave_data = wave_data
        self.locations = locations
        self.demands = demands
        self.population_size = population_size
        self.generations = generations
        
        self.population = []
        self.best_individual = None
        self.convergence_history = []
        
        # Algorithm parameters
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.local_search_rate = 0.3
        self.elitism_rate = 0.2
        
        # Wave-specific parameters
        self.wave_weight = 0.4
        self.cost_weight = 0.3
        self.time_weight = 0.3
        self.max_wave_height = 3.0  # meters
        self.min_delivery_interval = 2  # hours
        
        logger.info(f"Memetic algorithm initialized for {len(locations)} locations")
    
    def initialize_population(self) -> None:
        logger.info("Initializing population with wave-aware schedules...")
        
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_random_schedule()
            individual.fitness = self.evaluate_fitness(individual)
            self.population.append(individual)
        
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.population[0].copy()
    
    def create_random_schedule(self) -> MemeticIndividual:
        schedule = []
        available_times = list(self.wave_data.keys())
        
        for location in self.locations:
            delivery_time = np.random.choice(available_times)
            wave_condition = self.wave_data[delivery_time]
            fuel_amount = self.demands.get(location, 1000)
            
            gene = ScheduleGene(
                location_id=location,
                delivery_time=delivery_time,
                wave_condition=wave_condition,
                fuel_amount=fuel_amount
            )
            schedule.append(gene)
        
        return MemeticIndividual(schedule)
    
    def evaluate_fitness(self, individual: MemeticIndividual) -> float:
        wave_score = self.calculate_wave_score(individual)
        cost_score = self.calculate_cost_score(individual)
        time_score = self.calculate_time_score(individual)
        
        individual.wave_penalty = 1.0 - wave_score
        individual.cost_penalty = 1.0 - cost_score
        individual.time_penalty = 1.0 - time_score
        
        fitness = (self.wave_weight * wave_score + 
                  self.cost_weight * cost_score + 
                  self.time_weight * time_score)
        
        individual.feasible = self.check_feasibility(individual)
        if not individual.feasible:
            fitness *= 0.5
        
        return fitness
    
    def calculate_wave_score(self, individual: MemeticIndividual) -> float:
        total_feasibility = 0.0
        for gene in individual.schedule:
            feasibility = gene.wave_condition.delivery_feasibility
            
            # Additional penalty for very high waves
            if gene.wave_condition.wave_height > self.max_wave_height:
                feasibility *= 0.3
            
            # Bonus for calm conditions
            if gene.wave_condition.condition == WaveCondition.CALM:
                feasibility *= 1.2
            
            total_feasibility += min(feasibility, 1.0)
        
        return total_feasibility / len(individual.schedule)
    
    def calculate_cost_score(self, individual: MemeticIndividual) -> float:
        total_cost = 0.0
        
        for i, gene in enumerate(individual.schedule):
            # Base delivery cost
            base_cost = gene.fuel_amount * 10  # IDR per liter
            
            # Wave condition multiplier
            wave_multiplier = 1.0
            if gene.wave_condition.condition == WaveCondition.ROUGH:
                wave_multiplier = 1.5
            elif gene.wave_condition.condition == WaveCondition.SEVERE:
                wave_multiplier = 3.0
            
            # Time-based cost (fuel, crew overtime)
            time_cost = self.calculate_time_cost(gene.delivery_time)
            
            delivery_cost = (base_cost * wave_multiplier) + time_cost
            total_cost += delivery_cost
        
        individual.total_cost = total_cost
        
        # Normalize to 0-1 (higher cost = lower score)
        max_possible_cost = sum(self.demands.values()) * 50  # Worst case estimate
        return max(0, 1 - (total_cost / max_possible_cost))
    
    def calculate_time_score(self, individual: MemeticIndividual) -> float:
        schedule_times = [gene.delivery_time for gene in individual.schedule]
        schedule_times.sort()
        
        # Check minimum intervals between deliveries
        interval_violations = 0
        for i in range(1, len(schedule_times)):
            time_diff = (schedule_times[i] - schedule_times[i-1]).total_seconds() / 3600
            if time_diff < self.min_delivery_interval:
                interval_violations += 1
        
        # Penalty for clustering deliveries
        time_spread = (schedule_times[-1] - schedule_times[0]).total_seconds() / 3600
        spread_score = min(time_spread / 48, 1.0)  # Prefer 2-day spread
        
        interval_score = 1.0 - (interval_violations / len(schedule_times))
        
        return (spread_score + interval_score) / 2
    
    def calculate_time_cost(self, delivery_time: datetime) -> float:
        hour = delivery_time.hour
        
        # Regular hours (6-18): normal cost
        if 6 <= hour <= 18:
            return 1000
        # Evening (18-22): higher cost
        elif 18 < hour <= 22:
            return 1500
        # Night (22-6): highest cost
        else:
            return 2000
    
    def check_feasibility(self, individual: MemeticIndividual) -> bool:
        for gene in individual.schedule:
            # Cannot deliver in severe conditions
            if gene.wave_condition.condition == WaveCondition.SEVERE:
                return False
            
            # Cannot deliver with very high waves
            if gene.wave_condition.wave_height > self.max_wave_height * 1.5:
                return False
        
        return True
    
    def tournament_selection(self, tournament_size: int = 3) -> MemeticIndividual:
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def order_crossover(self, parent1: MemeticIndividual, parent2: MemeticIndividual) -> Tuple[MemeticIndividual, MemeticIndividual]:
        size = len(parent1.schedule)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        child1_schedule = [None] * size
        child2_schedule = [None] * size
        
        # Copy segments
        child1_schedule[start:end] = parent1.schedule[start:end]
        child2_schedule[start:end] = parent2.schedule[start:end]
        
        # Fill remaining with wave-aware logic
        self.fill_schedule_intelligent(child1_schedule, parent2.schedule, start, end)
        self.fill_schedule_intelligent(child2_schedule, parent1.schedule, start, end)
        
        child1 = MemeticIndividual(child1_schedule)
        child2 = MemeticIndividual(child2_schedule)
        
        return child1, child2
    
    def fill_schedule_intelligent(self, child_schedule: List, parent_schedule: List, start: int, end: int):
        used_locations = {gene.location_id for gene in child_schedule[start:end] if gene}
        
        parent_remaining = [gene for gene in parent_schedule if gene.location_id not in used_locations]
        
        # Fill remaining positions
        remaining_idx = 0
        for i in range(len(child_schedule)):
            if child_schedule[i] is None and remaining_idx < len(parent_remaining):
                child_schedule[i] = parent_remaining[remaining_idx]
                remaining_idx += 1
    
    def wave_aware_mutation(self, individual: MemeticIndividual) -> MemeticIndividual:
        mutated = individual.copy()
        
        # Select random gene for mutation
        gene_idx = np.random.randint(0, len(mutated.schedule))
        gene = mutated.schedule[gene_idx]
        
        # Find better wave conditions for this delivery
        available_times = [t for t, wave in self.wave_data.items() 
                          if wave.condition in [WaveCondition.CALM, WaveCondition.MODERATE]]
        
        if available_times:
            new_time = np.random.choice(available_times)
            gene.delivery_time = new_time
            gene.wave_condition = self.wave_data[new_time]
        
        return mutated
    
    def local_search_wave_optimization(self, individual: MemeticIndividual) -> MemeticIndividual:
        improved = individual.copy()
        
        for i, gene in enumerate(improved.schedule):
            current_feasibility = gene.wave_condition.delivery_feasibility
            
            # Try to find better wave conditions within ±12 hours
            current_time = gene.delivery_time
            time_window = [current_time + timedelta(hours=h) for h in range(-12, 13)]
            
            best_time = current_time
            best_feasibility = current_feasibility
            
            for test_time in time_window:
                if test_time in self.wave_data:
                    test_wave = self.wave_data[test_time]
                    if test_wave.delivery_feasibility > best_feasibility:
                        best_time = test_time
                        best_feasibility = test_wave.delivery_feasibility
            
            if best_time != current_time:
                improved.schedule[i].delivery_time = best_time
                improved.schedule[i].wave_condition = self.wave_data[best_time]
        
        return improved
    
    def evolve(self) -> Dict[str, Any]:
        logger.info("Starting memetic algorithm evolution...")
        start_time = time.time()
        
        self.initialize_population()
        
        for generation in range(self.generations):
            new_population = []
            
            # Elitism
            elite_size = int(self.population_size * self.elitism_rate)
            new_population.extend(self.population[:elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self.wave_aware_mutation(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self.wave_aware_mutation(child2)
                
                # Local search (memetic component)
                if np.random.random() < self.local_search_rate:
                    child1 = self.local_search_wave_optimization(child1)
                if np.random.random() < self.local_search_rate:
                    child2 = self.local_search_wave_optimization(child2)
                
                # Evaluate
                child1.fitness = self.evaluate_fitness(child1)
                child2.fitness = self.evaluate_fitness(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best
            if self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0].copy()
            
            # Track convergence
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            diversity = np.std([ind.fitness for ind in self.population])
            
            self.convergence_history.append({
                'generation': generation,
                'best_fitness': self.best_individual.fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity,
                'best_cost': self.best_individual.total_cost
            })
            
            if generation % 20 == 0:
                logger.info(f"Generation {generation}: Best fitness = {self.best_individual.fitness:.3f}")
        
        computation_time = time.time() - start_time
        
        results = {
            'best_schedule': self.format_best_schedule(),
            'best_fitness': self.best_individual.fitness,
            'total_cost': self.best_individual.total_cost,
            'convergence_history': self.convergence_history,
            'computation_time': computation_time,
            'feasible': self.best_individual.feasible,
            'wave_analysis': self.analyze_wave_impact(),
            'schedule_metrics': self.calculate_schedule_metrics()
        }
        
        logger.info(f"Memetic algorithm completed in {computation_time:.2f}s")
        return results
    
    def format_best_schedule(self) -> List[Dict]:
        schedule_data = []
        
        for gene in sorted(self.best_individual.schedule, key=lambda x: x.delivery_time):
            schedule_data.append({
                'location': gene.location_id,
                'delivery_time': gene.delivery_time.strftime('%Y-%m-%d %H:%M'),
                'fuel_amount': gene.fuel_amount,
                'wave_height': gene.wave_condition.wave_height,
                'wave_condition': gene.wave_condition.condition.value,
                'feasibility': gene.wave_condition.delivery_feasibility,
                'tide_level': gene.wave_condition.tide_level
            })
        
        return schedule_data
    
    def analyze_wave_impact(self) -> Dict[str, Any]:
        wave_conditions = [gene.wave_condition.condition.value for gene in self.best_individual.schedule]
        wave_heights = [gene.wave_condition.wave_height for gene in self.best_individual.schedule]
        feasibilities = [gene.wave_condition.delivery_feasibility for gene in self.best_individual.schedule]
        
        return {
            'avg_wave_height': np.mean(wave_heights),
            'max_wave_height': np.max(wave_heights),
            'avg_feasibility': np.mean(feasibilities),
            'condition_distribution': {
                'calm': wave_conditions.count('calm'),
                'moderate': wave_conditions.count('moderate'), 
                'rough': wave_conditions.count('rough'),
                'severe': wave_conditions.count('severe')
            },
            'weather_delay_risk': 1.0 - np.mean(feasibilities)
        }
    
    def calculate_schedule_metrics(self) -> Dict[str, Any]:
        delivery_times = [gene.delivery_time for gene in self.best_individual.schedule]
        delivery_times.sort()
        
        # Time span
        time_span = (delivery_times[-1] - delivery_times[0]).total_seconds() / 3600
        
        # Time distribution
        hours = [dt.hour for dt in delivery_times]
        
        return {
            'total_deliveries': len(delivery_times),
            'schedule_span_hours': time_span,
            'avg_daily_deliveries': len(delivery_times) / max(1, time_span / 24),
            'peak_hour': max(set(hours), key=hours.count),
            'night_deliveries': sum(1 for h in hours if h < 6 or h > 22),
            'optimal_conditions': sum(1 for gene in self.best_individual.schedule 
                                    if gene.wave_condition.condition == WaveCondition.CALM)
        }

def create_sample_wave_data(start_date: datetime, days: int = 7) -> Dict[datetime, WaveData]:
    wave_data = {}
    
    for day in range(days):
        for hour in range(0, 24, 2):  # Every 2 hours
            timestamp = start_date + timedelta(days=day, hours=hour)
            
            # Simulate wave patterns
            base_height = 1.0 + 0.5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            wave_height = base_height + np.random.normal(0, 0.3)
            wave_height = max(0.2, wave_height)
            
            # Determine condition
            if wave_height < 0.8:
                condition = WaveCondition.CALM
            elif wave_height < 1.5:
                condition = WaveCondition.MODERATE
            elif wave_height < 2.5:
                condition = WaveCondition.ROUGH
            else:
                condition = WaveCondition.SEVERE
            
            wave_data[timestamp] = WaveData(
                timestamp=timestamp,
                wave_height=wave_height,
                wave_period=6 + np.random.normal(0, 1),
                wind_speed=wave_height * 5 + np.random.normal(0, 2),
                tide_level=np.sin(2 * np.pi * hour / 12) * 0.5,  # Tidal cycle
                condition=condition
            )
    
    return wave_data

# Example usage
if __name__ == "__main__":
    # Sample data
    locations = ['SPBU_Coast_01', 'SPBU_Coast_02', 'SPBU_Island_01', 'SPBU_Offshore_01']
    demands = {'SPBU_Coast_01': 5000, 'SPBU_Coast_02': 3000, 'SPBU_Island_01': 2000, 'SPBU_Offshore_01': 1500}
    
    start_date = datetime(2024, 1, 1)
    wave_data = create_sample_wave_data(start_date, days=3)
    
    print("Memetic Algorithm Test:")
    print(f"Locations: {len(locations)}")
    print(f"Wave data points: {len(wave_data)}")
    print(f"Total demand: {sum(demands.values())} liters")
    
    # Run algorithm
    memetic = MemeticAlgorithm(wave_data, locations, demands, population_size=30, generations=50)
    results = memetic.evolve()
    
    print(f"\nResults:")
    print(f"Best fitness: {results['best_fitness']:.3f}")
    print(f"Total cost: IDR {results['total_cost']:,.0f}")
    print(f"Computation time: {results['computation_time']:.2f}s")
    print(f"Feasible: {results['feasible']}")
    
    print(f"\nWave Analysis:")
    wave_analysis = results['wave_analysis']
    print(f"Average wave height: {wave_analysis['avg_wave_height']:.2f}m")
    print(f"Weather delay risk: {wave_analysis['weather_delay_risk']:.1%}")
    
    print(f"\nOptimal Schedule:")
    for delivery in results['best_schedule'][:3]:
        print(f"  {delivery['location']}: {delivery['delivery_time']} "
              f"(Wave: {delivery['wave_height']:.1f}m, {delivery['wave_condition']})")
    
    print("\nMemetic algorithm test completed! 🌊")