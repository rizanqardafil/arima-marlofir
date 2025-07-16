"""
MARLOFIR-P Genetic Algorithm Implementation
Core GA engine for BBM distribution optimization
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GAParameters:
    """GA configuration parameters"""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    tournament_size: int = 5
    adaptive_mutation: bool = True
    local_search: bool = True

@dataclass
class Individual:
    """Individual representation for GA"""
    chromosome: List[int]  # Route sequence
    fitness: float = 0.0
    cost: float = 0.0
    distance: float = 0.0
    fuel_efficiency: float = 0.0
    time_penalty: float = 0.0
    capacity_penalty: float = 0.0

class GeneticAlgorithm:
    """
    Genetic Algorithm for BBM Distribution Optimization (MARLOFIR-P)
    Multi-objective optimization for route planning and resource allocation
    """
    
    def __init__(self, parameters: GAParameters = None):
        self.params = parameters or GAParameters()
        self.population = []
        self.best_individual = None
        self.fitness_history = []
        self.diversity_history = []
        self.generation = 0
        
        # Problem-specific constraints
        self.locations = []
        self.distances = {}
        self.demands = {}
        self.vehicle_capacity = 0
        self.fuel_cost = 0
        self.time_windows = {}
        
        logger.info(f"GA initialized with parameters: {self.params}")
    
    def set_problem_constraints(self, locations: List[str], distances: Dict, 
                              demands: Dict, vehicle_capacity: float, 
                              fuel_cost: float, time_windows: Dict = None):
        """
        Set problem-specific constraints for BBM distribution
        
        Args:
            locations: List of delivery locations
            distances: Distance matrix between locations
            demands: Fuel demand for each location
            vehicle_capacity: Maximum vehicle capacity
            fuel_cost: Cost per liter of fuel
            time_windows: Delivery time windows (optional)
        """
        self.locations = locations
        self.distances = distances
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.fuel_cost = fuel_cost
        self.time_windows = time_windows or {}
        
        logger.info(f"Problem constraints set for {len(locations)} locations")
    
    def initialize_population(self) -> List[Individual]:
        """
        Initialize population with random valid solutions
        
        Returns:
            List of initialized individuals
        """
        logger.info("Initializing population...")
        population = []
        
        for _ in range(self.params.population_size):
            # Create random route (excluding depot at index 0)
            route = list(range(1, len(self.locations)))
            random.shuffle(route)
            
            # Add depot at start and end
            chromosome = [0] + route + [0]
            
            individual = Individual(chromosome=chromosome)
            individual.fitness = self.evaluate_fitness(individual)
            population.append(individual)
        
        self.population = population
        logger.info(f"Population initialized with {len(population)} individuals")
        return population
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Multi-objective fitness evaluation
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        # Calculate route cost
        total_distance = 0
        total_cost = 0
        capacity_penalty = 0
        time_penalty = 0
        
        chromosome = individual.chromosome
        current_load = 0
        
        for i in range(len(chromosome) - 1):
            from_loc = chromosome[i]
            to_loc = chromosome[i + 1]
            
            # Distance and cost calculation
            if (from_loc, to_loc) in self.distances:
                distance = self.distances[(from_loc, to_loc)]
            else:
                distance = self.distances.get((to_loc, from_loc), 1000)  # High penalty for missing
            
            total_distance += distance
            total_cost += distance * self.fuel_cost
            
            # Capacity constraint
            if to_loc != 0:  # Not returning to depot
                location_name = self.locations[to_loc] if to_loc < len(self.locations) else f"loc_{to_loc}"
                demand = self.demands.get(location_name, 0)
                current_load += demand
                
                if current_load > self.vehicle_capacity:
                    capacity_penalty += (current_load - self.vehicle_capacity) * 100
            else:
                current_load = 0  # Reset at depot
        
        # Time window penalties (if applicable)
        if self.time_windows:
            for loc_idx in chromosome[1:-1]:  # Exclude depot
                location_name = self.locations[loc_idx] if loc_idx < len(self.locations) else f"loc_{loc_idx}"
                if location_name in self.time_windows:
                    # Simplified time penalty calculation
                    time_penalty += 10  # Placeholder
        
        # Fuel efficiency calculation
        fuel_efficiency = 1000 / (total_distance + 1) if total_distance > 0 else 0
        
        # Store individual metrics
        individual.cost = total_cost
        individual.distance = total_distance
        individual.fuel_efficiency = fuel_efficiency
        individual.time_penalty = time_penalty
        individual.capacity_penalty = capacity_penalty
        
        # Multi-objective fitness (minimize cost and distance, maximize efficiency)
        fitness = 1000 / (total_cost + 1) - capacity_penalty - time_penalty + fuel_efficiency
        
        return max(fitness, 0.1)  # Ensure positive fitness
    
    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """
        Tournament selection for parent selection
        
        Args:
            tournament_size: Size of tournament (default from params)
            
        Returns:
            Selected individual
        """
        tournament_size = tournament_size or self.params.tournament_size
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda ind: ind.fitness)
    
    def order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Order crossover (OX) for route representation
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        p1_chrom = parent1.chromosome[1:-1]  # Exclude depot
        p2_chrom = parent2.chromosome[1:-1]
        
        size = len(p1_chrom)
        
        # Select crossover points
        start, end = sorted(random.sample(range(size), 2))
        
        # Create offspring
        offspring1 = [None] * size
        offspring2 = [None] * size
        
        # Copy selected segments
        offspring1[start:end] = p1_chrom[start:end]
        offspring2[start:end] = p2_chrom[start:end]
        
        # Fill remaining positions
        def fill_offspring(offspring, other_parent):
            remaining = [x for x in other_parent if x not in offspring]
            j = 0
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = remaining[j]
                    j += 1
        
        fill_offspring(offspring1, p2_chrom)
        fill_offspring(offspring2, p1_chrom)
        
        # Add depot
        child1 = Individual(chromosome=[0] + offspring1 + [0])
        child2 = Individual(chromosome=[0] + offspring2 + [0])
        
        return child1, child2
    
    def swap_mutation(self, individual: Individual) -> Individual:
        """
        Swap mutation for route representation
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = copy.deepcopy(individual)
        chromosome = mutated.chromosome[1:-1]  # Exclude depot
        
        if len(chromosome) > 1:
            # Select two random positions
            i, j = random.sample(range(len(chromosome)), 2)
            # Swap
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
            
            mutated.chromosome = [0] + chromosome + [0]
        
        return mutated
    
    def adaptive_mutation_rate(self) -> float:
        """
        Adaptive mutation rate based on population diversity
        
        Returns:
            Adjusted mutation rate
        """
        if not self.params.adaptive_mutation:
            return self.params.mutation_rate
        
        # Calculate population diversity
        diversity = self.calculate_diversity()
        
        # Increase mutation rate if diversity is low
        if diversity < 0.3:
            return min(self.params.mutation_rate * 2, 0.3)
        else:
            return self.params.mutation_rate
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity
        
        Returns:
            Diversity score (0-1)
        """
        if len(self.population) < 2:
            return 0.0
        
        fitness_values = [ind.fitness for ind in self.population]
        return np.std(fitness_values) / (np.mean(fitness_values) + 1e-10)
    
    def local_search(self, individual: Individual) -> Individual:
        """
        Local search optimization (2-opt)
        
        Args:
            individual: Individual to improve
            
        Returns:
            Improved individual
        """
        if not self.params.local_search:
            return individual
        
        best = copy.deepcopy(individual)
        improved = True
        
        while improved:
            improved = False
            chromosome = best.chromosome[1:-1]  # Exclude depot
            
            for i in range(len(chromosome) - 1):
                for j in range(i + 2, len(chromosome)):
                    # 2-opt swap
                    new_chromosome = chromosome[:i] + chromosome[i:j+1][::-1] + chromosome[j+1:]
                    
                    new_individual = Individual(chromosome=[0] + new_chromosome + [0])
                    new_individual.fitness = self.evaluate_fitness(new_individual)
                    
                    if new_individual.fitness > best.fitness:
                        best = new_individual
                        improved = True
                        break
                
                if improved:
                    break
        
        return best
    
    def evolve_generation(self) -> None:
        """
        Evolve one generation
        """
        new_population = []
        
        # Elitism - keep best individuals
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elite_size = int(self.params.population_size * self.params.elitism_rate)
        new_population.extend(sorted_population[:elite_size])
        
        # Generate offspring
        current_mutation_rate = self.adaptive_mutation_rate()
        
        while len(new_population) < self.params.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < self.params.crossover_rate:
                child1, child2 = self.order_crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < current_mutation_rate:
                child1 = self.swap_mutation(child1)
            if random.random() < current_mutation_rate:
                child2 = self.swap_mutation(child2)
            
            # Evaluate fitness
            child1.fitness = self.evaluate_fitness(child1)
            child2.fitness = self.evaluate_fitness(child2)
            
            # Local search
            child1 = self.local_search(child1)
            child2 = self.local_search(child2)
            
            new_population.extend([child1, child2])
        
        # Ensure population size
        self.population = new_population[:self.params.population_size]
        
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(current_best)
        
        # Track statistics
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': self.best_individual.fitness,
            'avg_fitness': avg_fitness,
            'diversity': self.calculate_diversity()
        })
        
        self.generation += 1
    
    def optimize(self) -> Dict:
        """
        Main optimization loop
        
        Returns:
            Optimization results
        """
        logger.info("Starting GA optimization...")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.params.generations):
            self.evolve_generation()
            
            # Progress logging
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {self.best_individual.fitness:.2f}")
        
        # Prepare results
        results = {
            'best_solution': {
                'route': self.best_individual.chromosome,
                'fitness': self.best_individual.fitness,
                'cost': self.best_individual.cost,
                'distance': self.best_individual.distance,
                'fuel_efficiency': self.best_individual.fuel_efficiency
            },
            'convergence_history': self.fitness_history,
            'final_population': [
                {
                    'route': ind.chromosome,
                    'fitness': ind.fitness,
                    'cost': ind.cost
                } for ind in self.population
            ]
        }
        
        logger.info(f"GA optimization completed. Best fitness: {self.best_individual.fitness:.2f}")
        return results

# Example usage and testing
if __name__ == "__main__":
    # Example problem setup
    locations = ['Depot', 'SPBU_1', 'SPBU_2', 'SPBU_3', 'SPBU_4']
    
    # Distance matrix (simplified)
    distances = {
        (0, 1): 10, (1, 0): 10,
        (0, 2): 15, (2, 0): 15,
        (0, 3): 20, (3, 0): 20,
        (0, 4): 25, (4, 0): 25,
        (1, 2): 8, (2, 1): 8,
        (1, 3): 12, (3, 1): 12,
        (1, 4): 18, (4, 1): 18,
        (2, 3): 7, (3, 2): 7,
        (2, 4): 14, (4, 2): 14,
        (3, 4): 9, (4, 3): 9
    }
    
    # Demands
    demands = {
        'SPBU_1': 100,
        'SPBU_2': 150,
        'SPBU_3': 80,
        'SPBU_4': 120
    }
    
    # Initialize GA
    ga_params = GAParameters(
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    ga = GeneticAlgorithm(ga_params)
    ga.set_problem_constraints(locations, distances, demands, 500, 1.2)
    
    # Run optimization
    results = ga.optimize()
    
    print("Optimization Results:")
    print(f"Best route: {results['best_solution']['route']}")
    print(f"Best fitness: {results['best_solution']['fitness']:.2f}")
    print(f"Total cost: {results['best_solution']['cost']:.2f}")
    print(f"Total distance: {results['best_solution']['distance']:.2f}")