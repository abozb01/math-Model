import numpy as np
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
GENES_LENGTH = 10
MUTATION_RATE = 0.1
NUM_GENERATIONS = 50

# Function to optimize
def fitness_function(x):
    return np.sum(x)

# Initialize population
def initialize_population():
    return np.random.randint(2, size=(POPULATION_SIZE, GENES_LENGTH))

# Calculate fitness of each individual in population
def calculate_fitness(population):
    return np.array([fitness_function(individual) for individual in population])

# Select parents for crossover based on tournament selection
def select_parents(population, fitness_scores):
    indices = np.random.choice(len(population), size=2, replace=False)
    return population[indices]

# Perform crossover to produce offspring
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, GENES_LENGTH)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Perform mutation on individual
def mutate(individual):
    mutation_indices = np.random.choice(len(individual), size=int(MUTATION_RATE * GENES_LENGTH), replace=False)
    individual[mutation_indices] = 1 - individual[mutation_indices]
    return individual

# Evolutionary process
def evolve(population):
    fitness_scores = calculate_fitness(population)
    new_population = []

    for _ in range(POPULATION_SIZE // 2):
        parent1 = select_parents(population, fitness_scores)
        parent2 = select_parents(population, fitness_scores)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])

    return np.array(new_population)

# Main function
def main():
    population = initialize_population()

    for generation in range(NUM_GENERATIONS):
        population = evolve(population)

    best_individual = population[np.argmax(calculate_fitness(population))]
    best_fitness = fitness_function(best_individual)

    print("Best solution found:", best_individual)
    print("Fitness of best solution:", best_fitness)

    plt.plot(range(NUM_GENERATIONS), calculate_fitness(population))
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Genetic Algorithm Optimization')
    plt.show()

if __name__ == "__main__":
    main()
