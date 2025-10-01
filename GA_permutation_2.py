import numpy as np

# Đọc dữ liệu
def read_coords(filename):
    coords = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            x, y = float(parts[1]), float(parts[2])
            coords.append((x, y))
    return coords

def distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
            D[i, j] = D[j, i] = d
    return D


# Fitness
def fitness_function(individual, D):
    total_dist = sum(D[individual[i], individual[(i+1) % len(individual)]] 
                     for i in range(len(individual)))
    return 1.0 / total_dist

# Khởi tạo quần thể
def init_population(pop_size, n_cities):
    return np.array([np.random.permutation(n_cities) for _ in range(pop_size)])


# Selection
def roulette_selection(pop, fitness_values):
    probs = fitness_values / np.sum(fitness_values)
    idx = np.random.choice(len(pop), size=len(pop), p=probs)
    return pop[idx]

def tournament_selection(pop, fitness_values, k=3):
    new_pop = []
    for _ in range(len(pop)):
        candidates = np.random.choice(len(pop), k, replace=False)
        best = candidates[np.argmax(fitness_values[candidates])]
        new_pop.append(pop[best])
    return np.array(new_pop)


# Crossover
def order_crossover(parent1, parent2):
    n = len(parent1)
    c1, c2 = sorted(np.random.choice(range(n), 2, replace=False))
    child = np.full(n, -1)
    child[c1:c2+1] = parent1[c1:c2+1]

    p2_id = 0
    for i in range(n):
        pos = (c2 + 1 + i) % n
        if child[pos] == -1:
            while parent2[p2_id] in child:
                p2_id += 1
            child[pos] = parent2[p2_id]
            p2_id += 1
    return child

def pmx_crossover(parent1, parent2):
    n = len(parent1)
    c1, c2 = sorted(np.random.choice(range(n), 2, replace=False))
    child = np.full(n, -1)

    child[c1:c2+1] = parent1[c1:c2+1]

    for i in range(c1, c2+1):
        if parent2[i] not in child:
            pos = i
            while child[pos] != -1:
                pos = np.where(parent2 == parent1[pos])[0][0]
            child[pos] = parent2[i]

    for i in range(n):
        if child[i] == -1:
            child[i] = parent2[i]

    return child


# Mutation
def swap_mutation(individual):
    i, j = np.random.choice(len(individual), 2, replace=False)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

def inversion_mutation(individual):
    a, b = sorted(np.random.choice(len(individual), 2, replace=False))
    individual[a:b] = individual[a:b][::-1]
    return individual


# Genetic Algorithm
def GA_permutation(coords, pop_size, generations, crossover_rate, mutation_rate,
                   crossover_operator="OX", mutation_operator="swap", 
                   selection_operator="tournament", tournament_k=3, 
                   elitism=True, early_stop=50):
    
    n = len(coords)
    D = distance_matrix(coords)
    pop = init_population(pop_size, n)
    best_historic = []

    # Best toàn cục
    best_overall = None
    best_dist_overall = float("inf")
    no_improve = 0

    for gen in range(generations):
        fitness_vals = np.array([fitness_function(ind, D) for ind in pop])
        best_id = np.argmax(fitness_vals)
        best_dist = 1.0 / fitness_vals[best_id]
        elite = pop[best_id].copy()

        # Cập nhật best toàn cục
        if best_dist < best_dist_overall:
            best_dist_overall = best_dist
            best_overall = elite.copy()
            no_improve = 0
        else:
            no_improve += 1

        best_historic.append(best_dist_overall)

        # Kiểm tra stopping criteria
        if early_stop and no_improve >= early_stop:
            print(f"⏹ Dừng sớm ở thế hệ {gen} (không cải thiện {early_stop} thế hệ liên tiếp).")
            break

        # Selection
        if selection_operator == "roulette":
            mating_pool = roulette_selection(pop, fitness_vals)
        else:
            mating_pool = tournament_selection(pop, fitness_vals, k=tournament_k)

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[(i + 1) % pop_size]
            if np.random.rand() < crossover_rate:
                if crossover_operator == "OX":
                    child1 = order_crossover(parent1.copy(), parent2.copy())
                    child2 = order_crossover(parent2.copy(), parent1.copy())
                else:  # PMX
                    child1 = pmx_crossover(parent1.copy(), parent2.copy())
                    child2 = pmx_crossover(parent2.copy(), parent1.copy())
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        offspring = np.array(offspring)[:pop_size]

        # Mutation
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                if mutation_operator == "swap":
                    offspring[i] = swap_mutation(offspring[i])
                else:  # inversion
                    offspring[i] = inversion_mutation(offspring[i])

        # Elitism
        if elitism:
            worst_id = np.argmin([fitness_function(ind, D) for ind in offspring])
            offspring[worst_id] = elite

        pop = offspring

    return best_overall, best_dist_overall, best_historic

coords = read_coords("tsp.txt")

print("=== Chạy GA TSP ===")
best_tour, best_dist, hist = GA_permutation(
    coords,
    pop_size=100,
    generations=1000,       
    crossover_rate=0.9,
    mutation_rate=0.2,
    crossover_operator="PMX",       # "OX" or "PMX"
    mutation_operator="inversion",  # "swap" or "inversion"
    selection_operator="tournament", # "tournament" or "roulette"
    tournament_k=3,
    elitism=True,
    early_stop=50       
)

print("Best distance:", best_dist)
print("Best tour:", best_tour)

