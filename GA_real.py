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
def tour_length(tour, D):
    length = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i+1) % len(tour)]
        length += D[a, b]
    return length

def distance_to_fitness(dist):
    return 1.0 / (1.0 + dist)


def decode_random_keys(keys):
    return np.argsort(keys)

# SBX crossover
def sbx_crossover(parent1, parent2, eta=15.0):
    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(n):
        if np.random.rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                x1 = min(parent1[i], parent2[i])
                x2 = max(parent1[i], parent2[i])
                rand = np.random.rand()
                beta = 1.0 + (2.0*(x1 - 0.0)/(x2 - x1))
                alpha = 2.0 - beta**(-(eta+1))
                if rand <= 1.0/alpha:
                    betaq = (rand*alpha)**(1.0/(eta+1))
                else:
                    betaq = (1.0/(2.0 - rand*alpha))**(1.0/(eta+1))
                c1 = 0.5*((x1 + x2) - betaq*(x2 - x1))
                # symmetric for c2
                beta = 1.0 + (2.0*(1.0 - x2)/(x2 - x1))
                alpha = 2.0 - beta**(-(eta+1))
                if rand <= 1.0/alpha:
                    betaq = (rand*alpha)**(1.0/(eta+1))
                else:
                    betaq = (1.0/(2.0 - rand*alpha))**(1.0/(eta+1))
                c2 = 0.5*((x1 + x2) + betaq*(x2 - x1))
                # clip
                child1[i] = np.clip(c1, 0.0, 1.0)
                child2[i] = np.clip(c2, 0.0, 1.0)
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
    return child1, child2

# Polynomial mutation
def polynomial_mutation(child, eta=20.0, pm=0.1):
    n = len(child)
    for i in range(n):
        if np.random.rand() < pm:
            x = child[i]
            r = np.random.rand()
            if r < 0.5:
                delta = (2*r)**(1.0/(eta+1)) - 1
            else:
                delta = 1 - (2*(1 - r))**(1.0/(eta+1))
            x_new = x + delta
            child[i] = np.clip(x_new, 0.0, 1.0)
    return child

# Gaussian mutation
def gaussian_mutation(child, sigma=0.1, pm=0.1):
    n = len(child)
    for i in range(n):
        if np.random.rand() < pm:
            child[i] += np.random.normal(0, sigma)
            child[i] = np.clip(child[i], 0.0, 1.0)
    return child

# GA loop
def ga_realkey_tsp(coords, pop_size=50, generations=200,
                   crossover_rate=0.9, mutation_rate=0.2,
                   crossover_operator='SBX', mutation_operator='polynomial',
                   selection_operator='tournament', tournament_k=3,
                   elitism=True,
                   verbose=True):
    n = len(coords)
    D = distance_matrix(coords)
    pop = np.random.rand(pop_size, n)
    best_history = []

    best_so_far = float("inf")
    best_solution = None

    for gen in range(generations):
        # evaluate
        perms = np.array([decode_random_keys(ind) for ind in pop])
        distances = np.array([tour_length(p, D) for p in perms])
        fitness_vals = distance_to_fitness(distances)

        best_idx = np.argmin(distances)
        if distances[best_idx] < best_so_far:
            best_so_far = distances[best_idx]
            best_solution = pop[best_idx].copy()

        best_history.append(best_so_far)

        if verbose and (gen % max(1, generations//10) == 0 or gen == generations-1):
            print(f"[RealKey GA] Gen {gen}: best distance = {best_so_far:.4f}")

        # selection
        if selection_operator == 'roulette':
            mating_indices = np.random.choice(pop_size, size=pop_size, p=fitness_vals/np.sum(fitness_vals))
            mating_pool = pop[mating_indices]
        else:
            selected = []
            for _ in range(pop_size):
                cand = np.random.choice(pop_size, tournament_k, replace=False)
                best = cand[np.argmax(fitness_vals[cand])]
                selected.append(pop[best])
            mating_pool = np.array(selected)

        # crossover
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = mating_pool[i].copy()
            p2 = mating_pool[(i+1) % pop_size].copy()
            if np.random.rand() < crossover_rate:
                if crossover_operator == 'SBX':
                    c1, c2 = sbx_crossover(p1, p2)
                else:
                    alpha = np.random.rand(len(p1))
                    c1 = alpha * p1 + (1-alpha) * p2
                    c2 = alpha * p2 + (1-alpha) * p1
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1, p2])
        offspring = np.array(offspring)[:pop_size]

        # mutation
        for i in range(pop_size):
            if mutation_operator == 'polynomial':
                offspring[i] = polynomial_mutation(offspring[i], eta=20.0, pm=mutation_rate)
            else:
                offspring[i] = gaussian_mutation(offspring[i], sigma=0.05, pm=mutation_rate)

        # elitism
        if elitism and best_solution is not None:
            worst_idx = np.argmax(distances)
            offspring[worst_idx] = best_solution.copy()

        pop = offspring

    best_perm = decode_random_keys(best_solution)
    best_dist = best_so_far
    return best_perm, best_dist, best_history

# ======================
# chạy thử
coords = read_coords("tsp.txt")

print("\n=== Chạy GA với mã hóa số thực (Random Keys, SBX, Polynomial+Gaussian) ===")
best_tour_real, best_dist_real, hist_real = ga_realkey_tsp(
    coords,
    pop_size=100,
    generations=500,
    crossover_rate=0.9,
    mutation_rate=0.2,
    crossover_operator='SBX',
    mutation_operator='gaussian', # có thể chọn 'polynomial', 'gaussian'
    selection_operator='tournament',
    tournament_k=3,
    elitism=True,
    verbose=True
)
print("Kết quả RealKey GA: best distance =", best_dist_real)
print("Best tour:", best_tour_real)
