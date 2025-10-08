import numpy as np
import random

# -----------------------------
# ĐỌC DỮ LIỆU
# -----------------------------
def read_tsp(filename):
    coords = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                x, y = float(parts[1]), float(parts[2])
                coords.append((x, y))
    return coords

def read_knapsack(filename):
    items = []
    capacity = 0
    with open(filename, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        capacity = int(lines[0])
        for line in lines[1:]:
            w, v = map(int, line.split())
            items.append((w, v))
    return capacity, items

# -----------------------------
# HÀM TSP
# -----------------------------
def tsp_distance(order, coords):
    dist = 0.0
    for i in range(len(order)):
        a, b = coords[order[i]], coords[order[(i + 1) % len(order)]]
        dist += np.hypot(a[0] - b[0], a[1] - b[1])
    return dist

def two_opt_improve(order, coords, max_swaps=20):
    best = order.copy()
    best_len = tsp_distance(best, coords)
    swaps = 0
    improved = True
    while improved and swaps < max_swaps:
        improved = False
        i = np.random.randint(0, len(order) - 2)
        j = np.random.randint(i + 2, len(order))
        new = best.copy()
        new[i+1:j+1] = new[i+1:j+1][::-1]
        new_len = tsp_distance(new, coords)
        if new_len < best_len:
            best = new
            best_len = new_len
            improved = True
        swaps += 1
    return best, best_len

# HÀM KNAPSACK
def knapsack_value(solution, items, capacity):
    total_w, total_v = 0, 0
    for i, bit in enumerate(solution):
        if bit:
            total_w += items[i][0]
            total_v += items[i][1]
    if total_w > capacity:
        return 0
    return total_v

def knapsack_local_improve(sol, items, capacity, max_swaps=20):
    best = sol.copy()
    best_val = knapsack_value(best, items, capacity)
    for _ in range(max_swaps):
        i = np.random.randint(0, len(sol))
        new = best.copy()
        new[i] = 1 - new[i]
        val = knapsack_value(new, items, capacity)
        if val > best_val:
            best, best_val = new, val
    return best, best_val

# KHỞI TẠO & MFEA
def init_population(pop_size, n_tsp, n_knap):
    pop = []
    skill_factor = []
    for _ in range(pop_size):
        if random.random() < 0.5:  # cá thể TSP
            ind = np.random.permutation(n_tsp)
            skill = 0
        else:
            ind = np.random.randint(0, 2, n_knap)
            skill = 1
        pop.append(ind)
        skill_factor.append(skill)
    return pop, skill_factor

def pmx(parent1, parent2):
    parent1 = list(parent1)
    parent2 = list(parent2)
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b] = parent1[a:b]
    for x in parent2[a:b]:
        if x not in parent1[a:b]:
            pos = parent2.index(x)
            while a <= pos < b:
                pos = parent2.index(parent1[pos])
            child[pos] = x
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    return np.array(child)

def mutate_tsp(order):
    a, b = sorted(random.sample(range(len(order)), 2))
    order[a:b] = order[a:b][::-1]
    return order

def mutate_knapsack(sol):
    i = np.random.randint(0, len(sol))
    sol[i] = 1 - sol[i]
    return sol

def MFEA(coords, items, capacity, pop_size=100, max_gen=800):
    n_tsp, n_knap = len(coords), len(items)
    pop, skill_factor = init_population(pop_size, n_tsp, n_knap)

    best_tsp, best_knap = None, None
    best_tsp_fit, best_knap_fit = float("inf"), 0

    for gen in range(max_gen):
        offspring = []
        for _ in range(pop_size // 2):
            p1, p2 = random.sample(range(pop_size), 2)
            sf1, sf2 = skill_factor[p1], skill_factor[p2]

            if sf1 == 0 and sf2 == 0:
                c1 = pmx(pop[p1], pop[p2])
                c2 = pmx(pop[p2], pop[p1])
                if np.random.rand() < 0.1:
                    c1 = mutate_tsp(c1)
                    c2 = mutate_tsp(c2)
                if np.random.rand() < 0.1:
                    c1, _ = two_opt_improve(c1, coords)
                    c2, _ = two_opt_improve(c2, coords)
                offspring += [c1, c2]
                skill_factor += [0, 0]

            elif sf1 == 1 and sf2 == 1:
                c1 = pop[p1].copy()
                c2 = pop[p2].copy()
                if np.random.rand() < 0.5:
                    c1, c2 = c2, c1
                if np.random.rand() < 0.1:
                    c1 = mutate_knapsack(c1)
                    c2 = mutate_knapsack(c2)
                if np.random.rand() < 0.1:
                    c1, _ = knapsack_local_improve(c1, items, capacity)
                    c2, _ = knapsack_local_improve(c2, items, capacity)
                offspring += [c1, c2]
                skill_factor += [1, 1]

        pop += offspring

        # Đánh giá
        fitness_tsp, fitness_knap = [], []
        for i, ind in enumerate(pop):
            if skill_factor[i] == 0:
                fitness = tsp_distance(ind, coords)
                fitness_tsp.append((fitness, i))
            else:
                fitness = -knapsack_value(ind, items, capacity)
                fitness_knap.append((fitness, i))

        # Giữ lại top
        fitness_tsp.sort()
        fitness_knap.sort()
        new_pop = []
        new_sf = []
        for _, idx in fitness_tsp[:pop_size // 2]:
            new_pop.append(pop[idx])
            new_sf.append(0)
        for _, idx in fitness_knap[:pop_size // 2]:
            new_pop.append(pop[idx])
            new_sf.append(1)

        pop, skill_factor = new_pop, new_sf

        # Lưu best
        cur_tsp = fitness_tsp[0][0]
        cur_knap = -fitness_knap[0][0]
        if cur_tsp < best_tsp_fit:
            best_tsp_fit = cur_tsp
            best_tsp = pop[0]
        if cur_knap > best_knap_fit:
            best_knap_fit = cur_knap
            best_knap = pop[-1]

        if gen % 10 == 0:
            print(f"Gen {gen}: best TSP = {best_tsp_fit:.2f}, best Knapsack = {best_knap_fit}")

    return best_tsp, best_tsp_fit, best_knap, best_knap_fit

# -----------------------------
# CHẠY THỬ
# -----------------------------
coords = read_tsp("tsp.txt")
capacity, items = read_knapsack("knapsack.txt")

best_tsp, best_tsp_fit, best_knap, best_knap_fit = MFEA(coords, items, capacity)
print("\nKết quả cuối:")
print("TSP best distance:", best_tsp_fit)
print("Knapsack best value:", best_knap_fit)
