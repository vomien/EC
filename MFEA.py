import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class Individual:
    """Cá thể trong MFEA"""
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome  # Biểu diễn thống nhất
        self.factorial_costs = {}  # Chi phí trên từng nhiệm vụ
        self.factorial_ranks = {}  # Thứ hạng trên từng nhiệm vụ
        self.scalar_fitness = float('inf')  # Fitness tổng thể
        self.skill_factor = None  # Nhiệm vụ thực hiện tốt nhất

class TSPProblem:
    """Bài toán TSP (Traveling Salesman Problem)"""
    def __init__(self, n_cities: int = None, filepath: str = None):
        self.task_id = 0
        
        if filepath:
            self.load_from_file(filepath)
        elif n_cities:
            self.n_cities = n_cities
            # Tạo ma trận khoảng cách ngẫu nhiên
            self.distance_matrix = np.random.rand(n_cities, n_cities) * 100
            self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
            np.fill_diagonal(self.distance_matrix, 0)
        else:
            raise ValueError("Phải cung cấp n_cities hoặc filepath")
    
    def load_from_file(self, filepath: str):
        print(f"Đọc TSP từ file: {filepath}")
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        # Thử đọc dạng tọa độ (id x y hoặc x y)
        try:
            coords = []
            city_ids = []
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: id x y
                    city_id = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    city_ids.append(city_id)
                    coords.append([x, y])
                elif len(parts) >= 2:
                    # Format: x y
                    x, y = float(parts[0]), float(parts[1])
                    coords.append([x, y])
            
            if len(coords) == 0:
                raise ValueError("Không tìm thấy dữ liệu tọa độ")
            
            coords = np.array(coords)
            self.n_cities = len(coords)
            self.city_ids = city_ids if city_ids else list(range(self.n_cities))
            
            # Tính ma trận khoảng cách Euclidean
            self.distance_matrix = np.zeros((self.n_cities, self.n_cities))
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        dist = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                                     (coords[i][1] - coords[j][1])**2)
                        self.distance_matrix[i][j] = dist
            
            print(f"  Đọc thành công {self.n_cities} thành phố từ tọa độ")
            print(f"  Phạm vi tọa độ: X=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}], Y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]")
            return
        except:
            pass
        
        # Thử đọc dạng ma trận khoảng cách
        try:
            data = []
            for line in lines:
                row = [float(x) for x in line.split()]
                data.append(row)
            
            self.distance_matrix = np.array(data)
            self.n_cities = len(self.distance_matrix)
            self.city_ids = list(range(self.n_cities))
            print(f"  Đọc thành công ma trận {self.n_cities}x{self.n_cities}")
            return
        except Exception as e:
            raise ValueError(f"Không thể đọc file TSP: {e}")
        
    def decode(self, chromosome: np.ndarray) -> List[int]:
        """Giải mã chromosome thành tour TSP"""
        indices = np.argsort(chromosome[:self.n_cities])
        return indices.tolist()
    
    def evaluate(self, chromosome: np.ndarray) -> float:
        """Tính tổng độ dài tour"""
        tour = self.decode(chromosome)
        total_distance = 0
        for i in range(len(tour)):
            total_distance += self.distance_matrix[tour[i], tour[(i+1) % len(tour)]]
        return total_distance

class KnapsackProblem:
    """Bài toán Knapsack"""
    def __init__(self, n_items: int = None, capacity_ratio: float = 0.5, filepath: str = None):
        self.task_id = 1
        
        if filepath:
            self.load_from_file(filepath)
        elif n_items:
            self.n_items = n_items
            # Tạo trọng lượng và giá trị ngẫu nhiên
            self.weights = np.random.randint(1, 50, n_items)
            self.values = np.random.randint(1, 100, n_items)
            self.capacity = int(capacity_ratio * np.sum(self.weights))
        else:
            raise ValueError("Phải cung cấp n_items hoặc filepath")
    
    def load_from_file(self, filepath: str):
        
        print(f"Đọc Knapsack từ file: {filepath}")
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        try:
            # Thử format 1: n capacity
            first_line = lines[0].split()
            if len(first_line) == 2:
                self.n_items = int(first_line[0])
                self.capacity = int(first_line[1])
                start_idx = 1
            else:
                # Thử format 2: capacity ở dòng 1, n ở dòng 2
                self.capacity = int(first_line[0])
                self.n_items = int(lines[1])
                start_idx = 2
            
            # Đọc weights và values
            self.weights = []
            self.values = []
            
            for i in range(start_idx, min(start_idx + self.n_items, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 2:
                    self.weights.append(int(parts[0]))
                    self.values.append(int(parts[1]))
            
            self.weights = np.array(self.weights)
            self.values = np.array(self.values)
            self.n_items = len(self.weights)
            
            print(f"  Đọc thành công {self.n_items} items, capacity = {self.capacity}")
            print(f"  Tổng trọng lượng: {np.sum(self.weights)}")
            print(f"  Tổng giá trị: {np.sum(self.values)}")
            
        except Exception as e:
            raise ValueError(f"Không thể đọc file Knapsack: {e}")
        
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """Giải mã chromosome thành vector chọn/không chọn"""
        # Sắp xếp theo giá trị chromosome, chọn items cho đến khi vượt capacity
        indices = np.argsort(chromosome[:self.n_items])[::-1]
        selected = np.zeros(self.n_items, dtype=int)
        total_weight = 0
        
        for idx in indices:
            if total_weight + self.weights[idx] <= self.capacity:
                selected[idx] = 1
                total_weight += self.weights[idx]
                
        return selected
    
    def evaluate(self, chromosome: np.ndarray) -> float:
        """Tính tổng giá trị (âm vì ta minimize)"""
        selected = self.decode(chromosome)
        total_value = np.sum(selected * self.values)
        return -total_value  # Âm vì MFEA minimize

class MFEA:
    """Multifactorial Evolutionary Algorithm"""
    def __init__(self, 
                 tasks: List,
                 pop_size: int = 100,
                 max_gen: int = 200,
                 rmp: float = 0.3,
                 mutation_rate: float = 0.1):
        self.tasks = tasks
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.rmp = rmp  # Random mating probability
        self.mutation_rate = mutation_rate
        
        # Chiều dài chromosome = max của tất cả các nhiệm vụ
        self.dim = max(task.n_cities if hasattr(task, 'n_cities') 
                      else task.n_items for task in tasks)
        
        self.population = []
        self.best_solutions = {i: None for i in range(len(tasks))}
        self.history = {i: [] for i in range(len(tasks))}
        
    def initialize_population(self):
        """Khởi tạo quần thể ngẫu nhiên"""
        self.population = []
        for _ in range(self.pop_size):
            chromosome = np.random.rand(self.dim)
            ind = Individual(chromosome)
            self.population.append(ind)
            
        # Đánh giá trên tất cả các nhiệm vụ
        for ind in self.population:
            for task in self.tasks:
                ind.factorial_costs[task.task_id] = task.evaluate(ind.chromosome)
                
    def compute_scalar_fitness(self):
        """Tính scalar fitness và skill factor"""
        # Tính factorial rank cho từng nhiệm vụ
        for task in self.tasks:
            task_id = task.task_id
            # Sắp xếp theo factorial cost
            sorted_pop = sorted(self.population, 
                              key=lambda x: x.factorial_costs[task_id])
            for rank, ind in enumerate(sorted_pop):
                ind.factorial_ranks[task_id] = rank + 1
        
        # Tính scalar fitness và skill factor
        for ind in self.population:
            best_rank = min(ind.factorial_ranks.values())
            ind.scalar_fitness = 1.0 / best_rank
            ind.skill_factor = min(ind.factorial_ranks.items(), 
                                  key=lambda x: x[1])[0]
            
    def selection(self) -> Tuple[Individual, Individual]:
        """Chọn cha mẹ bằng tournament selection"""
        def tournament():
            i1, i2 = random.sample(range(self.pop_size), 2)
            return self.population[i1] if \
                   self.population[i1].scalar_fitness > self.population[i2].scalar_fitness \
                   else self.population[i2]
        return tournament(), tournament()
    
    def crossover(self, p1: Individual, p2: Individual) -> Tuple[np.ndarray, np.ndarray]:
        """Lai ghép SBX (Simulated Binary Crossover)"""
        c1 = np.copy(p1.chromosome)
        c2 = np.copy(p2.chromosome)
        
        for i in range(self.dim):
            if random.random() < 0.5:
                beta = random.random()
                c1[i] = 0.5 * ((1 + beta) * p1.chromosome[i] + 
                              (1 - beta) * p2.chromosome[i])
                c2[i] = 0.5 * ((1 - beta) * p1.chromosome[i] + 
                              (1 + beta) * p2.chromosome[i])
                
        return c1, c2
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Đột biến"""
        mutated = np.copy(chromosome)
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                mutated[i] = random.random()
        return mutated
    
    def evolve(self):
        """Tiến hóa quần thể"""
        offspring = []
        
        while len(offspring) < self.pop_size:
            p1, p2 = self.selection()
            
            # Assortative mating
            if p1.skill_factor == p2.skill_factor or random.random() < self.rmp:
                c1_chromo, c2_chromo = self.crossover(p1, p2)
                
                # Đột biến
                c1_chromo = self.mutate(c1_chromo)
                c2_chromo = self.mutate(c2_chromo)
                
                # Tạo con
                c1 = Individual(c1_chromo)
                c2 = Individual(c2_chromo)
                
                # Vertical cultural transmission: con thừa hưởng skill factor của cha
                for child, parent in [(c1, p1), (c2, p2)]:
                    task_id = parent.skill_factor
                    task = self.tasks[task_id]
                    child.factorial_costs[task_id] = task.evaluate(child.chromosome)
                    child.skill_factor = task_id
                    
                offspring.extend([c1, c2])
                
        return offspring[:self.pop_size]
    
    def run(self):
        """Chạy thuật toán MFEA"""
        print("Khởi tạo quần thể...")
        self.initialize_population()
        self.compute_scalar_fitness()
        
        for gen in range(self.max_gen):
            # Sinh con
            offspring = self.evolve()
            
            # Đánh giá con trên các nhiệm vụ còn lại (nếu cần)
            for child in offspring:
                for task in self.tasks:
                    if task.task_id not in child.factorial_costs:
                        child.factorial_costs[task.task_id] = \
                            task.evaluate(child.chromosome)
            
            # Kết hợp cha mẹ và con
            combined = self.population + offspring
            
            # Tính scalar fitness cho tất cả
            self.population = combined
            self.compute_scalar_fitness()
            
            # Chọn lọc: giữ pop_size cá thể tốt nhất
            self.population = sorted(self.population, 
                                    key=lambda x: x.scalar_fitness, 
                                    reverse=True)[:self.pop_size]
            
            # Lưu best solution cho từng nhiệm vụ
            for task in self.tasks:
                task_id = task.task_id
                best_ind = min(self.population, 
                             key=lambda x: x.factorial_costs[task_id])
                if self.best_solutions[task_id] is None or \
                   best_ind.factorial_costs[task_id] < \
                   self.best_solutions[task_id].factorial_costs[task_id]:
                    self.best_solutions[task_id] = best_ind
                
                self.history[task_id].append(
                    self.best_solutions[task_id].factorial_costs[task_id]
                )
            
            if (gen + 1) % 20 == 0:
                print(f"Gen {gen+1}/{self.max_gen}")
                for task in self.tasks:
                    task_id = task.task_id
                    print(f"  Task {task_id}: {self.history[task_id][-1]:.2f}")
                    
    def visualize_results(self):
        """Vẽ đồ thị kết quả"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for task in self.tasks:
            task_id = task.task_id
            task_name = "TSP" if task_id == 0 else "Knapsack"
            axes[task_id].plot(self.history[task_id], linewidth=2)
            axes[task_id].set_xlabel('Generation')
            axes[task_id].set_ylabel('Best Cost')
            axes[task_id].set_title(f'{task_name} - Convergence')
            axes[task_id].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('mfea_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def print_solutions(self):
        """In ra giải pháp tốt nhất"""
        print("\n" + "="*60)
        print("KẾT QUẢ TỐI ƯU")
        print("="*60)
        
        # TSP
        tsp = self.tasks[0]
        best_tsp = self.best_solutions[0]
        tour = tsp.decode(best_tsp.chromosome)
        
        print(f"\nTSP (Traveling Salesman Problem):")
        if hasattr(tsp, 'city_ids'):
            tour_ids = [tsp.city_ids[i] for i in tour]
            print(f"  Tour (IDs): {tour_ids}")
        print(f"  Tour (indices): {tour}")
        print(f"  Tổng khoảng cách: {best_tsp.factorial_costs[0]:.2f}")
        
        # Knapsack
        knapsack = self.tasks[1]
        best_knapsack = self.best_solutions[1]
        selected = knapsack.decode(best_knapsack.chromosome)
        selected_items = np.where(selected == 1)[0]
        total_weight = np.sum(knapsack.weights[selected_items])
        total_value = -best_knapsack.factorial_costs[1]  # Đổi dấu lại
        
        print(f"\nKnapsack Problem:")
        print(f"  Các items được chọn: {selected_items.tolist()}")
        print(f"  Số lượng items: {len(selected_items)}/{knapsack.n_items}")
        print(f"  Tổng trọng lượng: {total_weight}/{knapsack.capacity}")
        print(f"  Tổng giá trị: {total_value:.0f}")

# ========== CHẠY THỬ NGHIỆM ==========
if __name__ == "__main__":
    import os
    
    # Kiểm tra xem có file dữ liệu không
    tsp_file = "tsp.txt"
    knapsack_file = "knapsack.txt"
    
    tsp = TSPProblem(filepath=tsp_file)
    knapsack = KnapsackProblem(filepath=knapsack_file)
    
    # Chạy MFEA
    print("\n" + "="*60)
    print("BẮT ĐẦU CHẠY MFEA")
    print("="*60)
    mfea = MFEA(
        tasks=[tsp, knapsack],
        pop_size=100,
        max_gen=800,
        rmp=0.3,
        mutation_rate=0.1
    )
    
    mfea.run()
    
    # Hiển thị kết quả
    mfea.print_solutions()
    mfea.visualize_results()
    
