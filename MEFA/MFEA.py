import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from task import TSPProblem, KnapsackProblem

class Individual:
    """Cá thể trong MFEA"""
    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome  # Biểu diễn thống nhất
        self.factorial_costs = {}  # Chi phí trên từng nhiệm vụ
        self.factorial_ranks = {}  # Thứ hạng trên từng nhiệm vụ
        self.scalar_fitness = float('inf')  # Fitness tổng thể
        self.skill_factor = None  # Nhiệm vụ thực hiện tốt nhất

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
                
                # Con thừa hưởng skill factor của cha
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
            
            if (gen + 1) % (100) == 0:
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
    
