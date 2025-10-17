import numpy as np
from typing import List, Tuple, Dict

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
                #if len(parts) >= 3:
                # Format: id x y
                city_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                city_ids.append(city_id)
                coords.append([x, y])

                # elif len(parts) >= 2:
                #     # Format: x y
                #     x, y = float(parts[0]), float(parts[1])
                #     coords.append([x, y])
            
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
        #    print(f"  Đọc thành công ma trận {self.n_cities}x{self.n_cities}")
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
            # Format: n capacity
            first_line = lines[0].split()
          
            self.n_items = int(first_line[0])
            self.capacity = int(first_line[1])
            start_idx = 1
           
            
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
            
           # print(f"  Đọc thành công {self.n_items} items, capacity = {self.capacity}")
           # print(f"  Tổng trọng lượng: {np.sum(self.weights)}")
           # print(f"  Tổng giá trị: {np.sum(self.values)}")
            
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
