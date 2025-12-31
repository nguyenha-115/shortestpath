"""
Hệ thống benchmark và đánh giá các thuật toán tìm đường
Bao gồm: phạt chuyển đổi, độ dài, số đỉnh duyệt, thời gian
"""

import random
import time
import math
import json
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

from graph import load_graph
from algorithms.dijkstra import dijkstra_with_penalty
from algorithms.a_star import a_star_with_penalty
from algorithms.bidirectional_a_star import bidirectional_a_star_with_penalty
from algorithms.alt_a_star import alt_a_star_with_penalty


class BenchmarkRunner:
    """Class quản lý việc chạy và đánh giá các thuật toán"""
    
    def __init__(self, graph_file, landmarks_coords=None):
        """
        Args:
            graph_file: đường dẫn file graph pickle
            landmarks_coords: list tọa độ landmarks cho ALT A*
        """
        print(f"Đang load graph từ {graph_file}...")
        self.G = load_graph(graph_file)
        print(f"Đã load: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges\n")
        
        # Khởi tạo landmarks cho ALT
        self.landmarks = None
        if landmarks_coords:
            self.landmarks = [self._nearest_node(coord) for coord in landmarks_coords]
            print(f"Landmarks: {len(self.landmarks)} điểm")
        
        # Định nghĩa các thuật toán
        self.algorithms = {
            "Dijkstra": dijkstra_with_penalty,
            "A*": a_star_with_penalty,
            "Bidirectional A*": bidirectional_a_star_with_penalty,
            "ALT A*": alt_a_star_with_penalty
        }
        
    def _nearest_node(self, coord):
        """Tìm node gần nhất với tọa độ cho trước"""
        min_dist = float('inf')
        nearest = None
        for node in self.G.nodes:
            dist = math.hypot(coord[0] - node[0], coord[1] - node[1])
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest
    
    def _compute_path_cost(self, path):
        """Tính tổng chi phí đường đi (edge weight)"""
        if not path or len(path) < 2:
            return 0
        total = 0
        for i in range(len(path) - 1):
            total += self.G[path[i]][path[i + 1]].get("weight", 1)
        return total
    
    def _compute_transfer_penalty(self, path):
        """Tính phạt chuyển đổi (số lần đổi hướng)"""
        if not path or len(path) < 3:
            return 0
        
        penalty = 0
        for i in range(len(path) - 2):
            # Lấy thông tin cạnh
            edge1 = self.G[path[i]][path[i + 1]]
            edge2 = self.G[path[i + 1]][path[i + 2]]
            
            # Nếu đổi loại đường hoặc id khác nhau -> phạt
            if edge1.get('id', '').split('_')[0] != edge2.get('id', '').split('_')[0]:
                penalty += 1
                
        return penalty
    
    def _compute_path_length(self, path):
        """Tính tổng độ dài thực tế (mét)"""
        if not path or len(path) < 2:
            return 0
        total_length = 0
        for i in range(len(path) - 1):
            total_length += self.G[path[i]][path[i + 1]].get("length", 0)
        return total_length
    
    def benchmark_single(self, algo_name, algo_fn, start, goal):
        """
        Benchmark một thuật toán cho một query
        
        Returns:
            dict với các metric: cost, time, nodes_visited, path_length, 
                 transfers, actual_distance
        """
        start_time = time.time()
        
        # Chạy thuật toán
        try:
            if algo_name == "ALT A*":
                path, visited, _ = algo_fn(self.G, start, goal, landmarks=self.landmarks)
            else:
                path, visited, _ = algo_fn(self.G, start, goal)
        except Exception as e:
            print(f"Lỗi khi chạy {algo_name}: {e}")
            return None
        
        elapsed = time.time() - start_time
        
        # Tính các metric
        if not path:
            return {
                "cost": float('inf'),
                "time": elapsed,
                "nodes_visited": len(visited),
                "path_length": 0,
                "transfers": 0,
                "distance_m": 0,
                "success": False
            }
        
        return {
            "cost": self._compute_path_cost(path),
            "time": elapsed,
            "nodes_visited": len(visited),
            "path_length": len(path),
            "transfers": self._compute_transfer_penalty(path),
            "distance_m": self._compute_path_length(path),
            "success": True,
            "path": path
        }
    
    def run_experiments(self, num_trials=20, random_seed=42):
        """
        Chạy thực nghiệm với nhiều query ngẫu nhiên
        
        Args:
            num_trials: số lượng query test
            random_seed: seed cho random để tái lập
            
        Returns:
            dict chứa kết quả chi tiết và trung bình
        """
        random.seed(random_seed)
        nodes_list = list(self.G.nodes)
        
        # Lưu kết quả từng trial
        all_results = {algo: [] for algo in self.algorithms}
        queries = []
        
        print(f"Bắt đầu chạy {num_trials} queries...\n")
        
        for trial in range(num_trials):
            # Chọn ngẫu nhiên start và goal
            start = random.choice(nodes_list)
            goal = random.choice(nodes_list)
            
            while start == goal:
                goal = random.choice(nodes_list)
            
            queries.append((start, goal))
            print(f"Trial {trial + 1}/{num_trials}: Start {start}, Goal {goal}")
            
            # Chạy từng thuật toán
            for algo_name, algo_fn in self.algorithms.items():
                result = self.benchmark_single(algo_name, algo_fn, start, goal)
                if result:
                    all_results[algo_name].append(result)
                    print(f"  {algo_name}: {result['nodes_visited']} nodes, {result['time']:.4f}s")
            print()
        
        # Tính kết quả trung bình
        avg_results = self._compute_averages(all_results)
        
        return {
            "detailed": all_results,
            "average": avg_results,
            "queries": queries
        }
    
    def _compute_averages(self, all_results):
        """Tính trung bình các metric"""
        avg_results = {}
        
        for algo_name, results in all_results.items():
            if not results:
                continue
            
            successful = [r for r in results if r['success']]
            n = len(successful)
            
            if n == 0:
                avg_results[algo_name] = {
                    "success_rate": 0,
                    "avg_cost": 0,
                    "avg_time": 0,
                    "avg_nodes": 0,
                    "avg_path_length": 0,
                    "avg_transfers": 0,
                    "avg_distance": 0
                }
            else:
                avg_results[algo_name] = {
                    "success_rate": n / len(results) * 100,
                    "avg_cost": sum(r['cost'] for r in successful) / n,
                    "avg_time": sum(r['time'] for r in successful) / n,
                    "avg_nodes": sum(r['nodes_visited'] for r in successful) / n,
                    "avg_path_length": sum(r['path_length'] for r in successful) / n,
                    "avg_transfers": sum(r['transfers'] for r in successful) / n,
                    "avg_distance": sum(r['distance_m'] for r in successful) / n
                }
        
        return avg_results
    
    def print_comparison_table(self, avg_results):
        """In bảng so sánh các thuật toán"""
        print("\n" + "="*80)
        print("BẢNG SO SÁNH CÁC THUẬT TOÁN")
        print("="*80 + "\n")
        
        headers = [
            "Thuật toán",
            "Tỉ lệ thành công",
            "Chi phí TB",
            "Thời gian (s)",
            "Số đỉnh duyệt",
            "Độ dài lộ trình",
            "Số lần chuyển",
            "Khoảng cách (m)"
        ]
        
        table_data = []
        for algo_name, metrics in avg_results.items():
            row = [
                algo_name,
                f"{metrics['success_rate']:.1f}%",
                f"{metrics['avg_cost']:.2f}",
                f"{metrics['avg_time']:.5f}",
                f"{metrics['avg_nodes']:.0f}",
                f"{metrics['avg_path_length']:.1f}",
                f"{metrics['avg_transfers']:.1f}",
                f"{metrics['avg_distance']:.1f}"
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def plot_comparison(self, avg_results, save_path="benchmark_results.png"):
        """Vẽ biểu đồ so sánh"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('So sánh các thuật toán tìm đường', fontsize=16)
        
        algorithms = list(avg_results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        # 1. Thời gian chạy
        ax = axes[0, 0]
        times = [avg_results[algo]['avg_time'] for algo in algorithms]
        ax.bar(algorithms, times, color=colors)
        ax.set_ylabel('Thời gian (giây)')
        ax.set_title('Thời gian chạy trung bình')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Số đỉnh duyệt
        ax = axes[0, 1]
        nodes = [avg_results[algo]['avg_nodes'] for algo in algorithms]
        ax.bar(algorithms, nodes, color=colors)
        ax.set_ylabel('Số đỉnh')
        ax.set_title('Số đỉnh được duyệt')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Chi phí
        ax = axes[0, 2]
        costs = [avg_results[algo]['avg_cost'] for algo in algorithms]
        ax.bar(algorithms, costs, color=colors)
        ax.set_ylabel('Chi phí')
        ax.set_title('Chi phí trung bình')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Độ dài lộ trình
        ax = axes[1, 0]
        path_lengths = [avg_results[algo]['avg_path_length'] for algo in algorithms]
        ax.bar(algorithms, path_lengths, color=colors)
        ax.set_ylabel('Số đỉnh')
        ax.set_title('Độ dài lộ trình (số đỉnh)')
        ax.tick_params(axis='x', rotation=45)
        
        # 5. Số lần chuyển đổi
        ax = axes[1, 1]
        transfers = [avg_results[algo]['avg_transfers'] for algo in algorithms]
        ax.bar(algorithms, transfers, color=colors)
        ax.set_ylabel('Số lần')
        ax.set_title('Số lần chuyển đổi đường')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. Khoảng cách thực
        ax = axes[1, 2]
        distances = [avg_results[algo]['avg_distance'] for algo in algorithms]
        ax.bar(algorithms, distances, color=colors)
        ax.set_ylabel('Mét')
        ax.set_title('Khoảng cách thực tế')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ tại: {save_path}\n")
        
    def save_results(self, results, output_file="benchmark_results.json"):
        """Lưu kết quả ra file JSON"""
        # Chuyển đổi queries (tuple) thành list để serialize
        results_copy = results.copy()
        results_copy['queries'] = [
            {"start": list(s), "goal": list(g)} 
            for s, g in results['queries']
        ]
        
        # Loại bỏ path object để giảm kích thước file
        for algo in results_copy['detailed']:
            for result in results_copy['detailed'][algo]:
                if 'path' in result:
                    del result['path']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        
        print(f"Đã lưu kết quả chi tiết tại: {output_file}\n")


def main():
    """Hàm main chạy benchmark"""
    
    # Cấu hình
    GRAPH_FILE = "data/graph/graph_data.pkl"
    LANDMARKS_COORDS = [
        (105.8181305, 21.001585),
        (105.8196143, 21.0030077),
        (105.8202593, 21.0049961),
        (105.820742, 21.0065424)
    ]
    NUM_TRIALS = 20  # Số lượng query test
    
    # Khởi tạo benchmark runner
    runner = BenchmarkRunner(GRAPH_FILE, LANDMARKS_COORDS)
    
    # Chạy thực nghiệm
    results = runner.run_experiments(num_trials=NUM_TRIALS)
    
    # In bảng so sánh
    runner.print_comparison_table(results['average'])
    
    # Vẽ biểu đồ
    runner.plot_comparison(results['average'])
    
    # Lưu kết quả
    runner.save_results(results)
    
    print("="*80)
    print("HOÀN THÀNH!")
    print("="*80)


if __name__ == "__main__":
    main()