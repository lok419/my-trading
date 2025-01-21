from collections import deque
import numpy as np

class ExchangeRateGraph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, w):
        if u in self.graph:
            self.graph[u].append((v, w))
        else:
            self.graph[u] = [(v, w)]

    def bfs_path(self, start_node, end_node):
        if start_node not in self.graph or end_node not in self.graph:
            return None

        visited = set()
        queue = deque([(start_node, [(start_node, 1)])])
        visited.add(start_node)

        while queue:
            current_node, path = queue.popleft()                                  

            if current_node == end_node:
                return path

            if current_node in self.graph:
                for (neighbor, w) in self.graph[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [(neighbor, w)]))
                        visited.add(neighbor)
        return None
    
    def get_all_paths(self, ccy:str) -> dict:
        paths = {}
        for g in self.graph:
            paths[g] = self.bfs_path(g, ccy)
        return paths
    
    def get_all_rates(self, ccy:str) -> dict:
        ex_rates = {}
        for g in self.graph:
            ex_rates[g] = np.prod([x[1] for x in self.bfs_path(g, ccy)])
        return ex_rates