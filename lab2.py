from pdb import Pdb
from queue import Queue 
import time
import psutil
import sys

states = [
    [2, 5, 7, 0, 3, 0, 6, 4],
    [0, 6, 4, 7, 0, 2, 5, 2],
    [7, 1, 2, 2, 0, 6, 3, 0],
    [0, 2, 0, 5, 1, 5, 6, 3],
    [5, 0, 0, 1, 7, 0, 6, 3],
    [3, 1, 7, 2, 2, 7, 4, 1],
    [0, 3, 6, 0, 2, 2, 1, 7],
    [5, 3, 0, 4, 7, 3, 1, 2],
    [1, 3, 5, 7, 4, 0, 6, 4],
    [3, 5, 7, 1, 6, 0, 2, 3],
    [1, 4, 6, 8, 3, 7, 7, 5],
    [4, 6, 0, 3, 1, 7, 5, 4], 
    [3, 7, 4, 2, 2, 6, 1, 4],
    [2, 1, 3, 7, 4, 0, 3, 5],
    [1, 4, 7, 2, 2, 7, 1, 3],
    [1, 3, 5, 7, 2, 0, 6, 5],
    [6, 1, 5, 0, 0, 3, 7, 4],
    [5, 1, 4, 3, 0, 7, 1, 4],
    [1, 3, 5, 7, 1, 0, 6, 3],
    [1, 3, 5, 7, 2, 0, 2, 4]
]

def F2(state):
    attacking_pairs = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                attacking_pairs += 1
    return attacking_pairs

def is_safe1(state):
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] == state[j] or abs(i - j) == abs(state[i] - state[j]):
                return False;
    return True

def is_safe2(state, col, row):
    for i in range(col):
        if state[i] == row or abs(i - col) == abs(state[i] - row):
            return False
    return True

def BFS(init_state):
    queue = Queue()
    queue.put(init_state)
    nodes_in_memory = 0
    generated_nodes  = 0

    start_time = time.time()

    while not queue.empty():
        current_state = queue.get()
        nodes_in_memory -= 1
        generated_nodes += 1

        if is_safe1(current_state):
            end_time = time.time()
            memory_usage = psutil.Process().memory_info().rss
            return current_state, generated_nodes, nodes_in_memory, end_time - start_time, memory_usage
        
        for col in range(8):
            for row in range(8):
                if current_state[col] != row:
                    new_state = list(current_state)
                    new_state[col] = row
                    queue.put(new_state)
                    nodes_in_memory += 1
                    generated_nodes += 1

    end_time = time.time()
    return None, generated_nodes, nodes_in_memory, end_time - start_time, memory_usage

def RBFS(init_state, heuristic_func):
    stack = [(list(init_state), sys.maxsize, None)]
    generated_nodes = 0
    nodes_in_memory = 0
    start_time = time.time()

    while stack:
        state, f_limit, parent = stack[-1]
        if heuristic_func(state) == 0:
            end_time = time.time()
            memory_usage = psutil.Process().memory_info().rss
            return state, generated_nodes, nodes_in_memory, end_time - start_time, memory_usage

        successors = []

        for col in range(len(state)):
            for row in range(len(state)):
                if is_safe2(state, col, row):
                    new_state = list(state)
                    new_state[col] = row
                    successors.append((new_state, heuristic_func(new_state)))
                    generated_nodes +=1
                    nodes_in_memory +=1

        if not successors:
            stack.pop()
            continue

        generated_nodes += len(successors)

        successors.sort(key=lambda x: x[1])
        best_state, best_h = successors[0]

        if best_h > f_limit:
            stack.pop()
            continue

        nodes_in_memory += 1
        next_best = successors[1][1]

        if parent is not None:
            parent[2] = best_h

        stack[-1] = (state, f_limit, [best_state, next_best, parent])

        stack.append((best_state, min(f_limit, next_best), None))

    return None, generated_nodes, nodes_in_memory, None, None

if __name__ == "__main__":
    i = 1
    for state in states: 
        print("Експеримент №" + str(i) + ":")
        print("Початковий стан: " + str(state))
        bfs_result = BFS(state)
        print("BFS: " + str(bfs_result))
        i += 1 
        rbfs_result = RBFS(state, F2)
        print("RBFS: " + str(rbfs_result))