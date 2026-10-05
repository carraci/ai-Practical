from collections import deque
import time

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': ['H'],
    'F': [],
    'G': [],
    'H': []
}

def bfs(graph, start, goal):
    queue = deque([start])
    visited = set()

    while queue:
        node = queue.popleft()

        if node not in visited:
            print(node, end=" ")

            if node == goal:
                return True

            visited.add(node)

            for neighbour in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)

    return False

def iterative_dfs(graph, start, goal):
    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()

        if node not in visited:
            print(node, end=" ")

            if node == goal:
                return True

            visited.add(node)

            for neighbour in reversed(graph[node]):
                if neighbour not in visited:
                    stack.append(neighbour)

    return False

start_node = 'A'
goal_node = 'H'

print("Breadth First Search:")
start_time = time.perf_counter()

bfs_result = bfs(graph, start_node, goal_node)

bfs_time = time.perf_counter() - start_time

print("\nGoal Found:", bfs_result)
print("Execution Time:", bfs_time)

print("\n----------------------------")

print("Iterative Depth First Search:")
start_time = time.perf_counter()

dfs_result = iterative_dfs(graph, start_node, goal_node)

dfs_time = time.perf_counter() - start_time

print("\nGoal Found:", dfs_result)
print("Execution Time:", dfs_time)

print("\n----------------------------")

print("Performance Comparison")

if bfs_time < dfs_time:
    print("BFS executed faster.")
elif dfs_time < bfs_time:
    print("Iterative DFS executed faster.")
else:
    print("Both executed in approximately the same time.")



    #2
    from collections import deque

graph = {
    "Start": ["N1", "N2"],
    "N1": ["DeadEnd1", "DeadEnd2"],
    "N2": ["N3"],
    "N3": ["N4", "DeadEnd3"],
    "N4": ["Goal"],
    "DeadEnd1": [],
    "DeadEnd2": [],
    "DeadEnd3": [],
    "Goal": []
}

def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set()

    print("BFS Exploration Order:")

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node in visited:
            continue

        print("Exploring node:", node)
        visited.add(node)

        if node == goal:
            return path

        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.append(path + [neighbour])

    return None

def depth_limited_search(graph, node, goal, limit, path):
    if node == goal:
        return path

    if limit == 0:
        return None

    for neighbour in graph[node]:
        if neighbour not in path:
            result = depth_limited_search(
                graph,
                neighbour,
                goal,
                limit - 1,
                path + [neighbour]
            )

            if result is not None:
                return result

    return None

def iddfs(graph, start, goal, max_depth):
    print("\nIDDFS Iterations:")

    for depth in range(max_depth + 1):
        print("Searching with depth limit:", depth)

        result = depth_limited_search(
            graph,
            start,
            goal,
            depth,
            [start]
        )

        if result is not None:
            return result

    return None

print("Hariprasad Vishwakarma T127")
print()

bfs_path = bfs(graph, "Start", "Goal")
print("\nBFS Result Path:", bfs_path)

iddfs_path = iddfs(graph, "Start", "Goal", 10)
print("\nIDDFS Result Path:", iddfs_path)
