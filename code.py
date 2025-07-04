import heapq
import matplotlib.pyplot as plt
import numpy as np
import random


GRID_SIZE = 10
OBSTACLES = {
    (3, 3), (3, 4), (3, 5),
    (4, 5), (5, 5), (6, 5)
}
start = (0, 0)
goal = (9, 9)


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

# Heuristic 1: Manhattan Distance 
def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

# Heuristic 2: Noisy 
def noisy_heuristic(pos, goal):
    return manhattan(pos, goal) + random.uniform(-1, 2)

# Heuristic 3: Biased 
def biased_heuristic(pos, goal):
    dx = abs(pos[0] - goal[0])
    dy = abs(pos[1] - goal[1])
    return dx + dy + (1 if dx < dy else -1)

# --------------------------
# A* Algorithm
# --------------------------
def a_star(start, goal, heuristic_fn):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic_fn(start, goal), 0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored_nodes = []

    while open_list:
        _, cost, current = heapq.heappop(open_list)
        explored_nodes.append(current)

        if current == goal:
            break

        for dx, dy in DIRECTIONS:
            next_pos = (current[0] + dx, current[1] + dy)
            if is_valid(next_pos):
                new_cost = cost + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic_fn(next_pos, goal)
                    heapq.heappush(open_list, (priority, new_cost, next_pos))
                    came_from[next_pos] = current

    return came_from, explored_nodes


def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        if current not in came_from:
            return []  # Path not found
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


results = {}
for name, h_fn in {
    "Manhattan": manhattan,
    "Noisy": noisy_heuristic,
    "Biased": biased_heuristic
}.items():
    came_from, explored = a_star(start, goal, h_fn)
    path = reconstruct_path(came_from, start, goal)
    results[name] = {"path": path, "explored": explored}


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (name, result) in zip(axes, results.items()):
    ax.set_title(f"{name} Heuristic")
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True)

    # Draw Obstacles
    for obs in OBSTACLES:
        ax.add_patch(plt.Rectangle(obs, 1, 1, color="black"))

    # Draw Explored Nodes
    for node in result["explored"]:
        ax.add_patch(plt.Circle((node[0]+0.5, node[1]+0.5), 0.2, color="lightblue"))

    # Draw Final Path
    for node in result["path"]:
        ax.add_patch(plt.Circle((node[0]+0.5, node[1]+0.5), 0.3, color="green"))

    # Draw Start and Goal
    ax.add_patch(plt.Circle((start[0]+0.5, start[1]+0.5), 0.3, color="blue"))
    ax.add_patch(plt.Circle((goal[0]+0.5, goal[1]+0.5), 0.3, color="red"))

plt.tight_layout()
plt.show()
