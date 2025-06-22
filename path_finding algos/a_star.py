import numpy as np
import matplotlib.pyplot as plt

def generate_maze(width, height):
    # Initialize maze with walls everywhere
    maze = np.full((height, width), '#', dtype='U1')
    # Start carving from (1,1)
    stack = [(1, 1)]
    maze[1, 1] = ' '
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        x, y = stack[-1]
        # Shuffle directions
        np.random.shuffle(directions)
        moved = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width-1 and 0 < ny < height-1 and maze[ny, nx] == '#':
                # Carve out the path and the connecting cell
                maze[y + dy//2, x + dx//2] = ' '
                maze[ny, nx] = ' '
                stack.append((nx, ny))
                moved = True
                break
        if not moved:
            stack.pop()
    # Set start and end
    maze[1, 1] = 'S'
    maze[-2, -2] = 'E'
    return maze

# Generate the maze
maze = generate_maze(15, 15)

# Define color map
color_map = {
    '#': [0, 0, 0],    # black (wall)
    'S': [0, 1, 0],    # green (start)
    'E': [1, 0, 0],    # red (end)
    ' ': [1, 1, 1]     # white (path)
}

# Convert maze to image
img = np.zeros((*maze.shape, 3))
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        img[i, j] = color_map[maze[i, j]]

# Display the maze
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
plt.axis('off')
plt.show()

#heuristic function
def manhattan(node1, node2):
    # Calculate Manhattan distance between two nodes
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

actions = [(1,0), (0,1), (-1,0), (0,-1)]  # Down, Right, Up, Left
#Astar algorithm implementation
# This function implements the A* algorithm to find the shortest path in a maze
def astar(start, goal, maze):
    open_list = set()
    closed_list = set()
    g = {}         # Cost from start to node
    f = {}         # Total estimated cost
    h = {}         # Heuristic cost
    parent = {}    # For path reconstruction

    g[start] = 0
    h[start] = manhattan(start, goal)
    f[start] = h[start]
    open_list.add(start)

    while open_list:
        # Find node in open_list with lowest f value
        curr_node = min(open_list, key=lambda x: f.get(x, float('inf')))
        if curr_node == goal:
            # Reconstruct path
            path = []
            while curr_node in parent:
                path.append(curr_node)
                curr_node = parent[curr_node]
            path.append(start)
            path.reverse()
            return path

        open_list.remove(curr_node)
        closed_list.add(curr_node)

        for action in actions:
            next_node = (curr_node[0] + action[0], curr_node[1] + action[1])
            # Check bounds and wall
            if not (0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1]):
                continue
            if maze[next_node] == '#':
                continue
            if next_node in closed_list:
                continue

            tentative_g = g[curr_node] + 1  # Cost to move to neighbor

            if next_node not in open_list or tentative_g < g.get(next_node, float('inf')):
                parent[next_node] = curr_node
                g[next_node] = tentative_g
                h[next_node] = manhattan(next_node, goal)
                f[next_node] = g[next_node] + h[next_node]
                open_list.add(next_node)
    # If open_list is empty and goal not reached
    return None  # No path found
def find_symbol(maze, symbol):
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == symbol:
                return (i, j)
    return None

start = find_symbol(maze, 'S')
goal = find_symbol(maze, 'E')
print("Start:", start)
print("Goal:", goal)
path = astar(start, goal, maze)
print("Path from start to goal:", path)
