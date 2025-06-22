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

import heapq
#D* Lite implementation
class Dstar:
    def __init__(self, maze, start, goal):
        self.maze = maze.copy()
        self.start = start
        self.goal = goal
        self.h, self.w = maze.shape
        self.actions = [(1,0), (0,1), (-1,0), (0,-1)]
        
      
        self.g = {}  # Cost from start to goal 
        self.lcost = {}  # One-step lookahead cost, cost to reach goal if next best node is chosen
        self.priority = []  # Priority queue
        self.k_m = 0  # Key modifier for incremental search, to prioritise close-by nodes more can be considered as offset for start position
        
        # Initialize
        for i in range(self.h):
            for j in range(self.w):
                self.g[(i,j)] = float('inf')
                self.lcost[(i,j)] = float('inf')

        
        self.rhs[self.goal] = 0
        heapq.heappush(self.priority, (self.calculate_key(self.goal), self.goal))

    def manhattan(self, node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
    
    def calculate_key(self, node):
        return (min(self.g[node], self.lcost[node]+self.manhattan(self.start, node)+self.k_m), min(self.g[node], self.lcost[node])) 
    
    def is_valid(self, pos):
        r, c = pos
        return (0 <= r < self.h and 0 <= c < self.w and self.maze[r, c] != '#')
    
    def get_neighbors(self, node):
        neighbors = []
        for action in self.actions:
            next_node = (node[0] + action[0], node[1] + action[1])
            if self.is_valid(next_node):
                neighbors.append(next_node)
        return neighbors
    
    def cost(self, node1, node2):
        if not self.is_valid(node1) or not self.is_valid(node2):
            return float('inf')
        return 1
    
    def update_vertex(self, node):
        #update lcost value for a node
        if node!= self.goal:
            min_lcost = float('inf')
            for neighbor in self.get_neighbors(node):
                min_lcost = min(min_lcost, self.g[neighbor] + self.cost(node, neighbor))
            self.lcost[node] = min_lcost

        self.priority.remove((self.calculate_key(node), node)) #remove node from heap, as already visited
        heapq.heapify(self.priority)  # Re-heapify the priority queue
        if self.g[node] != self.lcost[node]:
            self.g[node] = self.lcost[node]
            heapq.heappush(self.priority, (self.calculate_key(node), node)) #add it back as it needs to be processed again; inconsistent node

    def compute_shortest_path(self):
        while (self.priority and (self.priority[0][0] < self.calculate_key(self.start)) or self.g[self.start] != self.lcost[self.start]):
            #while priority heap is not empty and the top element has a key less than the key of the start node or g[start] is not equal to lcost[start]
            #pop the top element from the priority queue
            current_key, current_node = heapq.heappop(self.priority)

            if current_key < self.calculate_key(current_node):
                # If the current node's key is less than its calculated key, we need to update it hence added it back to the priority queue
                heapq.heappush(self.priority, (self.calculate_key(current_node), current_node))
            elif self.g[current_node] > self.lcost[current_node]:
                # If g[current_node] is greater than lcost[current_node], underconsistent, but better path found
                self.g[current_node] = self.lcost[current_node] #update g value
                # Update the neighbors of the current node
                for neighbor in self.get_neighbors(current_node):
                    self.update_vertex(neighbor)
            else:   
                # If g[current_node] is equal to lcost[current_node], overconsistent, path is not realisable
                self.g[current_node] = float('inf')
                for neighbor in self.get_neighbors(current_node) + [current_node]:
                    self.update_vertex(neighbor)

    def get_path(self):
        path = []
        current_node = self.start
        while current_node != self.goal:
            path.append(current_node)
            next_node = None
            min_cost = float('inf')
            for neighbor in self.get_neighbors(current_node):
                if self.g[neighbor] < min_cost:
                    min_cost = self.g[neighbor]
                    next_node = neighbor
            if next_node is None:
                break
            current_node = next_node
        if current_node == self.goal:
            path.append(self.goal)
        return path
    def update_obstacles(self, new_maze, changed_cells):
        """D* incremental update when obstacles change"""
        old_start = self.start

        for cell in changed_cells:
            # Update maze
            self.maze[cell] = new_maze[cell]

            # Update affected vertices
            affected = [cell] + self.get_neighbors(cell)
            for s in affected:
                if s in self.g:  # Make sure vertex exists
                    self.update_vertex(s)

        # Update start position and key modifier
        self.k_m += self.manhattan(old_start, self.start)

        # Recompute shortest path incrementally
        self.compute_shortest_path()

    def plan(self):
            """Initial planning"""
            self.compute_shortest_path()
            return self.extract_path()

    def replan_incremental(self, new_maze, robot_pos, changed_cells):
            """Incremental replanning from current robot position"""
            self.start = robot_pos
            self.update_obstacles(new_maze, changed_cells)
            return self.extract_path()
dstar = Dstar(maze, start, goal)
dstar.compute_shortest_path()
path = dstar.extract_path()
print(path)
