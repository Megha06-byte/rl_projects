def euclidean_dist(node_a, node_b):
    return np.linalg.norm(np.array(node_a) - np.array(node_b))

def line_collision(node_a, node_b, obstacle_mask, x_min, x_max, y_min, y_max):
    steps = int(np.ceil(np.linalg.norm(np.array(node_a) - np.array(node_b)) * 10))
    for i in range(steps + 1):
        t = i / steps
        x = node_a[0] * (1 - t) + node_b[0] * t
        y = node_a[1] * (1 - t) + node_b[1] * t
        px, py = world_to_pixel(x, y, x_min, x_max, y_min, y_max, obstacle_mask.shape[1], obstacle_mask.shape[0])
        px = np.clip(px, 0, obstacle_mask.shape[1] - 1)
        py = np.clip(py, 0, obstacle_mask.shape[0] - 1)
        if obstacle_mask[py, px] > 0:
            return False
    return True
def rrt_star(start, goal, obstacle_mask, x_min, x_max, y_min, y_max, d_step=1.0, epsilon=1.0, max_iters=2000, radius=2.0):
    tree = [start]
    parent = {start: None}
    cost = {start: 0.0}  #track cost to come for each node
    for _ in range(max_iters):
        curr_node = (np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))
        tree_node = min(tree, key=lambda node: euclidean_dist(node, curr_node))
        direction = np.array(curr_node) - np.array(tree_node)
        if np.linalg.norm(direction) == 0:
            continue
        direction = direction / np.linalg.norm(direction)
        new_node = tuple(np.array(tree_node) + d_step * direction)
        if not line_collision(tree_node, new_node, obstacle_mask, x_min, x_max, y_min, y_max):
            continue


        # Find neighbors within radius (instead of just nearest)
        neighbors = [node for node in tree if euclidean_dist(node, new_node) <= radius]

        # Choose the best parent (lowest cost)
        min_cost = cost[tree_node] + euclidean_dist(tree_node, new_node) #initialize optimal cost as going through current chosen tree node
        best_parent = tree_node
        #search among neighbours for better parent nodes
        for node in neighbors:
            if line_collision(node, new_node, obstacle_mask, x_min, x_max, y_min, y_max): 
                c = cost[node] + euclidean_dist(node, new_node)  #calculate cost of reaching new node via a neighbour node within radius r
                if c < min_cost:
                    min_cost = c
                    best_parent = node

        tree.append(new_node)
        parent[new_node] = best_parent
        cost[new_node] = min_cost

        # Rewire neighbors if going through new_node is better
        for node in neighbors:
            if node == best_parent:
                continue
            if line_collision(new_node, node, obstacle_mask, x_min, x_max, y_min, y_max):
                c = cost[new_node] + euclidean_dist(new_node, node)
                if c < cost[node]:
                    parent[node] = new_node
                    cost[node] = c

        if euclidean_dist(new_node, goal) < epsilon:
            goal_node = min(tree, key=lambda node: euclidean_dist(node, goal))  
            path = [goal_node]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path, tree
    return None, tree
path, tree = rrt_star(
    start, goal,
    obstacle_mask,
    x_min, x_max, y_min, y_max,
    d_step=0.5, epsilon=0.5, max_iters=2000, radius=2.0
)

if path:
    print("Path found!")
else:
    print("No path found.")

