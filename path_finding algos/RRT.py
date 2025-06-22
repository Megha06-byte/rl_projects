def euclidean_dist(node_a, node_b):
    return np.linalg.norm(np.array(node_a) - np.array(node_b))
def rrt(start, goal, obstacle_mask, x_min, x_max, y_min, y_max, d_step=1.0, epsilon=1.0, max_iters=2000):
    tree = [start]
    parent = {start: None}
    for _ in range(max_iters):
        # 1. Sample random node
        curr_node = (np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))

        # 2. Find nearest node in tree
        tree_node = min(tree, key=lambda node: euclidean_dist(node, curr_node))

        # 3. Steer: move from tree_node towards curr_node by d_step
        direction = np.array(curr_node) - np.array(tree_node)
        if np.linalg.norm(direction) == 0:
            continue
        direction = direction / np.linalg.norm(direction)
        new_node = tuple(np.array(tree_node) + d_step * direction)

        # 4. Check for collision using the mask
        if line_collision(tree_node, new_node, obstacle_mask, x_min, x_max, y_min, y_max):
            tree.append(new_node)
            parent[new_node] = tree_node

            # 5. Check if goal is reached
            if euclidean_dist(new_node, goal) < epsilon:
                # Build path from start to goal
                path = [new_node]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path, tree
    return None, tree  # No path found
# world is 0-10 x 0-10
x_min, x_max, y_min, y_max = 0, 10, 0, 10

# convert pixel positions to world coordinates
def pixel_to_world(px, py, x_min, x_max, y_min, y_max, width, height):
    x = x_min + (px / (width - 1)) * (x_max - x_min)
    y = y_min + (py / (height - 1)) * (y_max - y_min)
    return (x, y)

start_px = (118, 112)  # detected
goal_px = (515, 510)
start = pixel_to_world(start_px[0], start_px[1], x_min, x_max, y_min, y_max, width, height)
goal = pixel_to_world(goal_px[0], goal_px[1], x_min, x_max, y_min, y_max, width, height)

path, tree = rrt(
    start, goal,
    obstacle_mask,
    x_min, x_max, y_min, y_max,
    d_step=0.5, epsilon=0.5, max_iters=2000
)

if path:
    print("Path found:", path)
else:
    print("No path found.")
