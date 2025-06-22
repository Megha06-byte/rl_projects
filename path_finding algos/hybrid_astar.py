import numpy as np

def deg2rad(deg):
    return deg * np.pi / 180.0

def heuristic(state, goal):
    # Euclidean distance
    x, y, _ = state
    gx, gy, _ = goal
    return np.hypot(gx - x, gy - y)

def simulate_motion(state, steering_angle, d_step, wheelbase):
    x, y, theta = state
    x_new = x + d_step * np.cos(theta)
    y_new = y + d_step * np.sin(theta)
    theta_new = theta + d_step / wheelbase * np.tan(steering_angle)
    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))  # Normalize
    return (round(x_new, 2), round(y_new, 2), round(theta_new, 2))

def is_within_bounds(state, x_min, x_max, y_min, y_max):
    x, y, _ = state
    return x_min <= x <= x_max and y_min <= y <= y_max

def collision_along_path(state1, state2, obstacle_mask, x_min, x_max, y_min, y_max):
    x1, y1, _ = state1
    x2, y2, _ = state2
    steps = int(np.ceil(np.hypot(x2 - x1, y2 - y1) * 10))
    for i in range(steps + 1):
        t = i / steps
        x = x1 * (1 - t) + x2 * t
        y = y1 * (1 - t) + y2 * t
        px, py = world_to_pixel(x, y, x_min, x_max, y_min, y_max, obstacle_mask.shape[1], obstacle_mask.shape[0])
        px = np.clip(px, 0, obstacle_mask.shape[1] - 1)
        py = np.clip(py, 0, obstacle_mask.shape[0] - 1)
        if obstacle_mask[py, px] > 0:
            return True
    return False

def cost(state1, state2):
    x1, y1, _ = state1
    x2, y2, _ = state2
    return np.hypot(x2 - x1, y2 - y1)

def world_to_pixel(x, y, x_min, x_max, y_min, y_max, width, height):
    px = int((x - x_min) / (x_max - x_min) * (width - 1))
    py = int((y - y_min) / (y_max - y_min) * (height - 1))
    return px, py

def is_goal(state, goal, epsilon=0.5):
    x, y, theta = state
    gx, gy, gtheta = goal
    return np.hypot(gx - x, gy - y) < epsilon and abs(np.arctan2(np.sin(gtheta-theta), np.cos(gtheta-theta))) < deg2rad(15)

def hybrid_astar(start, goal, obstacle_mask, x_min, x_max, y_min, y_max, d_step=0.5, epsilon=0.5, max_iters=1000, radius=2.0, max_steer=deg2rad(30), wheelbase=2.0):
    open_list = set()
    closed_list = set()
    g = {}
    f = {}
    parent = {}

    g[start] = 0
    f[start] = heuristic(start, goal)
    open_list.add(start)

    for _ in range(max_iters):
        if not open_list:
            break
        curr_state = min(open_list, key=lambda node: f.get(node, float('inf')))
        if is_goal(curr_state, goal, epsilon):
            # Reconstruct path
            path = []
            while curr_state in parent:
                path.append(curr_state)
                curr_state = parent[curr_state]
            path.append(start)
            path.reverse()
            return path
        open_list.remove(curr_state)
        closed_list.add(curr_state)

        for steering_angle in [-max_steer, 0, +max_steer]:
            next_state = simulate_motion(curr_state, steering_angle, d_step, wheelbase)
            if not is_within_bounds(next_state, x_min, x_max, y_min, y_max):
                continue
            if collision_along_path(curr_state, next_state, obstacle_mask, x_min, x_max, y_min, y_max):
                continue
            if next_state in closed_list:
                continue

            tentative_g = g[curr_state] + cost(curr_state, next_state)
            if next_state not in open_list or tentative_g < g.get(next_state, float('inf')):
                parent[next_state] = curr_state
                g[next_state] = tentative_g
                f[next_state] = g[next_state] + heuristic(next_state, goal)
                open_list.add(next_state)
    return None  # No path found
path, tree = hybrid_astar(
    start, goal,
    obstacle_mask,
    x_min, x_max, y_min, y_max,
    d_step=0.5, epsilon=0.5, max_iters=2000, radius=2.0
)

if path:
    print("Path found!")
else:
    print("No path found.")
