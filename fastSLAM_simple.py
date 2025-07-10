import numpy as np
class Particle:
    def __init__(self, pose, num_landmarks):
        self.pose = pose # [x,y, theta]
        self.landmarks = [None] * num_landmarks # List of landmarks
        self.landmark_cov = [None] * num_landmarks # List of covariance matrices for each landmark
        self.weight = 1.0 # Initial weight of the particle

class FastSLAM:
    #chosen bcoz of computational efficiency by considering pose and landmark to be conditionally independent 
    def __init__(self, num_particles, num_landmarks, motion_noise, obs_noise):
        self.num_particles = num_particles
        self.num_landmarks = num_landmarks
        self.particles = [Particle(np.zeros(3), num_landmarks) for _ in range(num_particles)] #initialize trajectory set
        self.motion_noise = motion_noise
        self.obs_noise = obs_noise
    
    def motion_model(self, pose, control, dt):
        x, y, theta = pose
        v, w = control
        theta += w * dt + np.random.randn()*np.sqrt(self.motion_noise[2,2])
        x += v * np.cos(theta) * dt + np.random.randn() * np.sqrt(self.motion_noise[0,0])
        y += v * np.sin(theta) * dt + np.random.randn() * np.sqrt(self.motion_noise[1,1])
        return np.array([x, y, theta])
    
    def observe_landmark(self, pose, landmark_pos):
        dx = landmark_pos[0] - pose[0]
        dy = landmark_pos[1] - pose[1]
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx) - pose[2]
        return np.array([r, phi]) 

   #landmark update function using different sensor data merged using EKF (Extended Kalman Filter)
    def update_landmark(self, particle, landmark_ids, observations):
        for obs, lm_id in zip(observations, landmark_ids):
            if particle.landmarks[lm_id] is None:
                # Initialize landmark if it does not exist  
                r, phi = obs
                lx = particle.pose[0] + r * np.cos(phi + particle.pose[2])
                ly = particle.pose[1] + r * np.sin(phi + particle.pose[2])
                particle.landmarks[lm_id] = np.array([lx, ly])
                particle.landmark_cov[lm_id] = self.obs_noise.copy()
                
            else:
                # EKF update
                lm_pos = particle.landmarks[lm_id]
                px, py, theta = particle.pose
                dx = lm_pos[0] - px
                dy = lm_pos[1] - py
                r = np.hypot(dx, dy)
                phi = np.arctan2(dy, dx) - theta
                H = np.array([[dx/r, dy/r],
                                [-dy/(r**2), dx/(r**2)]]) #Jacobian of the observation model
                S = H @ particle.landmark_cov[lm_id] @ H.T + self.obs_noise #residual covariance matrix
                K = particle.landmark_cov[lm_id] @ H.T @ np.linalg.inv(S) #Kalman gain
                
                innovation = obs - self.observe_landmark(particle.pose, lm_pos)
                lm_pos += K @ innovation
                particle.landmarks[lm_id] = lm_pos
                particle.landmark_cov[lm_id] = (np.eye(2) - K @ H) @ particle.landmark_cov[lm_id]
        
    def compute_weight(self, particle, observations, landmark_ids):
        weight = 1.0
        for obs, lm_id in zip(observations, landmark_ids):
            if particle.landmarks[lm_id] is not None:
                pred = self.observe_landmark(particle.pose, particle.landmarks[lm_id])
                error = obs - pred
                error[1] = (error[1] + np.pi) % (2 * np.pi) - np.pi
                cov = particle.landmark_cov[lm_id]
                det = np.linalg.det(cov)
                inv = np.linalg.inv(cov)
                weight *= np.exp(-0.5 * error.T @ inv @ error) / (2 * np.pi * np.sqrt(det)) #mahalanobis distance
        return weight
        
    def resample_particles(self):
        weights = np.array([particle.weight for particle in self.particles])
        weights /= np.sum(weights)  
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        new_particles = []
        for idx in indices:
            p = self.particles[idx]
            new_p = Particle(p.pose.copy(), len(p.landmarks))
            new_p.landmarks = [lm.copy() if lm is not None else None for lm in p.landmarks]
            new_p.landmark_cov = [cov.copy() if cov is not None else None for cov in p.landmark_cov]
            new_particles.append(new_p)
        self.particles = new_particles

    def estimate(self):
        return np.mean([p.pose for p in self.particles], axis=0) #taking mean over all particles for pose estimation




import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from geometry_msgs.msg import PoseStamped

class FastSLAMNode(Node):
    def __init__(self):
        super().__init__('fast_slam_node')
        self.fastslam = FastSLAM(num_particles=100, num_landmarks=10, motion_noise=np.diag([0.1, 0.1, 0.01]), obs_noise=np.diag([0.5, 0.1]))
        self.pose_pub = self.create_publisher(PoseStamped, '/fastslam/pose', 10)
        self.last_odom = None
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

    def odom_callback(self, msg):
        # Process odometry data
        # the odometry message contains a pose and twist
        self.last_odom = msg

    def scan_callback(self, msg):
        if self.last_odom is None:
            return
        # Process laser scan data
        observations, landmark_ids = extract_landmarks_from_scan(msg)
        # Use odometry to get control input (v, w)
        v = self.last_odom.twist.twist.linear.x
        w = self.last_odom.twist.twist.angular.z
        dt = 0.1  # Or compute from timestamps

        # FastSLAM steps
        for p in self.fastslam.particles:
            p.pose = self.fastslam.motion_model(p.pose, [v, w], dt) #update pose using motion model
            self.fastslam.update_landmark(p, observations, landmark_ids)
            p.weight = self.fastslam.compute_weight(p, observations, landmark_ids)
        self.fastslam.resample()

        # Publish estimated pose
        est_pose = self.fastslam.estimate()
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = est_pose[0]
        pose_msg.pose.position.y = est_pose[1]
        pose_msg.pose.orientation.z = np.sin(est_pose[2] / 2)
        pose_msg.pose.orientation.w = np.cos(est_pose[2] / 2)
        self.pose_pub.publish(pose_msg)

import numpy as np

def extract_landmarks_from_scan(scan_msg, distance_threshold=0.3, min_cluster_size=3):
    """
    Extracts landmarks from a LaserScan by clustering points that are close together.
    Returns:
        observations: list of (range, bearing) tuples
        landmark_ids: list of unique ids for each landmark
    """
    ranges = np.array(scan_msg.ranges)
    angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
    clusters = []
    current_cluster = []

    for i in range(len(ranges)):
        if np.isinf(ranges[i]) or np.isnan(ranges[i]):
            continue
        if not current_cluster:
            current_cluster.append(i)
        else:
            prev = current_cluster[-1]
            # Compute Euclidean distance between consecutive points
            dx = ranges[i]*np.cos(angles[i]) - ranges[prev]*np.cos(angles[prev])
            dy = ranges[i]*np.sin(angles[i]) - ranges[prev]*np.sin(angles[prev])
            dist = np.hypot(dx, dy)
            if dist < distance_threshold:
                current_cluster.append(i)
            else:
                if len(current_cluster) >= min_cluster_size:
                    clusters.append(current_cluster)
                current_cluster = [i]
    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    observations = []
    landmark_ids = []
    for idx, cluster in enumerate(clusters):
        # Compute centroid in polar coordinates
        cluster_ranges = ranges[cluster]
        cluster_angles = angles[cluster]
        mean_range = np.mean(cluster_ranges)
        mean_angle = np.mean(cluster_angles)
        observations.append([mean_range, mean_angle])
        landmark_ids.append(idx)  # assign a unique id per cluster

    return observations, landmark_ids


def main(args=None):
    rclpy.init(args=args)
    node = FastSLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
