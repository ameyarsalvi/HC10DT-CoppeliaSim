import numpy as np
from numpy.linalg import norm
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
import time
from scipy.spatial.transform import Rotation as R
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from stable_baselines3.common.env_checker import check_env

class coppTest():
    def __init__(self):        
        client = RemoteAPIClient()

        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()
        
        self.proximity = []

        # Get joint and EE handles
        joint_names = ["/base_link_respondable/joint_1_s", "/base_link_respondable/joint_2_l",
                       "/base_link_respondable/joint_3_u", "/base_link_respondable/joint_4_r",
                       "/base_link_respondable/joint_5_b", "/base_link_respondable/joint_6_t"]
        
        self.joints = [self.sim.getObject(joint) for joint in joint_names]
        self.ee_handle = self.sim.getObject('/base_link_respondable/EE')
        self.targFrame = self.sim.getObject('/TargetFrame')        
        
        self.generate_random_target_pose()  
        self.setTargFrame()      

    def step(self, action=[0, 0, 0, 0, 0, 0]):
        """
        Executes one step in the environment.
        """
        # Apply joint velocities
        for joint, vel in zip(self.joints, action):
            self.sim.setJointTargetVelocity(joint, float(vel))

        # Step simulation
        self.sim.step()

        # Get observation
        self.ee_position = self.sim.getObjectPose(self.ee_handle, self.sim.handle_world)

        # Convert quaternion to Euler angles
        self.ee_rotation = R.from_quat(self.ee_position[3:]).as_euler('xyz', degrees=False)

        observation = np.array([self.ee_position[0], self.ee_position[1], self.ee_position[2],
                                self.ee_rotation[0], self.ee_rotation[1], self.ee_rotation[2]], dtype=np.float32)

        # Compute reward
        reward = self.get_reward()

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        """
        super().reset(seed=seed)  # Ensure seed is properly set

        # Reset episode variables
        self.episode_length = 5000
        self.step_no = 0

        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate a new random target pose
        self.generate_random_target_pose()        

        # Stop and restart simulation
        self.sim.stopSimulation()

        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        
        self.sim.setStepping(True)
        self.sim.startSimulation()
        self.setTargFrame() 
        self.sim.step()

        # Get initial observation
        ee_position = self.sim.getObjectPose(self.ee_handle, self.sim.handle_world)
        ee_rotation = R.from_quat(ee_position[3:]).as_euler('xyz', degrees=False)

        observation = np.array([ee_position[0], ee_position[1], ee_position[2],
                                ee_rotation[0], ee_rotation[1], ee_rotation[2]], dtype=np.float32)

        return observation, {}

    def render(self):
        pass

    def setTargFrame(self):
        t = self.target_pose.tolist()

        # Move target frame in visualization        
        q = R.from_euler('xyz', t[3:6], degrees=False).as_quat()
        self.sim.setObjectPose(self.targFrame, [t[0], t[1], t[2], q[0], q[1], q[2], q[3]])                

    def generate_random_target_pose(self):
        """
        Generates a random target pose (x, y, z, roll, pitch, yaw) within reasonable bounds.
        """
        x = np.random.uniform(0.2, 1.0)   # Forward reach
        y = np.random.uniform(-0.8, 0.8)  # Sideways movement
        z = np.random.uniform(0.1, 1.2)   # Height
        
        roll = np.random.uniform(-np.pi/2, np.pi/2)
        pitch = np.random.uniform(-np.pi/2, np.pi/2)
        yaw = np.random.uniform(-np.pi, np.pi)
        	    
        self.target_pose = np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)

    def get_error(self):
        ee_position = np.array(self.ee_position[:3])
        ee_rotation = self.ee_rotation

        # Extract Target position and orientation
        target_position = np.array(self.target_pose[:3])
        target_rotation = np.array(self.target_pose[3:])

        # Compute Euclidean position error
        position_error = np.linalg.norm(ee_position - target_position)

        # Compute Euclidean orientation error
        orientation_error = np.linalg.norm(ee_rotation - target_rotation)

        print(position_error)
        print(orientation_error)

    def get_reward(self):
        """
        Computes a normalized reward based on the Euclidean distance between 
        the EE pose and the target pose.
        """
        # Extract EE position and convert quaternion to Euler angles
        # ee_position = np.array(ee_pose[:3])
        # ee_rotation = R.from_quat(ee_pose[3:]).as_euler('xyz', degrees=False)

        ee_position = np.array(self.ee_position[:3])
        ee_rotation = self.ee_rotation

        # Extract Target position and orientation
        target_position = np.array(self.target_pose[:3])
        target_rotation = np.array(self.target_pose[3:])

        # Compute Euclidean position error
        position_error = np.linalg.norm(ee_position - target_position)

        # Compute Euclidean orientation error
        orientation_error = 0 #np.linalg.norm(ee_rotation - target_rotation)

        # Define min/max bounds
        min_error = 0  # Best case: perfect match
        #max_error = 2.1 + 10.88  # Worst case: max position + max orientation error
        max_error = 2.4 + 18.84  # Worst case: max position + max orientation error

        # Compute total error
        total_error = position_error + orientation_error

        # Min-max normalize error (to [0,1]), avoiding divide-by-zero
        normalized_error = (total_error - min_error) / (max_error - min_error)
        timesteps_reward = -1 #(1 - (self.step_no)/(5000))**2

        # Reward: Higher is better, so use (1 - normalized_error)
        normalized_reward = (1 - normalized_error)**2  # 1 when perfect, 0 when worst
        # normalized_reward = -1*normalized_error**4 # 2
        #normalized_reward = -1*normalized_error**4 # 2

        #total_reward = normalized_reward + timesteps_reward

        self.proximity = total_error  # Store proximity metric
        
        if self.proximity < 0.001:
            total_reward = 10000  #timesteps_reward
        else:
            total_reward = normalized_reward #-1     #timesteps_reward

        return total_reward



# Validate the environment
#env = HC10DTCPEnv(port=23004, seed=1)
#check_env(env)
