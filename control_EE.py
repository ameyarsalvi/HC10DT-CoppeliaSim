import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Connect to CoppeliaSim
client = RemoteAPIClient('localhost', 23004)  # Adjust port if necessary
sim = client.getObject('sim')

# Get joint handles
joint_names = ["/base_link_respondable/joint_1_s", "/base_link_respondable/joint_2_l", "/base_link_respondable/joint_3_u", "/base_link_respondable/joint_4_r", "/base_link_respondable/joint_5_b", "/base_link_respondable/joint_6_t"]
joints = [sim.getObject(joint) for joint in joint_names]

# Get end-effector (EE) handle
ee_handle = sim.getObject('/base_link_respondable/EE')

# Start simulation
client.setStepping(True)
sim.startSimulation()

# Run simulation loop for 20 seconds
start_time = sim.getSimulationTime()
while (t := sim.getSimulationTime()) - start_time < 20:
    
    # Define manual velocities for each joint (rad/s)
    manual_velocities = [0.3, -0.2, 0.1, -0.4, 0.5, -0.3]  # Modify as needed

    # Apply the velocities to the joints
    for joint, vel in zip(joints, manual_velocities):
        sim.setJointTargetVelocity(joint, vel)


    # Get EE position
    ee_position = sim.getObjectPose(ee_handle, sim.handle_world)
    print(f"Time: {t:.2f} sec | EE Position: {ee_position}")

    # Step simulation
    client.step()
    #time.sleep(0.05)  # Adjust if needed

# Stop simulation
sim.stopSimulation()

print("Simulation complete.")
