from ur5py import UR5Robot
import numpy as np
import time
from autolab_core import RigidTransform
from autolab_core.transformations import quaternion_from_euler, quaternion_matrix, euler_matrix, quaternion_from_matrix, translation_from_matrix

def convert_pose_to_rig(pose):
    # assume pose is xyz + q_xyzw
    # convert to rigid transform
    trans = pose[:3]
    angles_quat = pose[3:] # q_xyzw

    #Flip quat to wxyz
    angles_quat_wxyz = np.concatenate((angles_quat[[-1]], angles_quat[:-1]))
    #Get rotation matrix (3x3)
    rot = RigidTransform.rotation_from_quaternion(angles_quat_wxyz)
    rigid = RigidTransform(
            rotation=rot,
            translation=trans,
            from_frame="tcp",
            to_frame="tcp",
        )
    return rigid

if __name__ == "__main__":
    ur = UR5Robot(gripper=2)
    ur.set_tcp(RigidTransform(translation=[0,0.0,0], rotation=RigidTransform.z_axis_rotation(-np.pi/2)))
    # joints_original = ur.get_pose(convert=False)
    # print(joints_original)
    # joints_original = ur.get_pose(convert=True)
    # print(joints_original)
    # # breakpoint()

    # # Testing nonblocking
    # joints_original = ur.get_pose(convert=False)
    # joints_goal = joints_original.copy() 
    # joints_goal[1] += 0.2
    
    # # ur.gripper.set_pos(128)
    # # Start moving to goal
    # runtime = 5 # seconds
    # joint_targets = np.linspace(joints_original, joints_goal, int(5/0.002), True)
    # start_time = time.time()
    # for j in joint_targets:
    #     ur.servo_pose(j, convert=False, gain=400)
    #     ur.gripper.close()
    #     # print("here")
    #     print(ur.gripper.get_pos())
    #     time.sleep(0.002)
    #     if time.time()-start_time > 2:
    #         break
    
    # ur.gripper.open()
    # # Give new goal
    # joint_targets = np.linspace(ur.get_pose(convert=False), joints_original, int(2/0.002), True)
    # start_time = time.time()
    # for j in joint_targets:
    #     ur.servo_pose(j, convert=False, gain=400)
    #     # print(ur.gripper.get_pos())
    #     time.sleep(0.002)
        
    pose = np.array([0.56343067,  0.00338203,  0.25930977,  3.13144658,  0.01786906, -0.00901092])
    new_pose = RigidTransform(translation=pose[:3], rotation=euler_matrix(pose[3], pose[4], pose[5], axes="ryxz")[:3, :3], from_frame="tcp", to_frame="tcp")
    print(new_pose)
    ur.move_pose(new_pose, vel=1, acc=5)
    
    
    
    
