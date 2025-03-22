from typing import List
import rtde_control
import rtde_receive
import rtde_io
from ur5py.robotiq_gripper_control import RobotiqGripper
from ur5py.socket_robotiq import SocketRobotiq
from ur5py.continuous_state_robotiq_gripper import ContinuousStateRobotiqGripper
import time
import numpy as np
import pdb
from autolab_core import RigidTransform


def RT2UR(rt: RigidTransform):
    """
    converts from rigidtransform pose to the UR format pose (x,y,z,rx,ry,rz)
    """
    pos = rt.translation.tolist() + rt.axis_angle.tolist()
    return pos


def UR2RT(pose: List):
    """
    converts UR format pose to RigidTransform
    """
    m = RigidTransform.rotation_from_axis_angle(pose[-3:])
    return RigidTransform(translation=pose[:3], rotation=m)


class UR5Robot:
    def __init__(
        self, ip="172.22.22.3", gripper: [bool, int] = False, gripper_port=63352
    ):
        gripper = int(gripper)
        self.ip = ip
        self.ur_c = rtde_control.RTDEControlInterface(ip)
        self.ur_r = rtde_receive.RTDEReceiveInterface(ip)
        self.ur_io = rtde_io.RTDEIOInterface(ip)
        if gripper == 1:
            self.gripper = RobotiqGripper(self.ur_c)
        elif gripper == 2:
            self.gripper = SocketRobotiq(ip, gripper_port)
        elif gripper == 3:
            self.gripper = ContinuousStateRobotiqGripper()
            self.gripper.connect(self.ip, 63352)

    def servo_joint(self, target, time=0.002, lookahead_time=0.1, gain=300):
        """
        Servoj can be used for online realtime control of joint positions.
        It is designed for movements over greater distances.
        time: time where the command is controlling the robot. The function is blocking for time t [S]
        lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time. A low value gives fast reaction, a high value prevents overshoot.
        gain: proportional gain for following target position, range [100,2000]. The higher the gain, the faster reaction the robot will have.
        """
        self.ur_c.servoJ(target, 0, 0, time, lookahead_time, gain)

    def servo_pose(
        self,
        target,
        time=0.002,
        lookahead_time=0.1,
        gain=300,
        convert=True,
    ):
        """
        target in Rigidtransform being converted to: x,y,z,rx,ry,rz
        """
        if convert:
            target = RT2UR(target)
        self.ur_c.servoL(target, 0, 0, time, lookahead_time, gain)

    def move_joint(self, target, interp="joint", vel=1.0, acc=1, asyn=False):
        """
        the movej() command offers you a complete trajectory planning with acceleration, de-acceleration etc.
        It is designed for movements over greater distances.
        """
        if interp == "joint":
            self.ur_c.moveJ(target, vel, acc, asyn)
        elif interp == "tcp":
            self.ur_c.moveL_FK(target, vel, acc, asyn)
        else:
            raise KeyError("interpolation muct be in joint or tcp space")

    def move_pose(
        self,
        target: [RigidTransform, List],
        interp="joint",
        vel=0.1,
        acc=1,
        asyn=False,
        convert=True,
    ):
        """
        target in Rigidtransform being converted to: x,y,z,rx,ry,rz
        """
        if convert:
            pos = RT2UR(target)
        else:
            pos = target
        if interp == "joint":
            self.ur_c.moveJ_IK(pos, vel, acc, asyn)
        elif interp == "tcp":
            self.ur_c.moveL(pos, vel, acc, asyn)
        else:
            raise KeyError("interpolation muct be in joint or tcp space")

    def speed_tcp(self, vel, acc=10, t=0):
        """
        Accelerate linearly in tcp space and continue with constant tcp speed.
        """
        self.ur_c.speedL(vel, acc, time=t)

    def speed_joint(self, vel, acc=10):
        """
        Accelerate linearly in joint space and continue with constant joint speed.
        """
        self.ur_c.speedJ(vel, acc)

    # TODO verify the below two are correct
    def move_joint_path(
        self, waypoints: List[np.ndarray], vels=None, accs=None, blends=None, asyn=False
    ):
        """
        waypoints input should be N*6 for joint
        The size of the blend radius is per default a shared value for all the waypoint.
        A smaller value will make the path turn sharper whereas a higher value will make the path smoother.
        """
        assert isinstance(waypoints, np.ndarray), "waypoints must be a numpy array"
        assert waypoints.shape[1] == 6, "dimension of waypoints must be Nx6"
        if vels is None:
            vels = np.ones(waypoints.shape[0]).reshape(-1, 1) * 0.1
        elif type(vels) in [float, int]:
            vels = np.ones(waypoints.shape[0]).reshape(-1, 1) * vels

        if accs is None:
            accs = np.ones(waypoints.shape[0]).reshape(-1, 1) * 0.1
        elif type(accs) in [float, int]:
            accs = np.ones(waypoints.shape[0]).reshape(-1, 1) * accs

        if blends is None:
            blends = np.ones(waypoints.shape[0]).reshape(-1, 1)
        elif type(blends) in [float, int]:
            blends = np.ones(waypoints.shape[0]).reshape(-1, 1) * blends

        path = np.hstack(
            (
                waypoints,
                np.array(vels),
                np.array(accs),
                np.array(blends),
            )
        )
        self.ur_c.moveJ(path, asyn)

    def move_tcp_path(
        self,
        waypoints: List[RigidTransform],
        vels: List[float],
        accs: List[float],
        blends: List[float],
        asyn=False,
    ):
        """
        waypoints input should be N*6 for joint
        The size of the blend radius is per default a shared value for all the waypoint.
        A smaller value will make the path turn sharper whereas a higher value will make the path smoother.
        """
        assert len(vels) == len(waypoints)
        assert len(accs) == len(waypoints)
        waypoints = [RT2UR(p) for p in waypoints]
        path = np.hstack(
            (
                waypoints,
                np.array(vels)[..., None],
                np.array(accs)[..., None],
                np.array(blends)[..., None],
            )
        )
        if vels is None:
            vels = np.ones(waypoints.shape[0]) * 0.1
        elif type(vels) is float:
            vels = np.ones(waypoints.shape[0]) * vels

        if accs is None:
            accs = np.ones(waypoints.shape[0]) * 0.1
        elif type(accs) is float:
            accs = np.ones(waypoints.shape[0]) * accs

        if blends is None:
            blends = np.ones(waypoints.shape[0])
        elif type(blends) is float:
            blends = np.ones(waypoints.shape[0]) * blends
        self.ur_c.moveL(path, asyn)

    def set_tcp(self, tcp: [RigidTransform, List], convert=True):
        if convert:
            self.ur_c.setTcp(RT2UR(tcp))
        else:
            self.ur_c.setTcp(tcp)

    def get_joints(self):
        q = self.ur_r.getActualQ()
        return q

    def get_pose(self, convert=True):
        p = self.ur_r.getActualTCPPose()
        if convert:
            return UR2RT(p)
        else:
            return p

    def is_stopped(self):
        return self.ur_c.isSteady()

    def move_until_contact(
        self, vel, thres, acc=0.25, direction=np.array((0, 0, 1, 0, 0, 0))
    ):
        assert len(vel) == 6, "dimension of vel mush be 6"
        self.ur_c.speedL(vel, acceleration=acc)
        time.sleep(0.5)
        startforce = np.array(self.ur_r.getActualTCPForce())
        while True:
            force = np.array(self.ur_r.getActualTCPForce())
            if np.linalg.norm((startforce - force).dot(direction)) > thres:
                break
            time.sleep(0.008)
        self.ur_c.speedStop()

    def move_linear(self, start, end, vel, thres=17, acc=0.25):
        assert (
            len(start) == 3 and len(end) == 3
        ), "dimension of start and goal must be 3"
        assert (
            len(vel) == 3
        ), "vel is for x,y,z direction and the first nonzero one would be used"

        if abs(end[0] - start[0]) > 0.01:
            direc_x = vel[0]
            direc_y = (
                np.clip((end[1] - start[1]) / (end[0] - start[0] + 1e-6), -10, 10)
                * direc_x
            )
            direc_z = (
                np.clip((end[2] - start[2]) / (end[0] - start[0] + 1e-6), -10, 10)
                * direc_x
            )
        elif abs(end[1] - start[1]) > 0.01:
            direc_y = vel[1]
            direc_x = (
                np.clip((end[0] - start[0]) / (end[1] - start[1] + 1e-6), -10, 10)
                * direc_y
            )
            direc_z = (
                np.clip((end[2] - start[2]) / (end[1] - start[1] + 1e-6), -10, 10)
                * direc_y
            )
        else:
            direc_z = vel[2]
            direc_x = (
                np.clip((end[0] - start[0]) / (end[2] - start[2] + 1e-6), -10, 10)
                * direc_z
            )
            direc_y = (
                np.clip((end[1] - start[1]) / (end[2] - start[2] + 1e-6), -10, 10)
                * direc_z
            )

        dire = [direc_x, direc_y, direc_z, 0, 0, 0]
        self.ur_c.speedL(dire, accelaration=acc)
        time.sleep(0.5)
        startforce = np.array(self.ur_r.getActualTCPForce())
        while True:
            force = np.array(self.ur_r.getActualTCPForce())
            if (
                np.linalg.norm((startforce - force).dot(dire)) > thres
                or np.linalg.norm(self.get_pose().translation - end) < 0.01
            ):
                break
            time.sleep(0.008)
        self.ur_c.speedStop()

    def start_teach(self):
        self.ur_c.teachMode()

    def stop_teach(self):
        self.ur_c.endTeachMode()

    def start_freedrive(self, free_axes=[1, 1, 1, 1, 1, 1], feature=[0, 0, 0, 0, 0, 0]):
        return self.ur_c.freedriveMode(free_axes, feature)

    def stop_freedrive(self):
        return self.ur_c.endfreedriveMode()

    def force_mode(
        self,
        task_frame,
        selection_vector,
        wrench,
        type,
        limits,
        damping=0.005,
    ):
        return self.ur_c.forceMode(
            task_frame, selection_vector, wrench, type, limits
        ) and self.ur_c.forceModeSetDamping(damping)

    def force_mode_set_damping(self, damping: float):
        return self.ur_c.forceModeSetDamping(damping)

    def end_force_mode(self):
        return self.ur_c.forceModeStop()

    def jog_mode(self, speeds, feature, custom_frame={}):
        return self.ur_c.jogStart(speeds, feature, custom_frame)

    def end_jog_mode(self):
        return self.ur_c.jogStop()

    def stop_joint(self, a: float = 10.0):
        self.ur_c.servoStop(a)

    def stop_tool(self, a: float = 10.0, asynchronous: bool = False):
        self.ur_c.stopL(a, asynchronous)

    def get_current_force(self):
        return self.ur_r.getActualTCPForce()

    def change_gripper(self, gripper_type: int):
        """
        Changes the type of gripper

        Type 1 uses robotiq preamble
        Type 2 uses socket connection
        """
        if gripper_type == 1:
            self.gripper = RobotiqGripper(self.ur_c)
        elif gripper_type == 2:
            self.gripper = SocketRobotiq(self.ip)
        else:
            self.gripper = None

    def kill(self):
        self.ur_c.disconnect()

    def set_playload(self, mass: float, center_of_gravity: List = []):
        self.ur_c.setPayload(mass, center_of_gravity)

    def get_fk(self, joint_angles):
        self.ur_c.getForwardKinematics(joint_angles, [])


# if __name__ == "__main__":
#     ur = UR5Robot()
#     ur.set_tcp(RigidTransform(translation=[0, 0.0, 0.07]))
#     start_pose = ur.get_pose()
#     wps, vels, accs, blends = [], [], [], []
#     for dx in [-0.1, 0.1, 0]:
#         newpose = start_pose.copy()
#         newpose.translation[0] += dx
#         wps.append(newpose)
#         vels.append(0.3)
#         accs.append(2)
#         blends.append(0.01)
#     ur.move_tcp_path(wps, vels, accs, blends)
#     print("Done")
