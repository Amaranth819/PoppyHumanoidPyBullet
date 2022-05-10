import pybullet as pb
import pybullet_data as pbd
import gym
import time
import numpy as np


class PoppyPyBulletEnv(gym.Env):
    def __init__(self, 
        use_gui = False,
        time_step = 1 / 240,
        control_mode = 'position',
        base_pos = [0.0, 0.0, 0.4125],
        base_ori = [0.0, 0.0, 0.0, 1.0]
    ) -> None:
        super(PoppyPyBulletEnv).__init__()

        # Initialize PyBullet
        self.use_gui = use_gui
        self.client_id = self._InitialPyBullet(use_gui)

        # Set PyBullet parameters
        pb.setTimeStep(time_step)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(pbd.getDataPath())
        self.plane_id = pb.loadURDF("plane.URDF")

        # Load the model files
        pb_control_mode = {'position' : pb.POSITION_CONTROL, 'velocity' : pb.VELOCITY_CONTROL, 'torque' : pb.TORQUE_CONTROL}
        self.control_mode = pb_control_mode[control_mode]

        robot_urdf_path = './fixed_base_poppy_humanoid.pybullet.urdf'
        self.robot_id, self.joint_name, self.joint_index, self.joint_type, self.joint_range, self.revolute_joints = self._LoadRobot(robot_urdf_path, base_pos = base_pos, base_ori = base_ori)
        self.num_joints = pb.getNumJoints(self.robot_id)

        self.control_panel = None
        self.ChangeDamping() # Inverse dynamics

        # !!!!!
        #disable the default velocity motors
        #and set some position control with small force to emulate joint friction/return to a rest pose
        jointFrictionForce = 1
        for joint in range(self.num_joints):
            pb.setJointMotorControl2(self.robot_id, joint, pb.POSITION_CONTROL, force=jointFrictionForce)
        self.StepSimulation()


    '''
        PyBullet settings
    '''
    def _InitialPyBullet(self, use_gui = False):
        client_id = pb.connect(pb.GUI if use_gui else pb.DIRECT)
        if use_gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        return client_id

    
    def close(self):
        pb.removeAllUserParameters()
        pb.close()


    '''
        Robot file
    '''
    def _LoadRobot(self, path, base_pos = [0.0, 0.0, 0.4125], base_ori = [0.0, 0.0, 0.0, 1.0]):
        robot_id = pb.loadURDF(path, base_pos, base_ori, useFixedBase = True, flags = 0)

        num_joints = pb.getNumJoints(robot_id)
        joint_name, joint_index, joint_type, joint_range = {}, {}, {}, {}
        revolute_joints = []

        for i in range(num_joints):
            info = pb.getJointInfo(robot_id, i)
            name = info[1].decode('UTF-8')
            joint_name[i] = name
            joint_index[name] = i
            joint_type[i] = info[2]
            joint_range[i] = (info[8], info[9])

            if info[2] == pb.JOINT_REVOLUTE and i >= 6:
                revolute_joints.append(i)
                pb.enableJointForceTorqueSensor(robot_id, i)
        
        return robot_id, joint_name, joint_index, joint_type, joint_range, revolute_joints


    '''
        Control
    '''
    def ApplyPositionControl(self, jpos_list):
        pb.setJointMotorControlArray(self.robot_id, self.revolute_joints, pb.POSITION_CONTROL, targetPositions = jpos_list)


    def ApplyVelocityControl(self, jvel_list):
        pb.setJointMotorControlArray(self.robot_id, self.revolute_joints, pb.VELOCITY_CONTROL, targetVelocities = jvel_list)


    def ApplyTorqueControl(self, jtorque_list):
        position_gains = [0.1] * len(jtorque_list)
        velocity_gains = [0.1] * len(jtorque_list)
        pb.setJointMotorControlArray(self.robot_id, self.revolute_joints, pb.TORQUE_CONTROL, forces = jtorque_list, positionGains = position_gains, velocityGains = velocity_gains)


    def AddGuiControlPanel(self):
        '''
            Add the control panel at initialization
        '''
        if self.use_gui == False:
            raise ValueError('Please set use_gui=True.')

        self.control_panel = {}

        jpos, _, _, _ = self.GetRevoluteJointSensorData()
        for i, joint_idx in enumerate(self.revolute_joints):
            panel_idx = pb.addUserDebugParameter(self.joint_name[joint_idx], *self.joint_range[joint_idx], jpos[i])
            self.control_panel[i] = panel_idx


    def ControlByUserDebugParameters(self):
        if self.control_panel is None:
            raise TypeError('Please add the control panel.')

        jpos_list = []
        
        for _, panel_idx in self.control_panel.items():
            panel_val = pb.readUserDebugParameter(panel_idx)
            jpos_list.append(panel_val)

        self.ApplyPositionControl(jpos_list)


    def StepSimulation(self):
        pb.stepSimulation()



    
    '''
        Sensor data
    '''
    def GetBaseSensorData(self):
        base_pos, base_ori = pb.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = pb.getBaseVelocity(self.robot_id)
        base_pos = np.array(base_pos)
        base_ori = np.array(base_ori)
        base_lin_vel = np.array(base_lin_vel)
        base_ang_vel = np.array(base_ang_vel)
        return base_pos, base_ori, base_lin_vel, base_ang_vel 


    def GetRevoluteJointSensorData(self):
        joint_states = list(zip(*pb.getJointStates(self.robot_id, self.revolute_joints)))
        joint_positions = np.array(joint_states[0])
        joint_velocities = np.array(joint_states[1])
        joint_forces = np.array(joint_states[2])
        applied_torques = np.array(joint_states[3])
        return joint_positions, joint_velocities, joint_forces, applied_torques


    def GetAllJointSensorData(self):
        joint_states = list(zip(*pb.getJointStates(self.robot_id, list(range(self.num_joints)))))
        joint_positions = np.array(joint_states[0])
        joint_velocities = np.array(joint_states[1])
        joint_forces = np.array(joint_states[2])
        applied_torques = np.array(joint_states[3])
        return joint_positions, joint_velocities, joint_forces, applied_torques


    def GetLinkSensorData(self):
        link_states = list(zip(*pb.getLinkStates(self.robot_id, self.revolute_joints, computeLinkVelocity = 1, computeForwardKinematics = 1)))
        world_pos = np.array(link_states[0])
        world_ori = np.array(link_states[1])
        world_lin_vel = np.array(link_states[6])
        world_ang_vel = np.array(link_states[7])
        return world_pos, world_ori, world_lin_vel, world_ang_vel


    def GetJacobian(self, link_idx):
        link_state = pb.getLinkState(self.robot_id, link_idx,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)

        q, _, _, _ = self.GetAllJointSensorData()
        zeros = [0.0] * len(q)

        # The jacobian matrix with respect to the base
        J_lin, J_ang = pb.calculateJacobian(self.robot_id, link_idx, link_state[2], q.tolist(), zeros, zeros)
        J_lin = np.array(J_lin)
        J_ang = np.array(J_ang)

        _, base_ori, _, _ = self.GetBaseSensorData()
        R_IB = np.array(pb.getMatrixFromQuaternion(base_ori)).reshape((3, 3))

        I_J_lin = R_IB @ J_lin
        I_J_ang = R_IB @ J_ang

        return I_J_lin, I_J_ang


    def ChangeDamping(self):
        '''
            Change the linear and angular damping of each joint to 0, to make sure forward dynamcs = inv(calculateInverseDynamics)
        '''
        for i in range(-1, self.num_joints):
            pb.changeDynamics(self.robot_id, i, linearDamping = 0, angularDamping = 0)


'''
    Testing functions
'''
def TestJacobian():
    env = PoppyPyBulletEnv(use_gui = True)
    env.AddGuiControlPanel()

    link_idx = 15
    for _ in range(3000):
        I_J_lin, I_J_ang = env.GetJacobian(link_idx)
        _, dq, _, _ = env.GetAllJointSensorData()
        
        link_state = pb.getLinkState(env.robot_id, link_idx, computeLinkVelocity = 1, computeForwardKinematics = 1)
        link_vel = np.array(link_state[6])

        print(np.linalg.norm(I_J_lin @ dq - link_vel))

        env.ControlByUserDebugParameters()
        env.StepSimulation()
        time.sleep(0.01)


if __name__ == '__main__':
    TestJacobian()