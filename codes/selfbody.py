""" This module contains a simple experiment where MIMo is tasked with touching parts of his own body.

The scene is empty except for MIMo, who is sitting on the ground. The task is for MIMo to touch a randomized target
body part with his right arm. MIMo is fixed in the initial sitting position and can only move his right arm.
Sensory inputs consist of touch and proprioception. Proprioception uses the default settings, but touch excludes
several body parts and uses a lowered resolution to improve runtime.
The body part can be any of the geoms constituting MIMo.

MIMos initial position is constant in all episodes. The target body part is randomized. An episode is completed
successfully if MIMo touches the target body part with his right arm.

The reward structure consists of a large fixed reward for touching the right body part, a shaping reward for touching
another body part, depending on the distance between the contact and the target body part, and a penalty for each time
step.

The class with the environment is :class:`~mimoEnv.envs.selfbody.MIMoSelfBodyEnv` while the path to the scene XML is
defined in :data:`SELFBODY_XML`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mimoEnv.utils as env_utils
from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS, SCENE_DIRECTORY
from mimoActuation.actuation import SpringDamperModel
from mimoTouch.touch import TrimeshTouch

#modify touch part
DiyTouchList = [22, 18, 21, 17, 20, 16, 2, 3, 4, 5, 6, 7, 8, 12, 13, 11]

#new reward
BODY_REWARD={
    'world': 330,
    'hip': 357,
    'lower_body': 389,
    'upper_body': 432,
    'head': 500,
    'left_eye': 450,
    'right_eye': 450 ,
    'right_upper_arm': 445 ,
    'right_lower_arm': 408 ,
    'right_hand': 368 ,
    'right_fingers': 350 ,
    'left_upper_arm': 445 ,
    'left_lower_arm': 400 ,
    'left_hand': 368 ,
    'left_fingers': 350 ,
    'right_upper_leg': 353 ,
    'right_lower_leg': 373 ,
    'right_foot': 300 ,
    'right_toes': 280 ,
    'left_upper_leg': 355 ,
    'left_lower_leg': 352 ,
    'left_foot': 333 ,
    'left_toes': 323,
 }


HEAD_DISTANCE={
    'world': 0.3398326383482117,
    'hip': 0.2854097469178288,
    'lower_body': 0.22045549602017542,
    'upper_body': 0.135000005952667,
    'head':0.0,
    'left_eye':0.10019800709095965,
    'right_eye':0.10019800709095966,
    'right_upper_arm':0.1089224452125349,
    'right_lower_arm':0.18397806632568517,
    'right_hand':0.26269160950975307,
    'right_fingers':0.3006440536667886,
    'left_upper_arm':0.10856893764816976,
    'left_lower_arm':0.20025890606442445,
    'left_hand':0.2638475745581758,
    'left_fingers':0.29807158056299443,
    'right_upper_leg':0.29394727929591835,
    'right_lower_leg':0.2530167172784273,
    'right_foot':0.4038258202242932,
    'right_toes':0.43706899628011736,
    'left_upper_leg':0.28843946203954923,
    'left_lower_leg':0.2959623976712721,
    'left_foot':0.3325238972167211,
    'left_toes':0.3539148643398256,
}

#TBD: change scales
# TOUCH_PARAMS = {
#     "scales": {
#         "left_foot": 0.05,
#         "right_foot": 0.05,
#         "left_lower_leg": 0.1,
#         "right_lower_leg": 0.1,
#         "left_upper_leg": 0.1,
#         "right_upper_leg": 0.1,
#         "hip": 0.1,
#         "lower_body": 0.1,
#         "upper_body": 0.1,
#         "head": 0.1,
#         "left_upper_arm": 0.01,
#         "left_lower_arm": 0.01,
#         "right_fingers": 0.01,
#         "left_eye": 1.0,
#         "right_eye": 1.0,
#         "right_upper_arm": 0.024,
#     },
#     "touch_function": "force_vector",
#     "response_function": "spread_linear",
# }

# orgianl
# TOUCH_PARAMS = {
#     "scales": {
#         "left_foot": 0.05,
#         "right_foot": 0.05,
#         "left_lower_leg": 0.1,
#         "right_lower_leg": 0.1,
#         "left_upper_leg": 0.1,
#         "right_upper_leg": 0.1,
#         "hip": 0.1,
#         "lower_body": 0.1,
#         "upper_body": 0.1,
#         "head": 0.1,
#         "left_upper_arm": 0.01,
#         "left_lower_arm": 0.01,
#         "right_fingers": 0.01
#     },
#     "touch_function": "force_vector",
#     "response_function": "spread_linear",
# }

TOUCH_PARAMS = {
    "scales": {
        "left_foot": 0.05,  # Smaller value for higher resolution
        "right_foot": 0.05,
        "left_lower_leg": 0.08,  # Adjusted from 0.1 for finer touch resolution
        "right_lower_leg": 0.08,
        "left_upper_leg": 0.12,  # Increased for lower resolution
        "right_upper_leg": 0.12,
        "hip": 0.15,  # Adjusted based on the size of the body part
        "lower_body": 0.15,
        "upper_body": 0.15,
        "head": 0.05,  # Larger value for broader sensing area
        "left_eye": 0.1,  # Typically, eyes would not have touch sensors
        "right_eye": 0.1,
        "right_upper_arm": 0.03,
        "left_upper_arm": 0.02,  # Increased for less sensitivity
        "left_lower_arm": 0.02,
        "right_fingers": 0.005,  # Decreased for more detailed touch data
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}

""" List of possible target bodies.

:meta hide-value:
"""

SITTING_POSITION_SIT = {
    "robot:hip_lean1": np.array([0.039088]), "robot:hip_rot1": np.array([0.113112]),
    "robot:hip_bend1": np.array([0.5323]), "robot:hip_lean2": np.array([0]), "robot:hip_rot2": np.array([0]),
    "robot:hip_bend2": np.array([0.5323]),
    "robot:head_swivel": np.array([0]), "robot:head_tilt": np.array([0]), "robot:head_tilt_side": np.array([0]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    "robot:left_shoulder_horizontal": np.array([0.683242]), "robot:left_shoulder_ad_ab": np.array([0.3747]),
    "robot:left_shoulder_rotation": np.array([-0.62714]), "robot:left_elbow": np.array([-0.756016]),
    "robot:left_hand1": np.array([0.28278]), "robot:left_hand2": np.array([0]), "robot:left_hand3": np.array([0]),
    "robot:left_fingers": np.array([-0.461583]),
    "robot:right_hip1": np.array([-1.51997]), "robot:right_hip2": np.array([-0.397578]),
    "robot:right_hip3": np.array([0.0976615]), "robot:right_knee": np.array([-1.85479]),
    "robot:right_foot1": np.array([-0.585865]), "robot:right_foot2": np.array([-0.358165]),
    "robot:right_foot3": np.array([0]), "robot:right_toes": np.array([0]),
    "robot:left_hip1": np.array([-1.23961]), "robot:left_hip2": np.array([-0.8901]),
    "robot:left_hip3": np.array([0.7156]), "robot:left_knee": np.array([-2.531]),
    "robot:left_foot1": np.array([-0.63562]), "robot:left_foot2": np.array([0.5411]),
    "robot:left_foot3": np.array([0.366514]), "robot:left_toes": np.array([0.24424]),
}

SITTING_POSITION_LAY = {
    # reset position
    "robot:hip_lean1": np.array([0.01]),
    "robot:hip_rot1": np.array([-0.0304]),
    "robot:hip_bend1": np.array([-0.299]),
    "robot:hip_lean2": np.array([0.0405]),
    "robot:hip_rot2": np.array([-0.02]),
    "robot:hip_bend2": np.array([-0.298]),
    "robot:head_swivel": np.array([0.0565]),
    "robot:head_tilt": np.array([-0.806]),
    "robot:head_tilt_side": np.array([0.0211]),
    "robot:left_eye_horizontal": np.array([8e-08]),
    "robot:left_eye_vertical": np.array([-3.18e-05]),
    "robot:left_eye_torsional": np.array([8.56e-08]),
    "robot:right_eye_horizontal": np.array([-8e-08]),
    "robot:right_eye_vertical": np.array([-3.18e-05]),
    "robot:right_eye_torsional": np.array([-8.56e-08]),
    "robot:left_shoulder_horizontal": np.array([-0.489]),
    "robot:left_shoulder_ad_ab": np.array([0.283]),
    "robot:left_shoulder_rotation": np.array([-0.273]),
    "robot:left_elbow": np.array([0.0873]),
    "robot:left_hand1": np.array([-0.489]),
    "robot:left_hand2": np.array([-0.0475]),
    "robot:left_hand3": np.array([-0.0565]),
    "robot:left_fingers": np.array([-0.325]),
    "robot:right_hip1": np.array([-0.57]),
    "robot:right_hip2": np.array([-0.008]),
    "robot:right_hip3": np.array([-0.0038]),
    "robot:right_knee": np.array([0.0279]),
    "robot:right_foot1": np.array([-0.0675]),
    "robot:right_foot2": np.array([-0.00733]),
    "robot:right_foot3": np.array([0.00688]),
    "robot:right_toes": np.array([-0.00243]),
    "robot:left_hip1": np.array([-0.0578]),
    "robot:left_hip2": np.array([0.00977]),
    "robot:left_hip3": np.array([0.00403]),
    "robot:left_knee": np.array([0.0348]),
    "robot:left_foot1": np.array([-0.0613]),
    "robot:left_foot2": np.array([0.00791]),
    "robot:left_foot3": np.array([-0.00749]),
    "robot:left_toes": np.array([-0.00194]),
}

""" Initial position of MIMo. Specifies initial values for all joints.
We grabbed these values by posing MIMo using the MuJoCo simulate executable and the positional actuator file.
We need these not just for the initial position but also resetting the position (excluding the right arm) each step.

:meta hide-value:
"""

SITTING_POSITION= SITTING_POSITION_SIT

SELFBODY_XML = os.path.join(SCENE_DIRECTORY, "selfbody_scene.xml")
""" Path to the scene for this experiment.

:meta hide-value:
"""

CATCH_CAMERA_CONFIG={
    "trackbodyid": 0,
    "distance": 1.5,
    "lookat": np.asarray([0.15, -0.04, 0.6]),
    "elevation": -20,
    "azimuth": 160,
}
""" Camera configuration so it looks straight at the hand.

:meta hide-value:
"""

class MIMoSelfBodyEnv(MIMoEnv):
    """ MIMo learns about his own body.

    MIMo is tasked with touching a given part of his body using his right arm.
    Attributes and parameters are mostly identical to the base class, but there are two changes.
    The constructor takes two arguments less, `goals_in_observation` and `done_active`, which are both permanently
    set to ``True``.
    Finally, there are two extra attributes for handling the goal state. The :attr:`.goal` attribute stores the target
    geom in a one hot encoding, while :attr:`.target_geom` and :attr:`.target_body` store the geom and its associated
    body as an index. For more information on geoms and bodies please see the MuJoCo documentation.

    Attributes:
        target_geom (int): The body part MIMo should try to touch, as a MuJoCo geom.
        target_body (str): The name of the kinematic body that the target geom is a part of.
        init_sitting_qpos (numpy.ndarray): The initial position.
    """

    def __init__(self,
                 model_path=SELFBODY_XML,
                 initial_qpos=SITTING_POSITION,
                 frame_skip=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 actuation_model=SpringDamperModel,
                 goals_in_observation=True,
                 done_active=True,
                 default_camera_config=CATCH_CAMERA_CONFIG,
                 show_sensors=True,
                 print_space_sizes=True,
                 ):

        self.target_geom = 0  # The geom on MIMo we are trying to touch
        self.target_body = ""  # The body that the goal geom belongs to
        self.goal = np.zeros(37)

        # extra parameters
        self.steps = 0
        self.show_sensors = show_sensors

        # statics
        self.touch_target=0
        self.touch_other=0
        self.touch_none=0
        self.touch_floor=0

        self.other_part='world'

        self.touch_dict={
        'world': 0,
        'hip': 0,
        'lower_body': 0,
        'upper_body': 0,
        'head': 0,
        'left_eye': 0,
        'right_eye': 0 ,
        'right_upper_arm': 0 ,
        'right_lower_arm': 0 ,
        'right_hand': 0 ,
        'right_fingers': 0 ,
        'left_upper_arm': 0 ,
        'left_lower_arm': 0 ,
        'left_hand': 0 ,
        'left_fingers': 0 ,
        'right_upper_leg': 0 ,
        'right_lower_leg': 0 ,
        'right_foot': 0 ,
        'right_toes': 0 ,
        'left_upper_leg': 0 ,
        'left_lower_leg': 0 ,
        'left_foot': 0 ,
        'left_toes': 0,
        'floor':0,
        }

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         default_camera_config=default_camera_config,
                         )

        env_utils.set_joint_qpos(self.model,
                                 self.data,
                                 "mimo_location",
                                 np.array([0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029]))
        #  "mimo_location": np.array([0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029]),
        for joint_name in SITTING_POSITION:
            env_utils.lock_joint(self.model, joint_name, joint_angle=SITTING_POSITION[joint_name][0])
        # Let sim settle for a few timesteps to allow weld and locks to settle
        self.do_simulation(np.zeros(self.action_space.shape), 25)
        self.init_sitting_qpos = self.data.qpos.copy()

        # if print_space_sizes:
        #     print("Observation space:")
        #     for key in self.observation_space:
        #         print(key, self.observation_space[key].shape)
        #     print("\nAction space: ", self.action_space.shape)

    #cover touch setup
    def touch_setup(self, touch_params):
        """ Perform the setup and initialization of the touch system.

        Uses the more complicated Trimesh implementation. Also plots the sensor points if :attr:`.show_sensors` is
        ``True``.

        Args:
            touch_params (dict): The parameter dictionary.
        """
        self.touch = TrimeshTouch(self, touch_params=touch_params)

        # Count and print the number of sensor points on each body
        count_touch_sensors = 0
        if self.show_sensors:
            print("Number of sensor points for each body: ")

        for body_id in self.touch.sensor_positions:
            # if self.show_sensors:
            #     print(self.model.body(body_id).name, self.touch.sensor_positions[body_id].shape[0])
            count_touch_sensors += self.touch.get_sensor_count(body_id)

        # print("Total number of sensor points: ", count_touch_sensors)

        # Plot the sensor points for each body once
        # if self.show_sensors:
        #     print('all sensor',self.touch.sensor_positions)
        #     env_utils.plot_points_dict(self.touch.sensor_positions, limit=1., title='whole sensor')
            # for body_id in self.touch.sensor_positions:
            #     body_name = self.model.body(body_id).name
            #     env_utils.plot_points(self.touch.sensor_positions[body_id], limit=1., title=body_name)

    #cover callback function
    def _step_callback(self):
        """ A custom callback that is called after stepping the simulation, but before collecting observations.

        Useful to enforce additional constraints on the simulation state before observations are collected.
        Note that the sensory modalities do not update until get_obs is called, so they will not have updated to the
        current timestep.
        """

    def _substep_callback(self):
        """ A custom callback that is called after each simulation substep.
        """

    def _obs_callback(self):
        """ A custom callback that is called after collecting the observations.

        Like _step_callback, but with up-to-date observations.
        """
        self.steps += 1

    def sample_goal(self):
        """Samples a new goal and returns it.

        The goal consists of a target geom that we try to touch, returned as a one-hot encoding.
        We also populate :attr:`.target_geom` and :attr:`.target_body`. which are used by other functions.

        Returns:
            numpy.ndarray: The target geom in a one hot encoding.
        """
        # randomly select geom as target (except for 2 latest geoms that correspond to fingers)
        active_geom_codes = list(self.touch.sensor_outputs.keys())  # DiyTouchList

        #random
        target_geom_idx = np.random.randint(len(active_geom_codes) - 2) # int

        #weighted random
        # active_geom_weights = [BODY_REWARD[self.model.body(self.model.geom(geom).bodyid).name] for geom in active_geom_codes[:-2]]
        # target_geom_idx = np.random.choice(len(active_geom_codes) - 2, p=active_geom_weights / np.sum(active_geom_weights))  # int

        self.target_geom = active_geom_codes[int(target_geom_idx)] # int

        # We want the output of the desired goal as a one hot encoding,
        # rather than the raw index
        target_geom_onehot = np.zeros(37)  # 36 geoms in MIMo
        if isinstance(self.target_geom, int):
            target_geom_onehot[self.target_geom] = 1
        self.target_body = self.model.body(self.model.geom(self.target_geom).bodyid).name
        # print('zzvliu Target body',self.target_body)
        return target_geom_onehot

    def is_success(self, achieved_goal, desired_goal):
        """ We have succeeded when we have a touch sensation on the goal body.

        We ignore the :attr:`.goal` attribute in this for performance reasons and determine the success condition
        using :attr:`.target_geom` instead. This allows us to save a number of array operations each step.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: If MIMo has touched the target geom.
        """
        # check if contact with target geom:
        target_geom_touch_max = np.max(self.touch.sensor_outputs[self.target_geom])
        contact_with_target_geom = (target_geom_touch_max > 0)
        return contact_with_target_geom

    # test function

    def once_test_func(self):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Loop over all body parts
        for i in range(37):  #37
            name = self.model.body(self.model.geom(i).bodyid).name
            xpos = self.data.body(name).xpos
            ax.scatter(*xpos,label=name)

        # for body_part, scale in TOUCH_PARAMS["scales"].items():
        #     xpos = self.data.body(body_part).xpos
        #     ax.scatter(*xpos, label=body_part)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()  # Add a legend to distinguish body parts
        # save as PDF （dpi=600）
        plt.savefig(  'bodysensor.png', format='png', dpi=600)
        plt.show()

    # check all sensor ouput
    def find_touch_max(self, sensor_output):
        touch_part=False
        max_values = {}
        for key, array in sensor_output.items():
            np_array = np.array(array)  # Convert list to NumPy array
            max_value = np.max(np_array)
            if max_value > 0:
                max_index = np.unravel_index(np.argmax(np_array), np_array.shape)
                max_values[key] = (max_value, max_index)
                touch_part = True
                self.other_part = self.model.body(self.model.geom(key).bodyid).name

        # Now max_values contains all the keys with their max values and indices

        # if max_values and all(k != self.target_geom for k in max_values):
        #     touch_part=True
            # for key, value in max_values.items():
                # print(f"touch id (Key): {key}, Max Value: {value[0]},target id:{self.target_geom}")
        # else:
        #     print("max_values is empty or contains key 100")

        return touch_part

    def touch_distance(self):
        head_pos = self.data.body('head').xpos
        for i in range(37):
            name = self.model.body(self.model.geom(i).bodyid).name
            target_body_pos = self.data.body(name).xpos
            distance = np.linalg.norm(head_pos - target_body_pos)
            print('zzv name',name,distance)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward each step.

        Three different rewards can be returned:

        - If we touched the target geom, the reward is 500.
        - If we touched a geom, but not the target, the reward is the negative of the distance between the touch
          contact and the target body.
        - Otherwise the reward is -1.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        # DiyTouchList
        active_geom_codes = list(self.touch.sensor_outputs.keys())

        #if touch right part
        touch_part = self.find_touch_max(self.touch.sensor_outputs)

        # fingers_touch_max = max(
        #     np.max(self.touch.sensor_outputs[active_geom_codes[-1]]),
        #     np.max(self.touch.sensor_outputs[active_geom_codes[-2]])
        # )

        #touch_part
        contact_with_fingers =  touch_part #(fingers_touch_max > 0)

        # touch floor
        touch_floor =False
        if self.data.body("right_fingers").xpos[2] <= 0.02:
            touch_floor=True

        # print body
        # self.once_test_func()

        # self.touch_distance()

        # compute reward:
        if info["is_success"]:
            if self.target_body in BODY_REWARD:
                reward = 500#BODY_REWARD[self.target_body]
                self.touch_dict[self.target_body]+=1
            else:
                reward = 500
                self.touch_dict[self.target_body] = 1

            self.touch_target += 1
            # print('reward: ',reward,' touched:',self.target_body)

        elif touch_floor:
            self.touch_floor += 1
            self.touch_dict['floor'] += 1
            reward = 0

        elif contact_with_fingers:
            target_body_pos = self.data.body(self.target_body).xpos
            fingers_pos = self.data.body("right_fingers").xpos
            distance = np.linalg.norm(fingers_pos - target_body_pos)
            reward = -distance

            self.touch_other += 1

            if self.target_body in BODY_REWARD:
                self.touch_dict[self.other_part] += 1
            else:
                self.touch_dict[self.other_part] = 1
            # print('zzvliu touched other part,distance: -', distance)
        else:
            reward = -1
            self.touch_none += 1

            # if self.data.body("right_fingers").xpos[2] <= 0.02:
            #     self.touch_floor += 1
            #     self.touch_dict['floor'] += 1
            # else:
            #     self.touch_none += 1

        return reward

    def reset_model(self):
        """ Reset to the initial sitting position.

        Returns:
            Dict: Observations after reset.
        """
        # set qpos as new initial position and velocity as zero
        qpos = self.init_sitting_qpos
        qvel = np.zeros(self.data.qvel.shape)
        self.set_state(qpos, qvel)
        print('Total:',self.steps,'Target:',self.touch_target ,'Other:',self.touch_other,'None:', self.touch_none,'Floor:',self.touch_floor)

        print('-------------------Touch list-------------------')
        for key, value in self.touch_dict.items():
            print(f"{key}: {value}")

        return self._get_obs()

    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function that always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``.
        """
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def get_achieved_goal(self):
        """ Dummy function that returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros(self.goal.shape)
