import os
import gymnasium as gym
import time
import cv2

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel

from modify.PPOdiy import PPO

def test(env, save_dir, test_for=1000, model=None, render_video=False):
    """ Testing function to view the behaviour of a model.

    Args:
        env (MIMoEnv): The environment on which the model should be tested. This does not have to be the same training
            environment, but action and observation spaces must match.
        save_dir (str): The directory in which any rendered videos will be saved.
        test_for (int): The number of timesteps the testing runs in total. This will be broken into multiple episodes
            if necessary.
        model:  The stable baselines model object. If ``None`` we take random actions instead. Default ``None``.
        render_video (bool): If ``True``, all episodes during testing will be recorded and saved as videos in
            `save_dir`.
    """
    obs, _ = env.reset()
    images = []
    im_counter = 0

    for idx in range(test_for):
        if model is None:
            print("No model, taking random actions")
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        obs, _, done, trunc, _ = env.step(action)
        if render_video:
            img = env.mujoco_renderer.render(render_mode="rgb_array")
            images.append(img)
        if done or trunc:
            time.sleep(1)
            obs, _ = env.reset()
            if render_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(os.path.join(save_dir, 'episode_{}.avi'.format(im_counter)), fourcc, 50, (500, 500))
                for img in images:
                    video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.destroyAllWindows()
                video.release()

                images = []
                im_counter += 1

    env.reset()

def main():
    """ CLI for the demonstration environments.

    Command line interface that can train and load models for the standup scenario. Possible parameters are:

    - ``--env``: The demonstration environment to use. Must be one of ``reach, standup, selfbody, catch``.
    - ``--train_for``: The number of time steps to train. No training takes place if this is 0. Default 0.
    - ``--test_for``: The number of time steps to test. Testing renders the environment to an interactive window, so
      the trained behaviour can be observed. Default 1000.
    - ``--save_every``: The number of time steps between model saves. This can be larger than the total training time,
      in which case we save once when training completes. Default 100000.
    - ``--algorithm``: The algorithm to train. This argument must be provided if you train. Must be one of
      ``PPO, SAC, TD3, DDPG, A2C, HER``.
    - ``--load_model``: The path to the model to load.
    - ``--save_model``: The directory name where the trained model will be saved. An input of "my_model", will lead to
        the model being saved under "models/<env>/my_model".
    - ``--use_muscles``: This flag switches between actuation models. By default, the spring-damper model is used. If
        this flag is set, the muscle model is used instead.
    - ``--render_video``: If this flag is set, each testing episode is recorded and saved as a video in the same
        directory as the models.
    """

    # parser.add_argument('--train_for', default=0, type=int,
    #                     help='Total timesteps of training')
    # parser.add_argument('--test_for', default=1000, type=int,
    #                     help='Total timesteps of testing of trained policy')
    # parser.add_argument('--save_every', default=100000, type=int,
    #                     help='Number of timesteps between model saves')
    # parser.add_argument('--load_model', default=False, type=str,
    #                     help='Name of model to load')
    # parser.add_argument('--save_model', default='', type=str,
    #                     help='Name of model to save')
    # parser.add_argument('--render_video', action='store_true',
    #                     help='Renders a video for each episode during the test run.')
    # parser.add_argument('--use_muscle', action='store_true',
    #                     help='Use the muscle actuation model instead of spring-damper model if provided.')

    # args = parser.parse_args()
    # env_name = args.env
    # algorithm = args.algorithm
    # load_model = args.load_model
    # save_model = args.save_model
    # save_every = args.save_every
    # train_for = args.train_for
    # test_for = args.test_for
    # render = args.render_video
    # use_muscle = args.use_muscle

    #  tensorboard --logdir mimoEnv/models/tensorboard_logs/selfbody/selfbodyTD3seed50/TD3_0/
    #  tensorboard --logdir mimoEnv/models/tensorboard_logs/selfbody/selfbodyPPOversion0.0/PPO_0/

    env_name = 'selfbody'
    alList=['PPO','TD3']
    algorithm = alList[0]
    save_model = 'selfbody' + algorithm+'version5.2_'+'1_new'
    save_every = 100000
    train_for = 250000
    test_for = 1000
    render = True
    use_muscle = True

    save_dir = os.path.join("models", env_name, save_model)
    actuation_model = MuscleModel if use_muscle else SpringDamperModel
    env = gym.make('MIMoSelfBody-v0', actuation_model=actuation_model)
    env.reset()

    if algorithm == 'PPO':
        from modify.PPOdiy import PPO as RL
    elif algorithm == 'TD3':
        from modify.TD3diy import TD3 as RL

    # load pretrained model or create new one
    model = RL("MultiInputPolicy", env,
               tensorboard_log=os.path.join("models", "tensorboard_logs", env_name, save_model),
               verbose=1)

    # train model
    counter = 0
    while train_for > 0:
        counter += 1
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model.learn(total_timesteps=train_for_iter, reset_num_timesteps=False)
        model.save(os.path.join(save_dir, "model_" + str(counter)))

    test(env, save_dir, model=model, test_for=test_for, render_video=render)

if __name__ == '__main__':
    main()
