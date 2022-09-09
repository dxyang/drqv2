# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import sys
sys.path.append(os.path.expanduser("~/code/rewardlearning-vid/"))

from pathlib import Path

import hydra
import numpy as np
import torch
import dm_env
from dm_env import StepType, specs

import drqv2.dmc as dmc
import drqv2.utils as utils
from drqv2.logger import Logger
from drqv2.replay_buffer import ReplayBufferStorage, make_replay_loader
from drqv2.video import TrainVideoRecorder, VideoRecorder


from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from models import ConvPolicy, Policy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

def process_metaworld_statespace_obs(o):
    return np.concatenate([o[:3], o[-3:]], dtype=np.float32)

class MetaworldWrapper():
    def __init__(self, env, use_image_state_space: bool, use_custom_reward: bool = False):
        self._env = env
        self._use_image_state_space = use_image_state_space
        self._use_custom_reward = use_custom_reward

        if self._use_custom_reward:
            if self._use_image_state_space:
                exp_name = "090122_metaworldreach_imgs"
                exp_folder = os.path.expanduser(f"~/code/rewardlearning-vid/output-exps/{exp_name}")
                assert os.path.exists(exp_folder)
                ranking_net = ConvPolicy(output_dim=1)
                ranking_net.load_state_dict(torch.load(f"{exp_folder}/ranking_policy.pt"))
            else:
                exp_name = "090122_metaworldreach_state"
                exp_folder = os.path.expanduser(f"~/code/rewardlearning-vid/output-exps/{exp_name}")
                assert os.path.exists(exp_folder)
                hidden_layer_size = 1000
                hidden_depth = 3
                obs_size = 6
                ranking_net = Policy(obs_size, 1, hidden_layer_size, hidden_depth)
                ranking_net.load_state_dict(torch.load(f"{exp_folder}/ranking_policy.pt"))

            ranking_net.to(device)
            ranking_net.eval()

            self.ranking_net = ranking_net

    def _convert_obs_to_timestep(self, obs_dict, step_type, action=None, reward=0.0, info={}):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        if self._use_image_state_space:
            obs = np.transpose(obs_dict["image_observation"], (2, 0, 1)) # CHW image
        else:
            obs = process_metaworld_statespace_obs(obs_dict["state_observation"]) # state, goal

        if self._use_custom_reward:
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
            with torch.no_grad():
                reward = self.ranking_net(batch_obs).cpu().item()

        return dmc.ExtendedTimeStep(observation=obs,
                                    step_type=step_type,
                                    action=action,
                                    reward=reward,
                                    discount=1.0)

    def reset(self):
        _ = self._env.reset()

        goal_size = self._env.goal_space.shape[0]
        goal_pos = np.random.uniform(low=[-0.3, 0.5, 0.175], high=[0.3, 0.9, 0.175], size=(goal_size,))
        hand_init = np.random.uniform(low=[0., 0.7, 0.175], high=[0., 0.7, 0.175], size=(goal_size,))
        obs_dict, obj_pos, goal_pos = self._env.reset_model_ood(goal_pos=goal_pos, hand_pos=hand_init)

        return self._convert_obs_to_timestep(obs_dict, StepType.FIRST)

    def step(self, action):
        obs_dict, reward, done, info = self._env.step(action)
        if done:
            return self._convert_obs_to_timestep(obs_dict, StepType.LAST, action, reward, info)
        else:
            return self._convert_obs_to_timestep(obs_dict, StepType.MID, action, reward, info)

    def observation_spec(self):
        if self._use_image_state_space:
            pixels_shape = (84, 84, 3)
            num_frames = 1
            env_obs_spec = specs.BoundedArray(shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                                dtype=np.uint8,
                                                minimum=0,
                                                maximum=255,
                                                name='observation')
        else:
            obs_shape = (6,) #self._env.observation_space.shape
            env_obs_spec = specs.BoundedArray(shape=obs_shape,
                                                dtype=np.float32,
                                                minimum=-np.inf,
                                                maximum=np.inf,
                                                name='observation')
        return env_obs_spec

    def action_spec(self):
        env_action_spec = specs.BoundedArray(shape=(4,),
                                            dtype=np.float32,
                                            minimum=-1,
                                            maximum=1,
                                            name='action')
        return env_action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)

class MetaworldWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        reach_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v2-goal-observable"]

        # TODO frame stack if desired
        self.train_env = reach_goal_observable_cls(seed=self.cfg.seed)
        self.train_env.random_init = False
        self.train_env._freeze_rand_vec = False
        self.train_env._obs_dict_state_space = True
        self.train_env._do_render_for_obs = True
        self.eval_env = reach_goal_observable_cls(seed=self.cfg.seed)
        self.eval_env.random_init = False
        self.eval_env._freeze_rand_vec = False
        self.eval_env._obs_dict_state_space = True
        self.eval_env._do_render_for_obs = True

        self.train_env = MetaworldWrapper(
            self.train_env, self.cfg.use_image_state_space,  self.cfg.use_custom_reward
        )
        self.eval_env = MetaworldWrapper(
            self.eval_env, self.cfg.use_image_state_space,  self.cfg.use_custom_reward
        )

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)

        if self.cfg.use_image_state_space:
            obs = np.transpose(time_step.observation["image_observation"], (2, 0, 1)) # CHW image
        else:
            obs = self.train_env.sim.render(
                256, 256, mode='offscreen', camera_name='topview'
            )
        self.train_video_recorder.init(obs)

        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                if self.cfg.use_image_state_space:
                    obs = np.transpose(time_step.observation["image_observation"], (2, 0, 1)) # CHW image
                else:
                    obs = self.train_env.sim.render(
                        256, 256, mode='offscreen', camera_name='topview'
                    )
                self.train_video_recorder.init(obs)

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)

            if self.cfg.use_image_state_space:
                obs = np.transpose(time_step.observation["image_observation"], (2, 0, 1)) # CHW image
                self.train_video_recorder.record(obs)
            else:
                obs = self.train_env.sim.render(
                    256, 256, mode='offscreen', camera_name='topview'
                )
                self.train_video_recorder.record(obs)

            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                import pdb; pdb.set_trace()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            print(time_step.discount)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v



@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import MetaworldWorkspace as W
    # from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()