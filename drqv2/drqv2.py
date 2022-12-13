# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import drqv2.utils as utils

from r3m import load_r3m

from reward_extraction.data import H5PyTrajDset

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class FeatureEncoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(utils.weight_init)
        self.repr_dim = output_dim

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class R3MFeatureExtractor(nn.Module):
    def __init__(
        self,
        do_multiply_255: bool = True,
        freeze_backbone: bool = True,
    ):
        super(R3MFeatureExtractor, self).__init__()

        # get the backbone
        self.r3m = load_r3m("resnet18") # resnet18, resnet34, resnet50

        self.freeze_r3m = freeze_backbone
        if self.freeze_r3m:
            self.r3m.eval()
            for param in self.r3m.parameters():
                param.requires_grad = False

        self.r3m_embedding_dim = 512 # for resnet 18 - 512, for resnet50 - 2048
        self.repr_dim = self.r3m_embedding_dim

        self.do_multiply_255 = do_multiply_255

    def forward(self, x: torch.Tensor):
        # x for drqv2 standpoint is 0-255 uint8 so make it a float
        x = x.float()

        # r3m expects things to be [0-255] instead of [0-1]!!!
        if self.do_multiply_255:
            x = x * 255.0

        # some computational savings
        if self.freeze_r3m:
            with torch.no_grad():
                x = self.r3m(x)
        else:
            x = self.r3m(x)

        return x


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 with_ppc):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.image_state_space = len(obs_shape) == 3
        self.with_ppc = with_ppc

        # models
        is_highres = False
        if self.image_state_space:
            if obs_shape == (3, 84, 84):
                self.encoder = Encoder(obs_shape).to(device)
            elif obs_shape == (3, 224, 224):
                is_highres = True
                self.encoder = R3MFeatureExtractor(do_multiply_255=False).to(device)

            self.proprioception_dim = 0
            if self.with_ppc:
                self.proprioception_dim = 4 # for metaworld, this is hand xyz and gripper state

            self.actor = Actor(self.encoder.repr_dim + self.proprioception_dim, action_shape, feature_dim,
                            hidden_dim).to(device)

            self.critic = Critic(self.encoder.repr_dim + self.proprioception_dim, action_shape, feature_dim,
                                hidden_dim).to(device)
            self.critic_target = Critic(self.encoder.repr_dim + self.proprioception_dim, action_shape,
                                        feature_dim, hidden_dim).to(device)
        else:
            input_dim = obs_shape[0]
            if input_dim in (6, 9, 10, 17):
                self.encoder = FeatureEncoder(input_dim, 512, 128, 2).to(device)
            elif input_dim == 512:
                self.encoder = FeatureEncoder(input_dim, 1024, 128, 2).to(device)
            else:
                assert False # unexpected

            self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                            hidden_dim).to(device)

            self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                                hidden_dim).to(device)
            self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                        feature_dim, hidden_dim).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        if is_highres:
            #84x84 => pad=4 so for 224x224 let's triple it roughly?
            self.aug = RandomShiftsAug(pad=12)
        else:
            self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode, proprioception=None):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))

        if self.image_state_space:
            if self.with_ppc:
                assert proprioception is not None
                ppc = torch.as_tensor(proprioception, device=self.device).unsqueeze(0)
                obs = torch.cat([obs, ppc], dim=1)

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, ppc, next_ppc = utils.to_torch(
            batch, self.device)

        # augment
        if self.image_state_space:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()

        # encode
        obs = self.encoder(obs)
        if self.image_state_space and self.with_ppc:
            obs = torch.cat([obs, ppc], dim=1)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
            if self.image_state_space and self.with_ppc:
                next_obs = torch.cat([next_obs, next_ppc], dim=1)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def bc_init_agent(self, bc_dataset: H5PyTrajDset, num_steps: int, num_trajs: int, batch_size: int = 64):
        TRAJ_HORIZON = 100
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        import pickle
        import os

        work_dir = os.path.dirname(bc_dataset.save_path)

        losses = []

        for i in tqdm(range(num_steps)):
            # extract observation from bc dataset
            traj_idxs = np.random.randint(num_trajs, size=(batch_size,))
            time_idxs = np.random.randint(TRAJ_HORIZON, size=(batch_size,))

            trajectories = [bc_dataset[traj_idx] for traj_idx in traj_idxs]
            s_ts = []
            a_ts = []
            metaworld_s_ts = []
            for traj, t_idx in zip(trajectories, time_idxs):
                s_ts.append(traj[0][t_idx])
                a_ts.append(traj[1][t_idx])
                metaworld_s_ts.append(traj[4][t_idx])
            obss = torch.as_tensor(np.array(s_ts), device=self.device)
            a_ts = torch.as_tensor(np.array(a_ts), device=self.device)
            metaworld_s_ts = torch.as_tensor(np.array(metaworld_s_ts), device=self.device)

            # preproces observation fancily
            obss = self.encoder(obss)
            if self.image_state_space:
                if self.with_ppc:
                    ppcs = metaworld_s_ts[:, :self.proprioception_dim]
                    ppcs = torch.as_tensor(ppcs, device=self.device)
                    obss = torch.cat([obss, ppcs], dim=1)

            # forward pass
            stddev = utils.schedule(self.stddev_schedule, 0)
            dist = self.actor(obss, stddev)
            action_hats = dist.mean

            # supervised loss
            mse_loss = nn.MSELoss()
            loss = mse_loss(a_ts, action_hats)

            # do the grad thing
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.actor_opt.step()

            losses.append(loss.detach().item())

            if i % 1000 == 0 or i == num_steps - 1:
                losses_np = np.array(losses)
                plt.clf(); plt.cla()
                plt.plot(losses_np, label="train", color='blue')
                plt.savefig(f"{work_dir}/bc_loss.png")
                losses_dict = {
                    "losses": losses_np,
                }
                pickle.dump(losses_dict, open(f"{work_dir}/bc_losses.pkl", "wb"))

