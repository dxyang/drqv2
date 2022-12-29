# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, root_dir, folder_name = 'eval_video', render_size=256, fps=20, camera_name='topview'):
        if root_dir is not None:
            self.save_dir = root_dir / folder_name
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.camera_name = camera_name
        self.render_size = render_size
        if render_size % 16 != 0:
            print(f"making render size for the eval video recorder a multiple of 16")
            self.render_size = (render_size // 16 + 1) * 16
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.sim.render(
                    self.render_size, self.render_size, mode='offscreen', camera_name=self.camera_name
                )
                if self.camera_name in ["topview", "top_cap2", "left_cap2", "right_cap2"]:
                    frame = np.flipud(frame)

            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)

class KitchenVideoRecorder(VideoRecorder):
    def record(self, env):
        if self.enabled:
            frame = env.sim.render(
                self.render_size, self.render_size, mode='offscreen', camera_name=self.camera_name
            )
            if self.camera_name in ["topview", "top_cap2", "left_cap2", "right_cap2"]:
                frame = np.flipud(frame)

            self.frames.append(frame)


class TrainVideoRecorder:
    '''
    this video recorder like it's obs to come in HWC format!
    '''
    def __init__(self, root_dir, folder_name = 'train_video', render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / folder_name
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if obs.shape[1] % 16 != 0:
            self.render_size = (obs.shape[1] // 16 + 1) * 16
        self.record(obs)

    def record(self, obs):
        try:
            if self.enabled:
                frame = obs
                if frame.shape[1] % 16 != 0:
                    # resize to multiple of 16
                    frame = cv2.resize(
                        obs,
                        dsize=(self.render_size, self.render_size),
                        interpolation=cv2.INTER_CUBIC
                    )
                # not needed for metaworld frames
                # frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                #                 dsize=(self.render_size, self.render_size),
                #                 interpolation=cv2.INTER_CUBIC)
                self.frames.append(frame)
        except:
            import pdb; pdb.set_trace()

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
