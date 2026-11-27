"""Gym-v3-like toy environment for the local synthetic tests only (never used on the server).

Observation = qpos[1:] + clip(qvel, -10, 10) like the gym v3 hopper / walker2d; set_state / _get_obs exist for the
j016 state-restoration path. Termination: height qpos[1] outside (0.8, 2.0) or |angle| > 1 (walker-like rule)."""
import numpy as np


class Space:
    def __init__(self, shape, high=None):
        self.shape = shape
        self.high = high


class FakeEnv:
    def __init__(self, sd, ad):
        self.sd, self.ad = sd, ad
        self.nq = 6 if sd == 11 else 9
        self.nv = self.nq
        self.observation_space = Space((sd,))
        self.action_space = Space((ad,), np.ones(ad))
        self.unwrapped = self
        self.frame_skip = 4
        self.rng = np.random.RandomState(0)
        self.t = 0
        self.qpos = np.zeros(self.nq)
        self.qvel = np.zeros(self.nv)

    def seed(self, s=None):
        self.rng = np.random.RandomState(s)

    def reset(self):
        self.qpos = np.r_[0., 1.25, self.rng.uniform(-.005, .005, self.nq-2)]
        self.qvel = self.rng.uniform(-.005, .005, self.nv)
        self.t = 0
        return self._get_obs()

    def set_state(self, qpos, qvel):
        self.qpos = np.array(qpos, dtype=float)
        self.qvel = np.array(qvel, dtype=float)

    def _get_obs(self):
        return np.concatenate([self.qpos[1:], np.clip(self.qvel, -10, 10)])

    def step(self, a):
        a = np.asarray(a, dtype=float)
        self.qvel = .9*self.qvel+.05*np.resize(a, self.nv)
        self.qpos = self.qpos+.02*self.qvel[:self.nq]
        self.qpos[1] += .01*float(a[0])
        self.t += 1
        z, ang = self.qpos[1], self.qpos[2]
        healthy = (.8 < z < 2.) and abs(ang) < 1.
        reward = 1.+float(self.qvel[0])-1e-3*float(np.sum(a**2))
        done = not healthy
        info = {}
        if self.t >= 1000 and not done:
            done = True
            info['TimeLimit.truncated'] = True
        return self._get_obs(), reward, done, info

    def close(self):
        pass


def make(sd, ad):
    return FakeEnv(sd, ad)
