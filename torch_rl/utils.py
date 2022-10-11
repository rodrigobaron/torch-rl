import os
import sys
import time

import numpy as np
import torch
import random
import pandas as pd

import imageio

from collections import defaultdict
from datetime import timedelta


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = deterministic


def handle_terminal_ob(obs, dones, infos):
    real_obs = obs.copy()
    for idx, d in enumerate(dones):
        if "terminal_observation" in infos[idx].keys() and d:
            real_obs[idx] = infos[idx]["terminal_observation"]
    return real_obs


class MetricStore:
    def __init__(self):
        self._data = defaultdict(list)

    def log_metric(self, metric_name, metric_value, step):
        self._data[metric_name].append(dict(step=step, value=metric_value))

    def get_metric(self, metric_name):
        df = pd.DataFrame(self._data[metric_name])
        df = df.drop_duplicates(subset=['step'], keep='last')
        df = df.set_index("step", drop=True)
        df = df.rename(columns = {'value': metric_name})
        return df

    def list_metrics(self):
        return list(self._data.keys())


def mp4_to_gif(mp4_path, gif_path):
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    reader = imageio.get_reader(mp4_path)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(gif_path, fps=fps)
    for i,im in enumerate(reader):
        sys.stdout.write(f"\rWriting frame {i}")
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")


def get_time_hh_mm_ss(sec):
    td_str = str(timedelta(seconds=sec))
    x = td_str.split(':')
    return_str = []
    if int(x[0]) > 0:
        return_str.append(f"{x[0]} hour(s)")
    if int(x[1]) > 0 or int(x[0]) > 0:
        return_str.append(f"{x[1]} minute(s)")
    return_str.append(f"{x[2]} second(s)")

    return ", ".join(return_str)


class SPS:
    def __init__(self, total_steps, start_time=None):
        self._total_steps = total_steps
        self._start_time = start_time if start_time is not None else time.time()
        self._current_step = 0
        self._current_value = -1

    def get_total_steps(self):
        return self._total_steps

    def get_curent_value(self):
        return self._current_value

    def step(self, current_step, current_time=None):
        current_time = current_time if current_time is not None else time.time()
        self._current_step = current_step
        self._current_value = int(current_step / (current_time - self._start_time))

    def get_perc(self):
        current_step = self._total_steps if self._current_step + 1 >= self._total_steps else self._current_step
        return int(current_step / self._total_steps * 100)

    def get_remaining_seconds(self):
        return int(((self._total_steps - self._current_step) / self._current_value))

