import numpy as np
import torch
import random
import pandas as pd

import imageio
import os, sys
from collections import defaultdict


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
