#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Example parameters
model_dir = '../../tests/data/output/test1'

gpu = 0
memfrac_gpu = 0.45
in_memory = False

num_workers = 8


# In[2]:


# Parameters
model_dir = "/home/ubuntu/wq/bpnet/examples/chip-nexus/2021-06-13_21-04-25_60667d18-3bc0-48c1-ae7a-8fce4360ba26"
gpu = 0
memfrac_gpu = 0.45
in_memory = True
num_workers = 8


# In[3]:


import bpnet
import pandas as pd
import numpy as np
import os
from pathlib import Path
from bpnet.dataspecs import DataSpec, TaskSpec
from bpnet.utils import create_tf_session
from bpnet.seqmodel import SeqModel
from bpnet.plot.evaluate import plot_loss, regression_eval


# In[4]:


model_dir = Path(model_dir)


# In[5]:


history_file = model_dir / "history.csv"


# In[6]:


create_tf_session(gpu)


# In[7]:


model = SeqModel.from_mdir(model_dir)


# ## Learning curves

# In[8]:


ds = DataSpec.load(model_dir / 'dataspec.yml')
tasks = list(ds.task_specs)


# In[9]:


# Best metrics
dfh = pd.read_csv(history_file)
dict(dfh.iloc[dfh.val_loss.idxmin()])


# In[10]:


plot_loss(dfh, [f"{task}/{p}_loss"
                for task in tasks
                for p in ['profile']
               ], figsize=(len(tasks)*4, 4))


# In[11]:


plot_loss(dfh, [f"{task}/{p}_loss"
                for task in tasks
                for p in ['counts']
               ], figsize=(len(tasks)*4, 4))


# ## Evaluation
# 
# Print the metrics:

# In[12]:


get_ipython().system('cat {model_dir}/evaluation.valid.json')


# In[13]:


from bpnet.utils import read_json
gin_config = read_json(model_dir / 'config.gin.json')


# In[14]:


from bpnet.datasets import StrandedProfile


# In[15]:


# TODO - add intervals?
dl_valid = StrandedProfile(ds, 
                           incl_chromosomes=gin_config['bpnet_data.valid_chr'], 
                           peak_width=gin_config['bpnet_data.peak_width'],
                           seq_width=gin_config['bpnet_data.seq_width'],
                           shuffle=False)


# In[16]:


valid = dl_valid.load_all(num_workers=num_workers)


# In[17]:


y_pred = model.predict(valid['inputs']['seq'])


# In[18]:


y_true = valid['targets']


# In[19]:


print(y_pred)


# In[20]:


print(y_true)


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


for task in tasks:
    plt.figure()
    yt = y_true[f'{task}/counts'].mean(-1)
    yp = y_pred[f'{task}/counts'].mean(-1)
    regression_eval(yt, 
                    yp, alpha=0.1, task=task)


# ## Profile plots

# In[21]:


peak_width = gin_config['bpnet_data.peak_width']


# In[22]:


np.random.seed(42)

N_RANDOM = 2
N_TOP_TOTAL_COUNT = 2
N_TOP_PER_BASE_COUNT = 2

def random_samples(arr, n=10, keep=None):
    """
    Randomly sample the values
      arr: numpy array
      n = number of samples to draw
    """
    if keep is None:
        keep = np.arange(len(arr))
    return list(pd.Series(np.arange(len(arr)))[keep].sample(n).index)


def top_summary_count(arr, end=10, start=0, keep=None, summary_fn=np.max):
    """
    Return indices where arr has the highest max(pos) + max(neg)

    Args:
      arr: can be an array or a list of arrays
      start: Where to start returning the values
      end: where to stop
    """
    if keep is None:
        keep = np.arange(len(arr))
    assert end > start
    # Top maxcount indicies
    return pd.Series(summary_fn(arr, axis=1).sum(1))[keep].sort_values(ascending=False).index[start:end]

# Check how much of the total counts is allocated at a single position
yt = sum([y_true[f'{task}/profile']
          for task in tasks])

(yt / yt.sum(axis=1, keepdims=True)).max(axis=(-1, -2)).max()
max_frac = (yt / yt.sum(axis=1, keepdims=True)).max(axis=(-1, -2))

max_pos = (yt ).max(axis=(-1, -2))
total_counts = (yt ).sum(axis=(-1, -2))
n_zeros = np.sum(yt == 0, axis=(-1, -2))

# Get the idx to test
idx_set = set()
for task in model.tasks:
    keep = (valid['metadata']['interval_from_task'] == task)
    idx_set.update(top_summary_count(y_true[f'{task}/profile'], N_TOP_TOTAL_COUNT, keep=keep, summary_fn=np.sum))
    idx_set.update(top_summary_count(y_true[f'{task}/profile'], N_TOP_PER_BASE_COUNT, keep=keep, summary_fn=np.max))
    idx_set.update(random_samples(y_true[f'{task}/profile'], N_RANDOM, keep=keep))


# In[23]:


import pybedtools
import seaborn as sns
from genomelake.extractors import FastaExtractor
from bpnet.preproc import resize_interval
from bpnet.data import get_dataset_item
from bpnet.plot.tracks import plot_tracks, filter_tracks
from bpnet.utils import flatten_list


# In[24]:


input_seqlen = gin_config['seq_width']


# In[25]:


def to_neg(track):
    """Use the negative sign for reads on the reverse strand
    """
    track = track.copy()
    track[:, 1] = - track[:, 1]
    return track    


# In[26]:


max_plot_width = 400
plot_seqlen = min(max_plot_width, input_seqlen)
trim_edge = max((input_seqlen - max_plot_width) // 2, 0)

xlim = [trim_edge, input_seqlen - trim_edge]
fig_width = 8 / 200 * plot_seqlen
rotate_y=90
fig_height_per_track=1
tasks = model.tasks

for idx in idx_set:
    # get the interval for that idx
    r = get_dataset_item(valid['metadata']['range'], idx)
    interval = pybedtools.create_interval_from_list([r['chr'], int(r['start']), int(r['end'])])
    interval_str = f"{interval.chrom}:{interval.start + trim_edge}-{interval.end - trim_edge}"

    # make prediction

    fe = FastaExtractor(ds.fasta_file)
    seq = fe([resize_interval(interval, input_seqlen, ignore_strand=True)])
    x = model.neutral_bias_inputs(input_seqlen, input_seqlen)
    x['seq'] = seq
    pred = model.predict(x)


    # compile the list of tracks to plot
    viz_dict =flatten_list([[
        # Observed
        (f"{task}\nObs", to_neg(y_true[f'{task}/profile'][idx])),
        # Predicted
        (f"\nPred", to_neg(pred[f'{task}/profile'][0] * np.exp(pred[f'{task}/counts'][0]))),
    ] for task_idx, task in enumerate(tasks)])

    sl = slice(*xlim)
    # Get ylim
    ylim = []
    for task in tasks:
        m = y_true[f'{task}/profile'][idx][sl].max()
        ylim.append((-m,m))
        m = (pred[f'{task}/profile'][0] * np.exp(pred[f'{task}/counts'][0])).max()
        ylim.append((-m,m))

    fig = plot_tracks(filter_tracks(viz_dict, xlim),
                      title=interval_str,
                      fig_height_per_track=fig_height_per_track,
                      rotate_y=rotate_y,
                      use_spine_subset=True,
                      # color=colors,
                      fig_width=fig_width,
                      ylim=ylim,
                      legend=False)
    fig.align_ylabels()
    sns.despine(top=True, right=True, bottom=True)

