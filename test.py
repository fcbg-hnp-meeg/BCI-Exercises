# Authors: Clemens Brunner <clemens.brunner@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

from prepocess import find_event_channel, preprocess


# load and preprocess data ####################################################
rawfile = r'C:\Users\adesvachez\git\pycnbi_data\testGUI\fif\20190619-160049-raw.fif'
raw = mne.io.Raw(rawfile, preload=True, verbose='ERROR')

# Find trigger channels
tch = find_event_channel(raw)

picks = mne.pick_channels(raw.info["ch_names"], ["F3", "F4", "Fz", "C3", "Cz", "C4", 'Pz', 'P3', 'P4'])

# Find events
events = mne.find_events(raw, stim_channel=raw.ch_names[tch], shortest_event=1, uint_cast=True, consecutive='increasing')
event_ids = {'RIGHT_GO':9, 'LEFT_GO':11}

# epoch data ##################################################################
tmin, tmax = -0.5, 4.5  # define epochs around events (in s)

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=picks, baseline=None, preload=True)

# compute ERDS maps ###########################################################
freqs = np.arange(2, 40, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-0.5, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None)  # for cluster test

# Run TF decomposition overall epochs
tfr = {}
for ev in event_ids:
    tfr[ev] = tfr_multitaper(epochs[ev], freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=True,
                     decim=2)
    tfr[ev].crop(tmin, tmax)
    tfr[ev].apply_baseline(baseline, mode="percent")
    
for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 10, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 10, 10, 10, 10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        tfr_ev.plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False)

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if not ax.is_first_col():
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()
    
    print('a')