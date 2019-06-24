import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

#----------------------------------------------------------------------
def compute_erds(epoch_trains, triggers, trf_window):
    """
    Compute and display the Event-Related desynchronisations/synchronisations
    """
    freqs = np.arange(2, 40, 1)  # frequencies from 2-40Hz
    n_cycles = freqs / 2  # use constant t/f resolution
    
    # Run TF decomposition overall epochs
    tfr = {}
    for ev in triggers:
        tfr[ev] = tfr_multitaper(epoch_trains[ev], freqs=freqs, n_cycles=n_cycles,
                         use_fft=True, return_itc=False, average=True,
                         decim=2)
        tfr[ev].crop(trf_window[0], trf_window[1])
        
    #Plotting
    plot_erds(tfr, epoch_trains, triggers)
    

#----------------------------------------------------------------------
def plot_erds(tfr, epochs, event_ids):
    '''
    Plot the ERDs and ERSs
    '''
    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                                 gridspec_kw={"width_ratios": [10, 10, 10, 1]})
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            tfr_ev.plot([ch],  baseline=(None, 0), mode='logratio', axes=ax, colorbar=False, show=False)
    
            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if not ax.is_first_col():
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1])
        fig.suptitle("ERD/S ({})".format(event))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fout = '%s/ERDS-%s' % (dir_path, event)
        fig.savefig(fout)
        