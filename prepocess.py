from utils import logger
import mne
import numpy as np

#----------------------------------------------------------------------
def preprocess(raw, spatial=None, spectral=None, picks_ch=None, n_jobs=1):
    """
    Apply spatial and spectral filter on picked channels
    """
    data = raw._data
    sfreq = raw.info['sfreq']
    
    picks = mne.pick_channels(raw.info['ch_names'], picks_ch)
    
    if spatial == 'CAR':
        logger.info_green('Apply spatial filtering: {}'.format(spatial))
        data[picks] -= np.mean(data[picks], axis=0)
        
    if spectral is not None:
        logger.info_green('Apply spectral filtering: {}'.format(spectral))
        mne.filter.filter_data(data, sfreq, spectral[0], spectral[1], picks=picks,
                                       filter_length='auto', l_trans_bandwidth='auto',
                                       h_trans_bandwidth='auto', n_jobs=n_jobs, method='fir',
                                       iir_params=None, copy=False, phase='zero',
                                       fir_window='hamming', fir_design='firwin', verbose='ERROR')
    raw._data = data
    
    return raw

#----------------------------------------------------------------------    
def find_event_channel(raw, ch_names=None):
    """
    Find event channel using heuristics for pcl files.

    Input:
        raw: mne.io.RawArray-like object or numpy array (n_channels x n_samples)

    Output:
        channel index or None if not found.
    """
    signals = raw._data
    for ch_name in raw.ch_names:
        if 'TRIGGER' in ch_name:
            trig_index = raw.ch_names.index(ch_name)
            logger.info_green('Events channel at position {}'.format(trig_index))
            return raw.ch_names.index(ch_name)
        
    logger.info_green('Events channel not found')
    return None

