import mne
from utils import logger


#----------------------------------------------------------------------
def epoching(raw, events, triggers, epoch_window, picks, ch_excluded=[]):
    """
    Exclude and epoch the data.
    
    raw = mne rawdata structure
    events = 3d events array, (1st colum: event time in sample, 3rd column: event id)
    triggers = dict of event id to consider
    epoch_window = beginning and end of the epochs to extract (sec)
    ch_excluded = channels to exclude
    """
    
    logger.info_green('Excluding channels: {}'.format(ch_excluded))
    
    for c in ch_excluded:
        if c not in raw.ch_names:
            logger.warning('Exclusion channel {} does not exist. Ignored.'.format(c))
            continue
        c_int = raw.ch_names.index(c)
        
        if c_int in picks:
            del picks[picks.index(c_int)]
            
    logger.info_green('Epoching in progress')
    epochs_train = mne.Epochs(raw, events, triggers, tmin=epoch_window[0], tmax=epoch_window[1], proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False, detrend=None, on_missing='warning')
    
    return epochs_train
