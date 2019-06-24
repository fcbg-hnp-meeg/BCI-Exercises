import mne
import os
import numpy as np
from utils import logger, parse_path_list, get_file_list
from prepocess import preprocess, find_event_channel

#----------------------------------------------------------------------
def load_multi(src, spatial=None, spectral=None, channels=None, n_jobs=1):
    """
    Load multiple data files and concatenate them into a single series

    - Assumes all files have the same sampling rate and channel order.
    - Event locations are updated accordingly with new offset.

    @params:
        src: directory or list of files.
        spatial: apply spatial filter while loading.
        spectral: apply spectral filter
        channels: list of channel names to apply filter.

    See load_raw() for more low-level details.

    """

    if type(src) == str:
        if not os.path.isdir(src):
            logger.error('%s is not a directory or does not exist.' % src)
            raise IOError
        flist = []
        for f in get_file_list(src):
            if parse_path_list(f)[2] == 'fif':
                flist.append(f)
    elif type(src) in [list, tuple]:
        flist = src
    else:
        logger.error('Unknown input type %s' % type(src))
        raise TypeError

    if len(flist) == 0:
        logger.error('load_multi(): No fif files found in %s.' % src)
        raise RuntimeError
    elif len(flist) == 1:
        return load_raw(flist[0], spatial=spatial, spectral=spectral, channels=channels, n_jobs=n_jobs)

    # load raw files
    rawlist = []
    for f in flist:
        logger.info('Loading %s' % f)
        raw, _ = load_raw(f, spatial=spatial, spectral=spectral, channels=channels, n_jobs=n_jobs)
        rawlist.append(raw)

    # concatenate signals
    signals = None
    for raw in rawlist:
        if signals is None:
            signals = raw._data
        else:
            signals = np.concatenate((signals, raw._data), axis=1) # append samples

    # create a concatenated raw object and update channel names
    raw = rawlist[0]
    trigch = find_event_channel(raw)
    ch_types = ['eeg'] * len(raw.ch_names)
    if trigch is not None:
        ch_types[trigch] = 'stim'
    info = mne.create_info(raw.ch_names, raw.info['sfreq'], ch_types)
    raw_merged = mne.io.RawArray(signals, info)

    # re-calculate event positions
    events = mne.find_events(raw_merged, stim_channel='TRIGGER', shortest_event=1, consecutive=True)
    
    return raw_merged, events

#----------------------------------------------------------------------
def load_raw(rawfile, spatial=None, spectral=None, channels=None, n_jobs=1, verbose='ERROR'):
    """
    Loads data from a fif-format file.
    You can convert non-fif files (.eeg, .bdf, .gdf, .pcl) to fif format.

    Parameters:
    rawfile: (absolute) data file path
    spatial: 'car' | None
    channels: channels to apply filters
    
    Returns:
    raw: mne.io.RawArray object. First channel (index 0) is always trigger channel.
    events: mne-compatible events numpy array object (N x [frame, 0, type])

    """

    if not os.path.exists(rawfile):
        logger.error('File %s not found' % rawfile)
        raise IOError
    if not os.path.isfile(rawfile):
        logger.error('%s is not a file' % rawfile)
        raise IOError

    raw = mne.io.Raw(rawfile, preload=True, verbose=verbose)
    if spatial is not None:
        raw = preprocess(raw, spatial=spatial, spectral=spectral, picks_ch=channels, n_jobs=n_jobs)

    tch = find_event_channel(raw)
    events = mne.find_events(raw, stim_channel=raw.ch_names[tch], shortest_event=1, uint_cast=True, consecutive='increasing')
    # MNE's annoying hidden cockroach: first_samp
    events[:, 0] -= raw.first_samp

    return raw, events

