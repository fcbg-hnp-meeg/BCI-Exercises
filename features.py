import os
import mne
import multiprocessing as mp
import numpy as np
from utils import logger

#----------------------------------------------------------------------
def get_psd_feature(epochs_train, window, psdparam, picks=None, preprocess=None, n_jobs=1):
    """
    Wrapper of get_psd() adding meta-information
    
    epochs_train = mne.Epochs object or list of mne.Epochs object.
    window = [t_start, t_end]. Time window range for computing PSD.
    psdparam = {fmin:float, fmax:float, wlen:float, wstep:int, decim:int}.
              fmin, fmax in Hz, wlen in seconds, wstep in number of samples.
    picks = Channels to compute features from.

    """
    sfreq = epochs_train[0].info['sfreq']
    w_frames = int(round(sfreq * psdparam['wlen']))  # window length in number of samples(frames)
    psde_sfreq = sfreq / psdparam['decim']
    
    psde = mne.decoding.PSDEstimator(sfreq=psde_sfreq, fmin=psdparam['fmin'], fmax=psdparam['fmax'],
            bandwidth=None, adaptive=False, low_bias=True, n_jobs=n_jobs, normalization='length', verbose='WARNING')
    
    logger.info_green('PSD computation')
    X_data, Y_data = get_psd(epochs_train, psde, w_frames, psdparam['wstep'], picks, n_jobs=n_jobs, decim=psdparam['decim'])
    w_starts = np.arange(0, epochs_train.get_data().shape[2] - w_frames, psdparam['wstep'])
    t_features = w_starts / sfreq + psdparam['wlen'] + window[0]
    
    return dict(X_data=X_data, Y_data=Y_data, wlen=psdparam['wlen'], ch_names=epochs_train[0].info['ch_names'], picks=picks,
                sw_frames=w_frames, psde=psde, times=t_features, decim=psdparam['decim'])


#----------------------------------------------------------------------
def get_psd(epochs, psde, wlen, wstep, picks=None, flatten=True, decim=1, n_jobs=1):
    """
    Compute multi-taper PSDs over a sliding window
    
    Input
    =====
    epochs = MNE Epochs object
    psde = MNE PSDEstimator object
    wlen = window length in frames
    wstep = window step in frames
    picks = channels to be used; use all if None
    flatten = boolean, see Returns section
    n_jobs = nubmer of cores to use, None = use all cores
    """
    
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if n_jobs > 1:
        logger.info('Opening a pool of %d workers' % n_jobs)
        pool = mp.Pool(n_jobs)    
    
    # compute PSD from sliding windows of each epoch
    labels = epochs.events[:, -1]
    epochs_data = epochs.get_data()
    w_starts = np.arange(0, epochs_data.shape[2] - wlen, wstep)
    X_data = None
    y_data = None
    results = []
    for ep in np.arange(len(labels)):
        title = 'Epoch %d / %d, Frames %d-%d' % (ep+1, len(labels), w_starts[0], w_starts[-1] + wlen - 1)
        if n_jobs == 1:
            # no multiprocessing
            results.append(slice_win(epochs_data[ep], w_starts, wlen, psde, picks, title, True))
        else:
            # parallel psd computation
            results.append(pool.apply_async(slice_win, [epochs_data[ep], w_starts, wlen, psde, picks, title, True]))    
    
    for ep in range(len(results)):
        if n_jobs == 1:
            r = results[ep]
        else:
            r = results[ep].get()  # windows x features
        X = r.reshape((1, r.shape[0], r.shape[1]))  # 1 x windows x features
        if X_data is None:
            X_data = X
        else:
            X_data = np.concatenate((X_data, X), axis=0)
    
        y = np.empty((1, r.shape[0]))  # 1 x windows
        y.fill(labels[ep])
        if y_data is None:
            y_data = y
        else:
            y_data = np.concatenate((y_data, y), axis=0)

    # close pool
    if n_jobs > 1:
        pool.close()
        pool.join()

    if flatten:
        return X_data, y_data.astype(np.int)
    else:
        xs = X_data.shape
        nch = len(epochs.ch_names)
        return X_data.reshape(xs[0], xs[1], nch, int(xs[2] / nch)), y_data.astype(np.int)    


#----------------------------------------------------------------------
def slice_win(epochs_data, w_starts, w_length, psde, picks=None, title=None, flatten=True, verbose=False):
    '''
    Compute PSD values of a sliding window

    Inputs
    ======
        epochs_data =([channels]x[samples]): raw epoch data
        w_starts (list) = starting indices of sample segments
        w_length (int) = window length in number of samples
        psde = MNE PSDEstimator object
        picks (list) = subset of channels within epochs_data
        title (string) = print out the title associated with PID
        flatten (boolean) = generate concatenated feature vectors
            If True: X = [windows] x [channels x freqs]
            If False: X = [windows] x [channels] x [freqs]
    Output:
    ======
        [windows] x [channels*freqs] or [windows] x [channels] x [freqs]
    '''

    # raise error for wrong indexing
    def WrongIndexError(Exception):
        logger.error('%s' % Exception)

    if type(w_length) is not int:
        logger.warning('w_length type is %s. Converting to int.' % type(w_length))
        w_length = int(w_length)
    if title is None:
        title = '[PID %d] Frames %d-%d' % (os.getpid(), w_starts[0], w_starts[-1]+w_length-1)
    else:
        title = '[PID %d] %s' % (os.getpid(), title)
    
    logger.info(title)

    X = None
    for n in w_starts:
        n = int(round(n))
        if n >= epochs_data.shape[1]:
            logger.error('w_starts has an out-of-bounds index %d for epoch length %d.' % (n, epochs_data.shape[1]))
            raise WrongIndexError
        window = epochs_data[:, n:(n + w_length)]

        # dimension: psde.transform( [epochs x channels x times] )
        psd = psde.transform(window.reshape((1, window.shape[0], window.shape[1])))
        psd = psd.reshape((psd.shape[0], psd.shape[1] * psd.shape[2]))
        if picks:
            psd = psd[0][picks]
            psd = psd.reshape((1, len(psd)))

        if X is None:
            X = psd
        else:
            X = np.concatenate((X, psd), axis=0)

        if verbose == True:
            logger.info('[PID %d] processing frame %d / %d' % (os.getpid(), n, w_starts[-1]))

    return X
