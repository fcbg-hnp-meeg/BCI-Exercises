import mne
import sys
from tfa import compute_erds
from prepocess import find_event_channel, preprocess
from epoching import epoching
from features import get_psd_feature
from trainer import cross_validate,  train_decoder
from loading import load_multi


#----------------------------------------------------------------------
def main():
    """
    Main
    """
    n_jobs = 8  # number of core for multiprocessing
    
    

    #------------------------------------------------------------------
    #  Load the data from several files
    #------------------------------------------------------------------
    #  Provide the folder path containing the files to analyse (add r before the path if \ in windows)
    filesDir = ''
    
    # Define the channels on which to apply both spatial and temporal filter, 
    # It is nmandatory to remove at least the TRIGGER, X1, X2, X3 and A2 channels
    filter_channels = [] 
    
    # Data loading:
    # spatial = None or 'CAR 'for spatial filtering
    # spectral = None or [freq1, freq2] for spectral filtering
    raw = load_multi(filesDir, spatial=, spectral=, channels=filter_channels, n_jobs=n_jobs)
    
    # Events extracted from the the raw file during loading
    events = raw[1]
    # Raw data, you can access to raw.ch_names, raw.info ...
    raw = raw[0]
    
    
    if len(sys.argv) < 2:
        cfg_file = input('Choose between erds or train')
    else:
        condition = sys.argv[1]
    
    if condition == 'erds':
        #------------------------------------------------------------------
        #  Time-Frequency Analysis: Event-Related De-Synchronization
        #------------------------------------------------------------------
        #  Fill erds() function
        erds(raw, events)
    elif condition == 'train':
        #------------------------------------------------------------------
        #  K-fold Cross-Validation and algorithm training
        #------------------------------------------------------------------
        #  Fill train() function
        train(raw, events, n_jobs)
    
#----------------------------------------------------------------------
def erds(raw, events):
    """
    Compute event-related synchronisation and desynchronisation
    
    raw = mne data structure
    events = events table
    """
    # Provide the trigger name and number on which to compute the epoching
    # You can find the events list in the file triggerdef_16.ini
    triggers = {} # {'name':val,...}

    # Provide the window around the triggers for time-frequency analysis
    #    A trial was lasting 5 sec
    trf_window = []    
    epoch_window = [x + y for x, y in zip(trf_window, [-1, 1])]   
    
    # Select the channels on which to epoch
    picks = mne.pick_channels(raw.info['ch_names'], ['HERE'])     
    
    # Epoching
    epoch_trains = epoching(raw, events, triggers, epoch_window, picks)
    
    # ERD/S
    compute_erds(epoch_trains, triggers, trf_window)
    
    
#----------------------------------------------------------------------
def train(raw, events, n_jobs=1):
    """"""
    # Provide the trigger name and number on which to compute the epoching 
    triggers = {}  # {'name':val,...}

    # Provide the epoch window
    #    A trial was lasting 5 sec
    epoch_window = []
    
    # Select the channels on which to epoch
    picks = mne.pick_channels(raw.info['ch_names'], ['HERE'])    

    # Epoching
    epoch_trains = epoching(raw, events, triggers, epoch_window, picks)
    
    # Define the parameters for PSD computation
    #    fmin = freq min of interest
    #    fmax = freq max of interest
    #    wlen = window length in sec
    #    wstep = window shift in samples
    #    decim = keep to one, downsampling factor
    PSD_params = dict(fmin=, fmax=, wlen=, wstep=, decim=1)
    
    # Compute PSD features [windows] x [channels x freqs]
    #    Picks = channels for PSD computation, if None all are used 
    featdata = get_psd_feature(epoch_trains, epoch_window, PSD_params, picks=None, n_jobs=n_jobs)

    # Training using k-fold cross-validation
    
    # Define the parameters for the random Forest
    #     Keep the predefined parameters
    random_forest = dict(trees=1000, depth=5,  seed=666)
    
    # Define the parameters for the k-fold cross-validation
    #    test_ratio = percentage of data used to test the classifier in each fold
    #    folds = number of folds (a classifier is trained each time on a different training dataset and its accuracy is estimated on a testing dataset)
    #    Keep the others with the predefined values
    cv = dict(test_ratio=, folds=, seed=0, export_result=True, ignore_thres=None, decision_thres=None, balance_samples=False)
    
    # Where to save the data
    cv_file = 'cv_results.txt'
    
    # Cross-validation
    #    The results can be seen in the cv_file
    cross_validate(random_forest, cv, featdata, cv_file, triggers, n_jobs)
    
    # Train the decoder, this time on the full dataset.
    #  One need to go online (real-time decoding) to see the real decoder accuracy
    train_decoder(random_forest, PSD_params, featdata, n_jobs)
    
    
if __name__ == '__main__':
    main()