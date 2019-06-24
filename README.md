# Brain-Computer Interface exercises

## Overview
The goal of this session is to prepare a Brain-Computer Interface, BCI. A BCI provides a mode of communication with an external device directly by the brain, without the involvement of any motor pathways.

The BCI modality used for this session is Motor Imagery, MI. MI is the translation of the user’s intention via mental imagination of motor movement, without limb movement. Here, the subject was asked to perform right- and left-hand motor imagery, for generating two possible commands.

An EEG dataset is provided to you. It contains 2 runs of 20 randomized trials (15 right- and 15 left-hand MI). The events labeling can be found in the file event.ini

## Installation

You will need to have python 3.X and pip installed. You can download them from the links below:

[python](https://www.python.org/downloads/)

[pip](https://pypi.org/project/pip/)

Conda can also be used.

Once both installations are done, type the following command in a terminal:

```pip install numpy scipy matplotlib mne scikit-learn```

Clone or download the git repository.

## Objectives:
1. Prepare the data
2. Perform a time-frequency analysis.
3. Train the classification algorithm and estimate its performance.

Open the main.py and fill the code. You do not need to modify any other file. 

To launch the code, go in the downloaded folder, open a terminal and run: ```python main.py```

The codes is based on MNE, an open-source Python library for analysing EEG data. [MNE Doc](https://www.martinos.org/mne/stable/documentation.html)

### Prepare the dataset

Load all the .fif files contains in the folder with 

```load_multi(filesDir, spatial=, spectral=, channels=filter_channels, n_jobs=n_jobs)``` 

and apply a spectral and a spatial filter on the eeg channels. Select them froom the list: 

```['TRIGGER', 'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Pz', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4']``` 

The cap layout is the [10-20 international layout](https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)).

The runs are concatenated in a mne raw format ```raw``` and the ```events``` are extracted.


### Time-frequency analysis

Rythmic neural activity within the alpha [8-12 Hz] and the beta [15-25 Hz] over the sensorymotor cortex is modulated during actual and imagined movements. 

Event-related desynchronisation (ERS) and event-related synchronisation (ERS) are short-lasting 
attenuation or augmentation of these rythms. 

ERD/S are shown on a spectrogram normalized by the baseline, computed few sec before the event. It has been shown that there is a contralateral dominance of mu and beta ERD and a ipsilateral dominance of mu ERS. 

Fill the funtion erds() and look to the spectrogram. 

### Train the classification algorithm and estimate its performance.

The classification algorithm used is the [Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#overview).

In order to estimate its accuracy, a [k-fold Cross-Validation](https://towardsdatascience.com/cross-validation-70289113a072) is used, where for each fold the data is split into a training and a testing set. 

The algorithm is trained on the training set and tested on the testing. The average accuracy over the k folds estimates
the real accuracy. At the end the algorithm is retrained on the complete dataset before being used.

Fill the function train() and look to the accuracy and the important features.

