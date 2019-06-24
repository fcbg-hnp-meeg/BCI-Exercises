import mne
import os
import features
import numpy as np
import sklearn.metrics as skmetrics
from sklearn.ensemble import RandomForestClassifier
from utils import logger
import multiprocessing as mp
from utils import feature2chz, save_obj, sort_by_value
import platform

# scikit-learn old version compatibility
try:
    from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut
    SKLEARN_OLD = False
except ImportError:
    from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
    SKLEARN_OLD = True

mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1'

#----------------------------------------------------------------------
def cross_validate(cfg_classifier, cfg_cv, featdata, cv_file, triggers, n_jobs=1):
    """
    Perform the stratified cross validation.
    """
    # Init a classifier
    cls = RandomForestClassifier(n_estimators=cfg_classifier['trees'], max_features='auto',
                                     max_depth=cfg_classifier['depth'], n_jobs=n_jobs, random_state=cfg_classifier['seed'],
                                     oob_score=False, class_weight='balanced_subsample')
    # Setup features
    X_data = featdata['X_data']
    Y_data = featdata['Y_data']
    wlen = featdata['wlen']
    
    #if cfg_classifier['wlen'] is None:
        #cfg_classifier['wlen'] = wlen

    # StratifiedShuffleSplit
    ntrials, nsamples, fsize = X_data.shape
    logger.info_green('%d-fold stratified cross-validation with test set ratio %.2f' % (cfg_cv['folds'], cfg_cv['test_ratio']))
    if SKLEARN_OLD:
        cv = StratifiedShuffleSplit(Y_data[:, 0], cfg_cv['folds'], test_size=cfg_cv['test_ratio'])
    else:
        cv = StratifiedShuffleSplit(n_splits=cfg_cv['folds'], test_size=cfg_cv['test_ratio'])
    logger.info('%d trials, %d samples per trial, %d feature dimension' % (ntrials, nsamples, fsize))

    # Do it!
    inv_trig = {v: k for k, v in triggers.items()}    
    scores, cm_txt = crossval_epochs(cv, X_data, Y_data, cls, inv_trig, cfg_cv['balance_samples'], n_jobs=n_jobs,
                                     ignore_thres=cfg_cv['ignore_thres'], decision_thres=cfg_cv['decision_thres'])

    # Compute stats
    cv_mean, cv_std = np.mean(scores), np.std(scores)
    txt = 'Cross-Validation Results'
    txt += '\n- Average CV accuracy over %d epochs (random seed=%s)\n' % (ntrials, cfg_cv['seed'])
    txt += "mean %.3f, std: %.3f\n" % (cv_mean, cv_std)
    txt += 'Classifier: %s, ' % 'RF'
    txt += '%d trees, %s max depth, random state %s\n' % (
            cfg_classifier['trees'], cfg_classifier['depth'], cfg_classifier['seed'])
    if cfg_cv['ignore_thres'] is not None:
        txt += 'Decision threshold: %.2f\n' % cfg_cv['IGNORE_THRES']
    txt += '\n- Confusion Matrix\n' + cm_txt
    logger.info(txt)

    # Export to a file
    if 'export_result' in cfg_cv and cfg_cv['export_result'] is True:
        fout = open(cv_file, 'w')
        fout.write(txt)
        fout.close()


#----------------------------------------------------------------------
def crossval_epochs(cv, epochs_data, labels, cls, label_names=None, do_balance=None, n_jobs=None, ignore_thres=None, decision_thres=None):
    """
    Epoch-based cross-validation used by cross_validate().

    Params
    ======
    cv: scikit-learn cross-validation object
    epochs_data: np.array of shape [epochs x samples x features]
    labels: np.array of shape [epochs x samples]
    cls: classifier
    label_names: associated label names {0:'Left', 1:'Right', ...}
    do_balance: oversample or undersample to match the number of samples among classes

    """

    scores = []
    cnum = 1
    cm_sum = 0
    label_set = np.unique(labels)
    num_labels = len(label_set)
    if label_names is None:
        label_names = {l:'%s' % l for l in label_set}

    if n_jobs is None:
        n_jobs = mp.cpu_count()

    if n_jobs > 1:
        logger.info('crossval_epochs(): Using %d cores' % n_jobs)
        pool = mp.Pool(n_jobs)
        results = []

    # for classifier itself, single core is usually faster
    cls.n_jobs = 1

    if SKLEARN_OLD:
        splits = cv
    else:
        splits = cv.split(epochs_data, labels[:, 0])
    for train, test in splits:
        X_train = np.concatenate(epochs_data[train])
        X_test = np.concatenate(epochs_data[test])
        Y_train = np.concatenate(labels[train])
        Y_test = np.concatenate(labels[test])
        if do_balance:
            X_train, Y_train = balance_samples(X_train, Y_train, do_balance)
            X_test, Y_test = balance_samples(X_test, Y_test, do_balance)

        if n_jobs > 1:
            results.append(pool.apply_async(fit_predict_thres,
                                            [cls, X_train, Y_train, X_test, Y_test, cnum, label_set, ignore_thres, decision_thres]))
        else:
            score, cm = fit_predict_thres(cls, X_train, Y_train, X_test, Y_test, cnum, label_set, ignore_thres, decision_thres)
            scores.append(score)
            cm_sum += cm
        cnum += 1

    if n_jobs > 1:
        pool.close()
        pool.join()

        for r in results:
            score, cm = r.get()
            scores.append(score)
            cm_sum += cm

    # confusion matrix
    cm_sum = cm_sum.astype('float')
    if cm_sum.shape[0] != cm_sum.shape[1]:
        # we have decision thresholding condition
        assert cm_sum.shape[0] < cm_sum.shape[1]
        cm_sum_all = cm_sum
        cm_sum = cm_sum[:, :cm_sum.shape[0]]
        underthres = np.array([r[-1] / sum(r) for r in cm_sum_all])
    else:
        underthres = None

    cm_rate = np.zeros(cm_sum.shape)
    for r_in, r_out in zip(cm_sum, cm_rate):
        rs = sum(r_in)
        if rs > 0:
            r_out[:] = r_in / rs
        else:
            assert min(r) == max(r) == 0
    if underthres is not None:
        cm_rate = np.concatenate((cm_rate, underthres[:, np.newaxis]), axis=1)

    cm_txt = 'Y: ground-truth, X: predicted\n'
    max_chars = 12
    tpl_str = '%%-%ds ' % max_chars
    tpl_float = '%%-%d.2f ' % max_chars
    for l in label_set:
        cm_txt += tpl_str % label_names[l][:max_chars]
    if underthres is not None:
        cm_txt += tpl_str % 'Ignored'
    cm_txt += '\n'
    for r in cm_rate:
        for c in r:
            cm_txt += tpl_float % c
        cm_txt += '\n'
    cm_txt += 'Average accuracy: %.2f\n' % np.mean(scores)

    return np.array(scores), cm_txt

#----------------------------------------------------------------------
def fit_predict_thres(cls, X_train, Y_train, X_test, Y_test, cnum, label_list, ignore_thres=None, decision_thres=None):
    """
    Any likelihood lower than a threshold is not counted as classification score

    Params
    ======
    ignore_thres = if not None or larger than 0, likelihood values lower than
    ignore_thres will be ignored while computing confusion matrix.
    """
    cls.fit(X_train, Y_train)
    assert ignore_thres is None or ignore_thres >= 0
    if ignore_thres is None or ignore_thres == 0:
        Y_pred = cls.predict(X_test)
        score = skmetrics.accuracy_score(Y_test, Y_pred)
        cm = skmetrics.confusion_matrix(Y_test, Y_pred, label_list)
    else:
        if decision_thres is not None:
            logger.error('decision threshold and ignore_thres cannot be set at the same time.')
            raise ValueError
        Y_pred = cls.predict_proba(X_test)
        Y_pred_labels = np.argmax(Y_pred, axis=1)
        Y_pred_maxes = np.array([x[i] for i, x in zip(Y_pred_labels, Y_pred)])
        Y_index_overthres = np.where(Y_pred_maxes >= ignore_thres)[0]
        Y_index_underthres = np.where(Y_pred_maxes < ignore_thres)[0]
        Y_pred_overthres = np.array([cls.classes_[x] for x in Y_pred_labels[Y_index_overthres]])
        Y_pred_underthres = np.array([cls.classes_[x] for x in Y_pred_labels[Y_index_underthres]])
        Y_pred_underthres_count = np.array([np.count_nonzero(Y_pred_underthres == c) for c in label_list])
        Y_test_overthres = Y_test[Y_index_overthres]
        score = skmetrics.accuracy_score(Y_test_overthres, Y_pred_overthres)
        cm = skmetrics.confusion_matrix(Y_test_overthres, Y_pred_overthres, label_list)
        cm = np.concatenate((cm, Y_pred_underthres_count[:, np.newaxis]), axis=1)

    return score, cm

#----------------------------------------------------------------------
def get_predict_proba(cls, X_train, Y_train, X_test, Y_test, cnum):
    """
    All likelihoods will be collected from every fold of a cross-validaiton. Based on these likelihoods,
    a threshold will be computed that will balance the true positive rate of each class.
    Available with binary classification scenario only.
    """
    cls.fit(X_train, Y_train)
    Y_pred = cls.predict_proba(X_test)
    
    return Y_pred[:,0]

#----------------------------------------------------------------------
def balance_samples(X, Y, balance_type, verbose=False):
    if balance_type == 'OVER':
        """
        Oversample from classes that lack samples
        """
        label_set = np.unique(Y)
        max_set = []
        X_balanced = np.array(X)
        Y_balanced = np.array(Y)

        # find a class with maximum number of samples
        for c in label_set:
            yl = np.where(Y == c)[0]
            if len(max_set) == 0 or len(yl) > max_set[1]:
                max_set = [c, len(yl)]
        for c in label_set:
            if c == max_set[0]: continue
            yl = np.where(Y == c)[0]
            extra_samples = max_set[1] - len(yl)
            extra_idx = np.random.choice(yl, extra_samples)
            X_balanced = np.append(X_balanced, X[extra_idx], axis=0)
            Y_balanced = np.append(Y_balanced, Y[extra_idx], axis=0)
    elif balance_type == 'UNDER':
        """
        Undersample from classes that are excessive
        """
        label_set = np.unique(Y)
        min_set = []

        # find a class with minimum number of samples
        for c in label_set:
            yl = np.where(Y == c)[0]
            if len(min_set) == 0 or len(yl) < min_set[1]:
                min_set = [c, len(yl)]
        yl = np.where(Y == min_set[0])[0]
        X_balanced = np.array(X[yl])
        Y_balanced = np.array(Y[yl])
        for c in label_set:
            if c == min_set[0]: continue
            yl = np.where(Y == c)[0]
            reduced_idx = np.random.choice(yl, min_set[1])
            X_balanced = np.append(X_balanced, X[reduced_idx], axis=0)
            Y_balanced = np.append(Y_balanced, Y[reduced_idx], axis=0)
    elif balance_type is None or balance_type is False:
        return X, Y
    else:
        logger.error('Unknown balancing type %s' % balance_type)
        raise ValueError

    logger.info_green('\nNumber of samples after %ssampling' % balance_type.lower())
    for c in label_set:
        logger.info('%s: %d -> %d' % (c, len(np.where(Y == c)[0]), len(np.where(Y_balanced == c)[0])))

    return X_balanced, Y_balanced

#----------------------------------------------------------------------
def train_decoder(cfg_classifier, PSD_params, featdata, n_jobs=1, feat_file=None):
    """
    Train the final decoder using all data
    """
    # Init a classifier
    cls = RandomForestClassifier(n_estimators=cfg_classifier['trees'], max_features='auto',
                                     max_depth=cfg_classifier['depth'], n_jobs=n_jobs, random_state=cfg_classifier['seed'],
                                     oob_score=False, class_weight='balanced_subsample')    

    # Setup features
    X_data = featdata['X_data']
    Y_data = featdata['Y_data']
    wlen = featdata['wlen']
    if PSD_params['wlen'] is None:
        PSD_params['wlen'] = wlen
    ch_names = featdata['ch_names']
    X_data_merged = np.concatenate(X_data)
    Y_data_merged = np.concatenate(Y_data)
   
    # Start training the decoder
    logger.info_green('Training the decoder')
    cls.n_jobs = n_jobs
    cls.fit(X_data_merged, Y_data_merged)
    cls.n_jobs = 1 # always set n_jobs=1 for testing

    # Export the decoder
    data = dict(cls=cls, ch_names=ch_names, psde=featdata['psde'], picks=featdata['picks'])
    clsfile = '%s/classifier-%s.pkl' % (os.path.dirname(os.path.realpath(__file__)), platform.architecture()[0])
    save_obj(clsfile, data)
    logger.info('Decoder saved to %s' % clsfile)

    # Reverse-lookup frequency from FFT
    fq = 0
    if type(PSD_params['wlen']) == list:
        fq_res = 1.0 / PSD_params['wlen'][0]
    else:
        fq_res = 1.0 / PSD_params['wlen']
    fqlist = []
    while fq <= PSD_params['fmax']:
        if fq >= PSD_params['fmin']:
            fqlist.append(fq)
        fq += fq_res

    # Show top distinctive features

    logger.info_green('Good features ordered by importance')
    keys, values = sort_by_value(list(cls.feature_importances_), rev=True)
    keys = np.array(keys)
    values = np.array(values)

    gfout = open('%s/good_features.txt' %os.path.dirname(os.path.realpath(__file__)), 'w')
    
    if type(wlen) is not list and featdata['picks'] is not None:
        ch_names = [ch_names[c] for c in featdata['picks']]

    chlist, hzlist = feature2chz(keys, fqlist, ch_names=ch_names)
    FEAT_TOPN = 100
    valnorm = values[:FEAT_TOPN].copy()
    valsum = np.sum(valnorm)
    if valsum == 0:
        valsum = 1
    valnorm = valnorm / valsum * 100.0

    # show top-N features
    for i, (ch, hz) in enumerate(zip(chlist, hzlist)):
        if i >= FEAT_TOPN:
            break
        txt = '%-3s %5.1f Hz  normalized importance %-6s  raw importance %-6s  feature %-5d' %\
              (ch, hz, '%.2f%%' % valnorm[i], '%.2f%%' % (values[i] * 100.0), keys[i])
        logger.info(txt)

    gfout.write('Importance(%) Channel Frequency Index\n')
    for i, (ch, hz) in enumerate(zip(chlist, hzlist)):
        gfout.write('%.3f\t%s\t%s\t%d\n' % (values[i]*100.0, ch, hz, keys[i]))
    gfout.close()