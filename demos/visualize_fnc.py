import os
import matlab.engine
print("starting matlab engine")
eng = matlab.engine.start_matlab()
print("engine started")
from matkmeans import MKMeans
import torch
import numpy as np
import sklearn.metrics as skm
import sklearn.cluster as skc
import matplotlib
try: 
   matplotlib.use('TkAgg')
except Exception:
   print("Cant use TkAgg")
   pass
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats.stats import pearsonr
# from mutual_info import mutual_information_2d
from npeet import entropy_estimators as ee
from scipy.stats import ttest_ind as ttest
import scipy.io as sio
from vfnc_utils import local_maxima
from itertools import combinations
from scipy.spatial.distance import cdist
import copy
CORTEXPATH = "/export/mialab/users/bbaker/projects/cortex/"
DATASET = "iris"
NAME = "MyClassifier"
NETNAME = "classifier"
ISCLASSIFIER = True
KMIN = 2
KMAX = 30
EPOCHS = 200
WINSIZE = 32
FILE_FORM = "%s_%d.t7"
BATCH = 64
FILE_FOLDER = os.path.join(CORTEXPATH, "results/%s/binaries/" % DATASET)
DATAPATH = os.path.join(CORTEXPATH, "data", DATASET)
train_fp = os.path.join(DATAPATH, "%s_train.npy" % DATASET)
test_fp = os.path.join(DATAPATH, "%s_test.npy" % DATASET)
train_fpl = os.path.join(DATAPATH, "%s_target_train.npy" % DATASET)
test_fpl = os.path.join(DATAPATH, "%s_target_test.npy" % DATASET)

GRAB_MODELS = {"hidden4": 2,
               "output": 4 }


def corr(A, B):
    return np.corrcoef(A)


METRIC = corr
AGG = np.median

train_data = np.load(train_fp)
train_label = np.load(train_fpl)
train_label_expanded = []
test_data = np.load(test_fp)
test_label = np.load(test_fpl)
test_label_expaned = []
num_subj = len(train_data)
num_train_states = int(np.ceil(len(train_data) / BATCH))

gathered_train = {grab: [] for grab in GRAB_MODELS.keys()}
gathered_test = {grab: [] for grab in GRAB_MODELS.keys()}
gathered_train["all"] = []
gathered_test["all"] = []
LAYER_SIZES = {}
all_train_acc = []

# Begin gathering data over all Epochs
for e in range(1, EPOCHS):
    print("On epoch %d" % e)
    filepath = os.path.join(FILE_FOLDER, FILE_FORM % (NAME, e))
    binary = torch.load(filepath)
    summary = binary.get("summary")
    train_loss = summary.get("train").get("losses").get(NETNAME)
    train_acc = summary.get("train").get("%s_accuracy" % NAME)[-1] if ISCLASSIFIER else 0
    all_train_acc.append(train_acc)
    test_loss = summary.get("test").get("losses").get(NETNAME)
    test_acc = summary.get("test").get("%s_accuracy" % NAME)[-1] if ISCLASSIFIER else 0
    all_states = binary.get("nets").get(NETNAME).all_states
    train_states, test_state = all_states[:num_train_states], all_states[-1]
    train_grab = {}
    test_grab = {}
    trall = None
    teall = None
    for grab, layer in GRAB_MODELS.items():
        tr = np.vstack([tr[layer].cpu().detach().numpy() for tr in train_states])
        if grab == 'output':
            tr = np.maximum(tr, 0)
        LAYER_SIZES[grab] = tr.shape[1]
        te = test_state[layer].cpu().detach().numpy()
        if trall is None:
            trall = tr
        else:
            trall = np.hstack([trall, tr])
        if teall is None:
            teall = te
        else:
            teall = np.hstack([teall, te])
        gathered_train[grab].append(tr)
        gathered_test[grab].append(te)
    grab = "all"
    gathered_train[grab].append(trall)
    gathered_test[grab].append(teall)

LAYER_SIZES['all'] = sum(list(LAYER_SIZES.values()))
# End gathering data over all epochs

# Begin measuring windowed metrics
X = {}
Xact = {}
Xsact = {}
Xs = {}
Y = {}
Ys = {}
train_label_expanded = {}
test_label_expanded = {}
X_ex = {}
inds = []
all_subjects_indices = {}
all_subject_cov = {} # [np.zeros((EPOCHS - WINSIZE,))] * num_subj
triu_inds = {}
    
for t in range(EPOCHS - WINSIZE - 1):
    print("On window %s" % t)
    for grab in list(GRAB_MODELS.keys()) + ['all']:
        if grab not in X.keys():
            all_subjects_indices[grab] = []
            X[grab] = []
            Xact[grab] = []
            Xsact[grab] = []
            train_label_expanded[grab] = []
            Xs[grab] = []
            all_subject_cov[grab] = [np.zeros((EPOCHS - WINSIZE -1,))] * num_subj
        triu_inds[grab] = np.triu_indices(LAYER_SIZES[grab], 1)
        window = gathered_train[grab][t:(t + WINSIZE)]
        subjects = []
        for s in range(num_subj):
            subjects.append(np.zeros((LAYER_SIZES[grab], WINSIZE)))
        last_batch_end = 0
        for epoch, batches in enumerate(window):
            for subject_index in range(batches.shape[0]):
                subjects[subject_index][:,epoch] = batches[subject_index,:]
            last_batch_end += batches.shape[0]
        for subject_index in range(num_subj):
            C = METRIC(subjects[subject_index], subjects[subject_index])
            if len(Xs[grab]) <= subject_index:
                Xs[grab].append([])
                Xsact[grab].append([])
            if np.isnan(C).any():
                C = np.nan_to_num(C)
            X[grab].append(C[triu_inds[grab]])
            Xact[grab].append(subjects[subject_index])
            Xs[grab][subject_index].append(C[triu_inds[grab]])
            Xsact[grab][subject_index].append(subjects[subject_index])
            train_label_expanded[grab].append(train_label[subject_index])
            all_subjects_indices[grab].append(subject_index)
            all_subject_cov[grab][subject_index][t] = np.var(subjects[subject_index])
# End measuring windowed metrics
# Exemplars
do_exemplars = True
for grab in list(GRAB_MODELS.keys()) + ['all']:
    if do_exemplars:
        if grab not in X_ex.keys():
            X_ex[grab] = []
        for subject_index in range(num_subj):
            mm, LM = local_maxima(np.array(all_subject_cov[grab][subject_index]))
            X_ex[grab] += [Xs[grab][subject_index][lm] for lm in LM]
    else:
        X_ex[grab] = X[grab]
# End Exemplars

plt.set_cmap("jet")
C = {}
if not os.path.exists(DATASET):
    os.makedirs(DATASET)
plt.close('all')
if not os.path.exists(os.path.join(DATASET, 'groups')):
    os.makedirs(os.path.join(DATASET, "groups"))

for grab in list(GRAB_MODELS.keys()) + ['all']:

    # Silhouette Score evaluation
    silhouettes = []
    km = []
    for k in range(KMIN, KMAX + 1):
        print("Evaluating %s exemplars at k=%d" % (grab, k))
        #kmeans = skc.KMeans(n_clusters=k, random_state=0, n_jobs=10, n_init=10).fit(np.array(X_ex[grab]))
        kmeans = MKMeans(eng, n_clusters=k, n_init=10).fit(np.array(X_ex[grab]))
        silhouette = sum(np.min(np.nan_to_num(cdist(np.array(X_ex[grab]), kmeans.cluster_centers_, 'correlation'), np.inf), axis=1))
# skm.silhouette_score(np.array(X_ex[grab]), kmeans.labels_)
        print("silhouette %f, k=%d" % (silhouette, k))
        silhouettes.append(silhouette)
        km.append(kmeans)
        #best_sil = 0
        #best_kmeans = None
        #for n_iter in range(10):
        #    kmeans = nkm.Kmeans(np.array(X_ex[grab]), k=k, metric=np.correlate)
        #    kmeans.cluster_centers_ = kmeans.centres
        #    kmeans.labels_ = kmeans.Xtocentre
        #    silhouette = skm.silhouette_score(np.array(X_ex[grab]), kmeans.labels_)
        #    if silhouette > best_sil:
        #        best_kmeans = kmeans
        #        best_sil = silhouette
        #print("silhouette %f, k=%d" % (best_sil, k))
        #silhouettes.append(best_sil)
        #km.append(best_kmeans)
    # best = silhouettes.index(np.max(silhouettes))
    plt.plot(range(KMIN, KMAX + 1), silhouettes, lw=2)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.savefig('%s/%s_%s_silhouettes.png' % (DATASET, DATASET, grab), bbox_inches='tight')
    plt.show()
    input_k = input('k=?')
    k = int(input_k)
    best = list(range(KMIN,KMAX+1)).index(k)
    print("Chose k=%d" % k)
    plt.close('all')



    exemplar_centroids = km[best].cluster_centers_
    k = list(range(KMIN, KMAX + 1))[best]
    print("Best silhouette %f, k=%d" % (np.max(silhouettes), k))
    if do_exemplars:
        print("Performing KMeans on full data")
        #kmeans = skc.KMeans(n_clusters=k, random_state=0, n_jobs=10, init=exemplar_centroids, n_init=1).fit(np.array(X[grab]))
        kmeans = MKMeans(eng, n_clusters=k, n_init=1, init=exemplar_centroids).fit(np.array(X[grab]))
        # kmeans = MKmeans(n_clusters=k, init=exemplar_centroids, n_init=1).fit(np.array(X[grab]))
        # kmeans = nkm.Kmeans(np.array(X[grab]), centres=exemplar_centroids)
        # kmeans.cluster_centers_ = kmeans.centres
        # kmeans.labels_ = kmeans.Xtocentre
    else:
        kmeans = km[best]
    # End silhouette score evaluation

    # Subject labelling
    Y[grab] = kmeans.labels_
    Ys[grab] = []
    for subject_index, cluster_label in zip(all_subjects_indices[grab], Y[grab]):
         if len(Ys[grab]) <= subject_index:
             Ys[grab].append([])
         Ys[grab][subject_index].append(cluster_label)
    # End Subject labelling

    # Centroid plotting
    print("Plotting")
    C[grab] = []
    for i in range(k):
        C[grab].append(np.zeros((LAYER_SIZES[grab], LAYER_SIZES[grab])))
        C[grab][i][triu_inds[grab]] = kmeans.cluster_centers_[i]
        C[grab][i] = C[grab][i].T
        C[grab][i][triu_inds[grab]] = kmeans.cluster_centers_[i]
        fig, ax = plt.subplots()
        img = ax.imshow(C[grab][i], vmin=-1, vmax=1, interpolation=None)

        ax.autoscale(False)
        ax.set_xticks(range(LAYER_SIZES[grab]))
        ax.set_yticks(range(LAYER_SIZES[grab]))
        plt.colorbar(img, ax=ax)
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        if grab == 'all':
            last_bound = 0
            for grab2 in GRAB_MODELS.keys():
                if grab2 == 'all':
                    continue
                bound = last_bound + LAYER_SIZES[grab2]
                plt.plot([bound - 0.5, bound - 0.5], [-0.5, LAYER_SIZES['all'] + 0.5], 'k-', lw=1.5)  # Vertical
                plt.plot([-0.5, LAYER_SIZES['all'] + 0.5], [bound - 0.5, bound - 0.5], 'k-', lw=1.5)  # Horizontal
                last_bound = bound
        plt.savefig("%s/%s_%s_centroid_%d.png" % (DATASET, DATASET, grab, i), bbox_inches='tight')
        plt.close('all')
    # End centroid plotting

    # Begin group plotting
    medians = []
    groups = []
    for label in np.unique(train_label):
        # Group centroid plotting
        group = [x for y, x in zip(train_label_expanded[grab], X[grab]) if y == label]
        groups.append([])
        group_k_labels = [l for y, l in zip(train_label_expanded[grab], Y[grab]) if y == label]
        medians.append([])
        group_act = [x for y, x in zip(train_label_expanded[grab], Xact[grab]) if y == label]
        for k_i in range(k):
            group_k = [x for x, x_k in zip(group, group_k_labels) if x_k == k_i]
            group_k_act = [x for x, x_k in zip(group_act, group_k_labels) if x_k == k_i]
            groups[label].append(group_k)
            median_data = np.median(np.vstack(group_k), 0)
            median = np.zeros((LAYER_SIZES[grab], LAYER_SIZES[grab]))
            median[triu_inds[grab]] = median_data
            median = median.T
            median[triu_inds[grab]] = median_data
            medians[label].append(copy.deepcopy(median))
            median_act = np.median(np.hstack(group_k_act), 1).reshape(1,LAYER_SIZES[grab])

            fig, ax = plt.subplots()
            img = ax.imshow(medians[label][k_i], vmin=-1, vmax=1, interpolation=None)

            ax.autoscale(False)
            ax.set_xticks(range(LAYER_SIZES[grab]))
            ax.set_yticks(range(LAYER_SIZES[grab]))
            plt.colorbar(img, ax=ax)
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            if grab == 'all':
                last_bound = 0
                for grab2 in GRAB_MODELS.keys():
                    if grab2 == 'all':
                        continue
                    bound = last_bound + LAYER_SIZES[grab2]
                    plt.plot([bound - 0.5, bound - 0.5], [-0.5, LAYER_SIZES['all'] + 0.5], 'k-', lw=1.5)  # Vertical
                    plt.plot([-0.5, LAYER_SIZES['all'] + 0.5], [bound - 0.5, bound - 0.5], 'k-', lw=1.5)  # Horizontal
                    last_bound = bound
            plt.savefig("%s/groups/%s_group_%s_%s_centroid_%s.png" %
                        (DATASET, DATASET, label, grab, k_i), bbox_inches='tight')
            plt.close('all')
            fig, ax = plt.subplots()
            img = ax.imshow(median_act,vmin=0,vmax=1, interpolation=None)

            ax.autoscale(False)
            ax.set_xticks(range(LAYER_SIZES[grab]))
            ax.set_yticks(range(1))
            plt.colorbar(img, ax=ax)
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            if grab == 'all':
                last_bound = 0
                for grab2 in GRAB_MODELS.keys():
                    if grab2 == 'all':
                        continue
                    bound = last_bound + LAYER_SIZES[grab2]
                    plt.plot([bound - 0.5, bound - 0.5], [-0.5, LAYER_SIZES['all'] + 0.5], 'k-', lw=1.5)  # Vertical
                    plt.plot([-0.5, LAYER_SIZES['all'] + 0.5], [bound - 0.5, bound - 0.5], 'k-', lw=1.5)  # Horizontal
                    last_bound = bound
            plt.savefig("%s/groups/%s_group_%s_%s_activations_%s.png" %
                        (DATASET, DATASET, label, grab, k_i), bbox_inches='tight')
            plt.close('all')
        # End Group centroid plotting
        # Group Transition plotting
        
        group_s = [x for y, x in zip(train_label, Xs[grab]) if y == label]
        group_k_labels = [l for y, l in zip(train_label, Ys[grab]) if y == label]
        median_transitions = np.median(np.vstack(group_k_labels),0)
        plt.plot(median_transitions, lw=2)
        plt.title("Group %s State Transitions" % label)
        plt.ylabel("State")
        plt.xlabel("Epoch")
        plt.savefig("%s/groups/%s_group_%s_%s_states.png" %
                    (DATASET, DATASET, label, grab), bbox_inches='tight')
        plt.close('all')
        
        # End Group Transition plotting
    
    # Begin Group differences
    comparisons = combinations(np.unique(train_label), 2)
    for comp in comparisons:
        for k_i in range(k):
            A = medians[comp[0]][k_i]
            B = medians[comp[1]][k_i]
            diff = A - B
            fig, ax = plt.subplots()
            img = ax.imshow(diff, interpolation=None)

            ax.autoscale(False)
            ax.set_xticks(range(LAYER_SIZES[grab]))
            ax.set_yticks(range(LAYER_SIZES[grab]))
            plt.colorbar(img, ax=ax)
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            if grab == 'all':
                last_bound = 0
                for grab2 in GRAB_MODELS.keys():
                    if grab2 == 'all':
                        continue
                    bound = last_bound + LAYER_SIZES[grab2]
                    plt.plot([bound - 0.5, bound - 0.5], [-0.5, LAYER_SIZES['all'] + 0.5], 'k-', lw=1.5)  # Vertical
                    plt.plot([-0.5, LAYER_SIZES['all'] + 0.5], [bound - 0.5, bound - 0.5], 'k-', lw=1.5)  # Horizontal
                    last_bound = bound
            plt.savefig("%s/groups/%s_diffs_groups_%s_%s_%s_centroid_%s.png" %
                        (DATASET, DATASET, comp[0], comp[1], grab, k_i), bbox_inches='tight')
            plt.close('all')
            ttest_thresh = 0.05
            A = groups[comp[0]][k_i]
            B = groups[comp[1]][k_i]
            ttest_val = ttest(A, B)
            diff_data = ttest_val.pvalue
            diff_data[diff_data > ttest_thresh] = np.nan
            diff = np.zeros((LAYER_SIZES[grab], LAYER_SIZES[grab]))
            diff[triu_inds[grab]] = diff_data
            diff = diff.T
            diff[triu_inds[grab]] = diff_data
            fig, ax = plt.subplots()
            img = ax.imshow(diff, interpolation=None)

            ax.autoscale(False)
            ax.set_xticks(range(LAYER_SIZES[grab]))
            ax.set_yticks(range(LAYER_SIZES[grab]))
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            plt.colorbar(img, ax=ax)
            if grab == 'all':
                last_bound = 0
                for grab2 in GRAB_MODELS.keys():
                    if grab2 == 'all':
                        continue
                    bound = last_bound + LAYER_SIZES[grab2]
                    plt.plot([bound - 0.5, bound - 0.5], [-0.5, LAYER_SIZES['all'] + 0.5], 'k-', lw=1.5)  # Vertical
                    plt.plot([-0.5, LAYER_SIZES['all'] + 0.5], [bound - 0.5, bound - 0.5], 'k-', lw=1.5)  # Horizontal
                    last_bound = bound
            plt.savefig("%s/groups/%s_ttest_groups_%s_%s_%s_centroid_%s.png" %
                        (DATASET, DATASET, comp[0], comp[1], grab, k_i), bbox_inches='tight')
            plt.close('all')
            
    # End Group differences
    

    # End group plotting

    # Transition plotting
    median_states = np.median(np.vstack(Ys[grab]), 0)
    plt.plot(median_states, lw=2)
    plt.title("State Transitions")
    plt.xlabel("State")
    plt.ylabel("Epoch")
    plt.savefig("%s/%s_%s_states.png" % (DATASET, DATASET, grab))
    plt.close('all')
    # End transition plotting


plt.plot(train_loss, lw=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("%s/%s_loss.png" % (DATASET, DATASET))
plt.close('all')
plt.plot(all_train_acc, lw=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("%s/%s_acc.png" % (DATASET, DATASET))
plt.close('all')

eng.quit()

#sio.savemat("%s/%s_results.mat" % (DATASET, DATASET), 
#{"medians":medians,
#"groups":groups,
#"Ys": Ys,
#"Xs": Xs}
#)
