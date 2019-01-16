'''
Wraps the SKLearn data set generation for creating
synthetic datasets
'''

import os

from cortex.plugins import DatasetPlugin, register_plugin
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
from .utils import build_transforms

import torch
import torch.utils.data as data
import numpy as np

SKLEARN_FUNCS = dict(
    boston=skd.load_boston,
    iris=skd.load_iris,
    diabetes=skd.load_diabetes,
    digits=skd.load_digits,
    linnerud=skd.load_linnerud,
    wine=skd.load_wine,
    breast_cancer=skd.load_breast_cancer,
    olivetti_faces=skd.fetch_olivetti_faces,
    newsgroups=skd.fetch_20newsgroups,
    lfw_people=skd.fetch_lfw_people,
    covtype=skd.fetch_covtype,
    rcv1=skd.fetch_rcv1,
    kddcup99=skd.fetch_kddcup99,
    california_housing=skd.fetch_california_housing,
    blobs=skd.make_blobs,
    circles=skd.make_circles,
    classification=skd.make_classification,
    friedman1=skd.make_friedman1,
    friedman2=skd.make_friedman2,
    friedman3=skd.make_friedman3,
    gaussian_quantiles=skd.make_gaussian_quantiles,
    hastie_10_2=skd.make_hastie_10_2,
    low_rank_matrix=skd.make_low_rank_matrix,
    moons=skd.make_moons,
    multilabel_classification=skd.make_multilabel_classification,
    regression=skd.make_regression,
    s_curve=skd.make_s_curve,
    sparse_coded_signal=skd.make_sparse_coded_signal,
    sparse_spd_matrix=skd.make_sparse_spd_matrix,
    sparse_uncorrelated=skd.make_sparse_uncorrelated,
    spd_matrix=skd.make_spd_matrix,
    swiss_roll=skd.make_swiss_roll
)
SKLEARN_DOWNLOADABLE = [
    'breast_cancer',
    'olivetti_faces',
    'newsgroups',
    'lfw_people',
    'lfw_pairs',
    'covtype',
    'rcv1',
    'kddcup99',
    'california_housing',
]
SKLEARN_KWARGS = dict(
    boston=[],
    iris=[],
    diabetes=[],
    digits=['n_class'],
    linnerud=[],
    wine=[],
    breast_cancer=[],
    olivetti_faces=['shuffle'],
    newsgroups=['subset', 'categories', 'shuffle', 'remove'],
    lfw_people=['funneled', 'resize', 'min_faces_per_person', 'color'],
    covtype=['shuffle'],
    rcv1=['subset', 'shuffle'],
    kddcup99=['subset', 'shuffle', 'percent10', ],
    california_housing=[],
    biclusters=['noise', 'minval', 'maxval', 'shuffle'],
    blobs=['n_samples', 'n_features', 'centers', 'cluster_std', 'center_box', 'shuffle'],
    checkerboard=['noise', 'minval', 'maxval', 'shuffle'],
    circles=['n_samples', 'shuffle', 'noise', 'factor'],
    classification=['n_samples', 'n_features', 'n_informative', 'n_redundant',
                    'n_repeated', 'n_classes', 'n_clusters_per_class', 'weights', 'flip_y',
                    'class_sep', 'hypercube', 'shift', 'scale', 'shuffle'],
    friedman1=['n_samples', 'n_features', 'noise'],
    friedman2=['n_samples', 'n_features', 'noise'],
    friedman3=['n_samples', 'n_features', 'noise'],
    gaussian_quantiles=['n_samples', 'n_features', 'n_classes', 'mean', 'cov', 'shuffle'],
    hastie_10_2=['n_samples'],
    low_rank_matrix=['n_samples', 'effective_rank', 'tail_strength'],
    moons=['n_samples', 'shuffle', 'noise'],
    multilabel_classification=['n_samples', 'n_features', 'n_classes', 'n_labels',
                               'length', 'allow_unlabeled', 'sparse'],
    regression=['n_samples', 'n_features', 'n_informative', 'n_targets',
                'bias', 'effective_rank', 'tail_strength', 'noise', 'shuffle'],
    s_curve=['n_samples', 'noise', 'random_state'],
    sparse_spd_matrix=['dim', 'alpha', 'norm_diag', 'smallest_coef', 'largest_coef'],
    sparse_uncorrelated=['n_samples', 'n_features'],
    swiss_roll=['n_samples', 'noise']
)
DEFAULT_shape = (10000, 10)
DEFAULT_n_clusters = 2


class _SklearnDataset(data.TensorDataset):

    def __init__(self, root, *args, train=False, split=0.1,
                 load=False, save=False, name='sklearn_dataset', **kwargs):
        self.root = os.path.expanduser(root)
        self.save = save
        self.name = name
        self.split = split
        self.filename = os.path.join(self.root, self.name)
        self.filename_train = self.filename + '_train'
        self.target_filename_train = self.filename + '_target' + '_train'
        self.filename_test = self.filename + '_test'
        self.target_filename_test = self.filename + '_target' + '_test'
        self.train = train
        if not self.check_exists():
            load = False
        if not load:
            self.data, self.target = self.generate(**kwargs)
            load = True
        if self.check_exists() and load:
            self.data, self.target = self.prepare(*args)
            super().__init__(data, target)
        else:
            raise RuntimeError('Sklearn Dataset not found')

    def generate(self, **kwargs):
        valid_kwargs = dict(return_X_y=True)
        try:
            valid_kwargs = {key: val for key,
                            val in kwargs.items() if key in SKLEARN_KWARGS[self.name]}
        except KeyError:
            raise (Exception("Sklearn dataset %s not supported." % self.name))
        if self.name in SKLEARN_DOWNLOADABLE:
            valid_kwargs['download_if_missing'] = False
        X, y = SKLEARN_FUNCS[self.name](**valid_kwargs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split)
        if self.save:
            np.savetxt(self.filename_train + '_train.txt', X_train)
            np.savetxt(self.target_filename_train + '_train.txt', y_train)
            np.savetxt(self.filename_test + '_test.txt', X_test)
            np.savetxt(self.target_filename_test + '_test.txt', y_test)

        return X, y

    def prepare(self, *args):
        if self.train:
            return (torch.Tensor(np.loadtxt(self.filename_train)),
                    torch.Tensor(np.loadtxt(self.target_filename_train)))
        else:
            return (torch.Tensor(np.loadtxt(self.filename_test)),
                    torch.Tensor(np.loadtxt(self.target_filename_test)))

    def check_exists(self):
        return (os.path.exists(self.filename_train) and os.path.exists(self.target_filename_train) and
                os.path.exists(self.filename_test) and os.path.exists(self.target_filename_test))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target
            class.
        """
        label = self.target[index]
        img = self.data[index]

        return np.array(img), label


class SklearnDatasetPlugin(DatasetPlugin):
    sources = [
        'sklearn_boston',
        'sklearn_iris',
        'sklearn_diabetes',
        'sklearn_digits',
        'sklearn_linnerud',
        'sklearn_wine',
        'sklearn_breast_cancer',
        'sklearn_olivetti_faces',
        'sklearn_newsgroups',
        'sklearn_lfw_people',
        'sklearn_lfw_pairs',
        'sklearn_covtype',
        'sklearn_rcv1',
        'sklearn_kddcup99',
        'sklearn_california_housing',
        'sklearn_blobs',
        'sklearn_circles',
        'sklearn_classification',
        'sklearn_friedman1',
        'sklearn_friedman2',
        'sklearn_friedman3',
        'sklearn_gaussian_quantiles',
        'sklearn_hastie_10_2',
        'sklearn_low_rank_matrix',
        'sklearn_moons',
        'sklearn_multilabel_classification',
        'sklearn_regression',
        'sklearn_s_curve',
        'sklearn_sparse_coded_signal',
        'sklearn_sparse_spd_matrix',
        'sklearn_sparse_uncorrelated',
        'sklearn_spd_matrix',
        'sklearn_swiss_roll'
    ]

    def handle(self, source, copy_to_local=False, normalize=True,
               sklearn_kwargs={}, **kwargs):
        Dataset = self.make_indexing(_SklearnDataset)
        data_path = self.get_path('data')
        name = source.replace('sklearn_', '')
        train_set = Dataset(data_path, train=True, load=False, save=True, name=name, **kwargs)
        test_set = Dataset(data_path, train=False, load=True, save=True, name=name, **kwargs)
        input_names = ['images', 'targets', 'index']
        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)


register_plugin(SklearnDatasetPlugin)
