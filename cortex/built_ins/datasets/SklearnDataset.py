'''
Wraps the SKLearn data set generation for creating
synthetic datasets
'''

import os
import ast
import json
from cortex.plugins import DatasetPlugin, register_plugin
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
from .utils import build_transforms

import torch
import torch.utils.data as data
import numpy as np

SKLEARN_FUNCS = dict(
    openml=skd.fetch_openml,
    boston=skd.load_boston,
    iris=skd.load_iris,
    diabetes=skd.load_diabetes,
    digits=skd.load_digits,
    linnerud=skd.load_linnerud,
    wine=skd.load_wine,
    breast_cancer=skd.load_breast_cancer,
    olivetti_faces=skd.fetch_olivetti_faces,
    newsgroups=skd.fetch_20newsgroups_vectorized,
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
    'olivetti_faces',
    'newsgroups',
    'lfw_people',
    'lfw_pairs',
    'covtype',
    'rcv1',
    'kddcup99',
    'california_housing',
    'openml'
]
SKLEARN_CLASSIFICATION = [
    'newsgroups',
    'covtype',
    'kddcup99',
    'lfw_people',
    'lfw_pairs',
    'olivetti_faces',
    'breast_cancer',
    'digits',
    'iris',
    'wine',
    'classification',
    'multilabel_classification',
    'hastie_10_2',
    'gaussian_quantiles',
    'openml'
]
SKLEARN_KWARGS = dict(
    openml=['name', 'version', 'data_id', 'return_X_y'],
    boston=['return_X_y'],
    iris=['return_X_y'],
    diabetes=['return_X_y'],
    digits=['return_X_y', 'n_class'],
    linnerud=['return_X_y'],
    wine=['return_X_y'],
    breast_cancer=['return_X_y'],
    olivetti_faces=['shuffle'],
    olivetti_faces_full=['shuffle'],
    newsgroups=['return_X_y','subset', 'categories', 'shuffle', 'remove'],
    lfw_people=['return_X_y','funneled', 'resize', 'min_faces_per_person', 'color'],
    lfw_people_full=['return_X_y','funneled', 'resize', 'min_faces_per_person', 'color'],
    covtype=['return_X_y', 'shuffle'],
    rcv1=['return_X_y', 'subset', 'shuffle'],
    kddcup99=['return_X_y', 'subset', 'shuffle', 'percent10', ],
    california_housing=['return_X_y'],
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
        if self.name == 'openml' and 'openml_name' in kwargs.keys():
            self.name += '_%s' % kwargs['openml_name']
        self.split = split
        self.savedir = os.path.join(self.root, self.name)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.filename = os.path.join(self.savedir, self.name)
        self.filename_train = self.filename + '_train.npy'
        self.target_filename_train = self.filename + '_target' + '_train.npy'
        self.filename_test = self.filename + '_test.npy'
        self.target_filename_test = self.filename + '_target' + '_test.npy'
        self.train = train
        self.classification = self.name in SKLEARN_CLASSIFICATION
        self.target_dtype = np.int if self.classification else np.float
        self.torch_dtype = torch.long if self.classification else torch.float
        self.target_fmt = '%d' if self.classification else '%f'
        if not self.check_exists():
            load = False
        if not load:
            self.data, self.target = self.generate(**kwargs)
            load = True
        if self.check_exists() and load:
            self.data, self.target = self.prepare(*args)
            super().__init__(self.data, self.target)
        else:
            raise RuntimeError('Sklearn Dataset not found')

    def generate(self, **kwargs):
        valid_kwargs = {}
        try:
            valid_kwargs = {key: val for key,
                            val in kwargs.items() if key in SKLEARN_KWARGS[self.name]}
        except KeyError:
            raise (Exception("Sklearn dataset %s not supported." % self.name))
        if self.name in SKLEARN_DOWNLOADABLE:
            valid_kwargs['download_if_missing'] = True
            valid_kwargs['data_home'] = self.savedir
        if 'return_X_y' in SKLEARN_KWARGS[self.name]:
            valid_kwargs['return_X_y'] = True
        if self.name in ['olivetti_faces']:
            dataset = SKLEARN_FUNCS[self.name](**valid_kwargs)
            X = dataset.data
            y = dataset.target
        elif self.name in ['olivetti_faces_full', 'lfw_people_full']:
            dataset = SKLEARN_FUNCS[self.name.replace('_full','')](**valid_kwargs)
            X = dataset.images
            y = dataset.target
        else:
            X, y = SKLEARN_FUNCS[self.name](**valid_kwargs)
        if self.classification:
            _,y_new = np.unique(y,return_inverse=True)
            y = y_new.reshape(y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split)
        
        if self.save:
            np.save(self.filename_train, X_train)
            np.save(self.target_filename_train, y_train)
            np.save(self.filename_test, X_test)
            np.save(self.target_filename_test, y_test)

        return X, y

    def prepare(self, *args):
        if self.train:
            return (torch.tensor(np.load(self.filename_train),dtype=torch.float),
                    torch.tensor(np.load(self.target_filename_train), dtype=self.torch_dtype))
        else:
            return (torch.tensor(np.load(self.filename_test),dtype=torch.float),
                    torch.tensor(np.load(self.target_filename_test), dtype=self.torch_dtype))

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
        'sklearn_olivetti_faces_full',
        'sklearn_newsgroups',
        'sklearn_lfw_people',
        'sklearn_lfw_people_full',
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
        'sklearn_swiss_roll',
        'sklearn_openml'
    ]

    def handle(self, source, copy_to_local=False, normalize=True,
               convolutions=False, **kwargs):
        Dataset = self.make_indexing(_SklearnDataset)
        sklearn_kwargs = {}
        if 'sklearn_kwargs' in kwargs.keys():
             if type(kwargs['sklearn_kwargs']) is dict:
                  sklearn_kwargs = kwargs['sklearn_kwargs']
             else:
                 try:
                     sklearn_kwargs=ast.literal_eval(kwargs['sklearn_kwargs'])
                 except Exception as ex1:
                     try:
                         sklearn_kwargs=json.loads(kwargs['sklearn_kwargs'])
                     except Exception as ex2:
                         raise(Exception("Error while parsing SKLEARN kwargs:\n%s\nERROR1:\n%s\nERROR2:\n%s" % (kwargs['sklearn_kwargs'],str(ex1), str(ex2))))
             print("SKLEARN Keyword Args")
             for key, val in sklearn_kwargs.items():
                 print("%s: %s" % (key, val))
        data_path = self.get_path('local')
        name = source.replace('sklearn_', '')
        train_set = Dataset(data_path, train=True, load=False, save=True, name=name,**sklearn_kwargs, **kwargs)
        test_set = Dataset(data_path, train=False, load=True, save=True, name=name,**sklearn_kwargs, **kwargs)
        if name in SKLEARN_CLASSIFICATION:
            dim_l = len(np.unique(train_set.target))
        else:
            dim_l = 1
        if len(train_set[0][0].shape) == 4:
            dim_c, dim_x, dim_y, dim_z = train_set[0][0].shape
            dims = dict(c=dim_c, x=dim_x, y=dim_y, z=dim_z, labels=dim_l)
        elif len(train_set[0][0].shape) == 3:
            dim_c, dim_x, dim_y = train_set[0][0].shape
            dims = dict(c=dim_c, x=dim_x, y=dim_y, labels=dim_l)
        elif len(train_set[0][0].shape) == 2:
            dim_c = 1
            dim_x, dim_y = train_set[0][0].shape
            dims = dict(x=dim_x, y=dim_y, labels=dim_l)
        elif len(train_set[0][0].shape) == 1:
            dim_x = 1
            dim_y = train_set[0][0].shape[0]
            dims = dict( x=dim_x, y=dim_y, labels=dim_l)
        else:
            raise(Exception("Invalid input tensor with %d dims." % len(train_set[0][0].shape)))
        print("Size is %s" % dims)
        input_names = ['images', 'targets', 'index']
        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)


register_plugin(SklearnDatasetPlugin)
