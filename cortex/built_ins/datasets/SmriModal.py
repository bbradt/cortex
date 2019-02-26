from os import path

from cortex.plugins import DatasetPlugin, register_plugin
from cortex.built_ins.datasets.multimodal_dataload import ImageFolder as NII_ImageFolder

import torchvision.transforms as transforms

def unit_interval_normalization(x):
    return (x - x.min()) / (x.max() - x.min())


class SmriModalPlugin(DatasetPlugin):
    sources = ['SmriModal']

    def handle(self, source, copy_to_local=False, **transform_args):
        Dataset = self.make_indexing(NII_ImageFolder)
        data_path = self.get_path(source)

        train_path = path.join(data_path, 'train')
        test_path = path.join(data_path, 'val')

        transform = transforms.Compose([
           # transforms.ToTensor(),
            transforms.Lambda(lambda x: unit_interval_normalization(x)),
          # transforms.ToTensor()
        ])

        train_set = Dataset(root=train_path, transform=transform)
        test_set = Dataset(root=test_path, transform=transform)
        print (len(train_set), len(test_set))
        input_names = ['images', 'targets', 'index']
        print(train_set[0][0].shape)
        dim_y= train_set[0][0].size
        # print(dim_c, dim_x, dim_y, dim_z)
        print (train_set[0][0].min(), train_set[0][0].max())
        dim_l = len(train_set.classes)
    
        dims = dict(x=1,y=dim_y, labels=dim_l)

        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        self.set_scale((0, 1))

register_plugin(SmriModalPlugin)
