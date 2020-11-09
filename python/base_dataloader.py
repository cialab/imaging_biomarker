import h5py
#import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from datetime import datetime
import random

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, load_data, train, seed, split, data_cache_size=3, transform=False):
        super(HDF5Dataset,self).__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.idx=0
        self.train=train

        # Search for all h5 files
        p = Path(file_path)
        neg = sorted(p.glob('0/*.h5'))
        pos = sorted(p.glob('1/*.h5'))
        random.seed(seed)
        random.shuffle(neg)
        random.shuffle(pos)
        random.seed(datetime.now()) # assuming this makes things non-deterministic again

        # split is the number testing slides per class
        if train:
            neg=neg[0:-int(split)]
            pos=pos[0:-int(split)]
            files=neg+pos
            self.idx=len(neg)   # ngl, this is pretty poor programming
        else:
            neg=neg[-int(split):]
            pos=pos[-int(split):]
            files=neg+pos

        for h5dataset_fp in files:
            a=1
            while a==1:
                try:
                    self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
                    a=0
                except:
                    print('collision. trying again.')
                    a=1
                
    def __getitem__(self, index):
        # get random data; ignore index
        if self.train:
            v=int(round(random.random()))
            if v:
                # get positive bag
                i=random.randint(self.idx,self.__len__()-1)
            else:
                # get negative bag
                i=random.randint(0,self.idx-1)
        # for testing
        else:
            i=index

        x=self.get_data('bag',i)            
        if self.transform:
            x = np.float32(x)
            x = torch.from_numpy(x/127.5 - 1.)
        else:
            x = torch.from_numpy(np.float32(x))

        # get label
        y = self.data_info[i]['label']
        y = torch.from_numpy(np.array(y))
        
        # get mouse number
        n = self.data_info[i]['mousenum']
        
        return (x, y, n)

    def __len__(self):
        return len(self.get_data_infos('bag'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            for dname, ds in h5_file.items():
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(ds.value, file_path)
                
                label=int(file_path.split('/')[-2])
                mousenum=int(file_path.split('/')[-1].split('.')[0])
                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'bag' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx, 'label': label, 'mousenum': mousenum})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for dname, ds in h5_file.items():
                # add data to the data cache and retrieve
                # the cache index
                idx = self._add_to_cache(ds.value, file_path)

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                # the data info should have the same index since we loaded it in the same way
                self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]