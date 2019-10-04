import os
import json
import cv2

from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self, data_dir, results_dir, preprocessing_fn=None):

        self.data_dir = data_dir
        self.results_dir = results_dir
        self.preprocessing_fn = preprocessing_fn

        if 'info.json' in os.listdir(self.results_dir):
            self._loadInfo()
        else:
            self._createInfo()

    def _loadInfo(self):
        with open(f'{self.results_dir}info.json', 'r') as read_file:
            self.info = json.load(read_file)
        
        self.updateInfo()
    
    def _createInfo(self):
        path_to_images = [os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir) if (os.path.isfile(os.path.join(self.data_dir, file_name)) and not(file_name.startswith ('.')))]
        self.info = {
            'path_to_image': path_to_images,
            'path_to_result': [None for x in range(len(path_to_images))],
            'classes': [None for x in range(len(path_to_images))],
            'bn_boxes': [None for x in range(len(path_to_images))],
            'flag': [False for x in range(len(path_to_images))]
        }
    
    def updateInfo(self):
        path_to_images = [os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir) if (os.path.isfile(os.path.join(self.data_dir, file_name)) and not(file_name.startswith ('.')))]
        
        #add new files
        path_new_images = list(set(path_to_images) - set(self.info['path_to_image']))
        for new_image in path_new_images:
            self.info['path_to_image'].append(new_image)
            self.info['flag'].append(False)
            for key in self.info.keys():
                if (key != 'path_to_image') & (key != 'flag'):
                    self.info[key].append(None)

        #remove delete files
        del_files = list(set(self.info['path_to_image']) - set(path_to_images))
        for del_image in del_files:
            os.remove(os.path.join(self.results_dir, f'result_{os.path.basename(del_image)}'))
        for del_image in del_files:
            idx_del = self.info['path_to_image'].index(del_image)
            for key in self.info.keys():
                _ = self.info[key].pop(idx_del)
    
    def save_to_json(self):
        with open(f'{self.results_dir}info.json', 'w') as write_file:
            json.dump(self.info, write_file)

    def writeInfo(self, **kwargs):
        for key, items in kwargs.items():
            try:
                if key != 'idx':
                    self.info[key][kwargs['idx']] = items
            except KeyError:
                print('Written uncorrected value of key')
    
    def __len__(self):
        return len(self.info['path_to_image'])
    
    def __iter__(self):
        try:
            self.idx = self.info['flag'].index(False)
        except ValueError:
            self.idx = len(self)
        return self

    def __next__(self):
        if self.idx >= len(self):
                raise StopIteration

        return_value = self[self.idx] 
        self.idx += 1
        return return_value

    def __getitem__(self, idx):
        """Return image with index idx in self.info

        Args:
            idx: image's index that must return
        """
        return_image = cv2.imread(self.info['path_to_image'][idx])
        original_size = return_image.shape

        if self.preprocessing_fn:
            return idx, return_image, self.preprocessing_fn(return_image)

        return idx, return_image

    