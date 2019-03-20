import torch
import pandas
import os
from skimage import io
import numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import Resize, ToTensor

def driver():
    image_data_loader = load_data(250, 1, "train.csv", "/scratch/gd66/spring2019/lab2/kaggleamazon/", 32)

    for i_batch, sample_batched in enumerate(image_data_loader):
        print(i_batch, sample_batched['image_tensor'].size(),
          sample_batched['image_label_tensor'].size())



def load_data(batch_size, workers, csv_file, root_dir, image_dim=32):
    image_transformer =transforms.Compose([Resize(image_dim, image_dim),ToTensor()])
    dataset = ImageDataSet(csv_file, root_dir, image_transformer)

    image_data_loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

    return image_data_loader

class ImageDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_data = pandas.read_csv(root_dir+csv_file) #csv file with file location and labels
        self.root_dir = root_dir    #root dir of image files
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image_location = os.path.join(self.root_dir, "train-jpg",
                                self.image_data.iloc[index, 0])
        image_location = image_location+".jpg"
        image_label_string_list = self.image_data.iloc[index, 1].split(" ")
        image_label_tensor = self.get_image_label_tensor(image_label_string_list)

        image_np_array = io.imread(image_location)
        if self.transform:
            image_np_array = self.transform(image_np_array)

        image_tensor = torch.tensor(image_np_array)
        sample = {'image_tensor': image_tensor, 'image_label_tensor' : image_label_tensor}

        return sample



    def get_image_label_tensor(self, image_label_string_list):
        image_label_list = [0 for x in range(17)]
        labels = list(map(int, image_label_string_list))

        for label in labels:
            image_label_list[label] = 1

        image_label_tensor = torch.tensor(image_label_list)
        return image_label_tensor

