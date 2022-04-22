import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def rgb2gray(rgb):

    if(len(rgb.shape) == 2):return rgb
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = config.both_transforms(image=image)["image"]
        image = rgb2gray(image)
        high_res = config.highres_transform(image=image)["image"]
        #print(high_res.shape)
        low_res = config.lowres_transform(image=image)["image"]
        #print(low_res.shape)
        if(len(high_res.shape) != 3):
            low_res = np.reshape(low_res,(1,low_res.shape[0],low_res.shape[1],low_res.shape[2]))
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="dataset/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)
    # for low_res,high_res in loader:
    #     print(low_res.shape)
    #     print(high_res.shape)
    print(loader.dataset.__getitem__(2)[0].shape)


if __name__ == "__main__":
    test()