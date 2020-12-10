import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler
from torchvision import transforms, datasets
import numpy as np
import random
import torchvision

def sample_data(p, n=100):
  return np.random.multivariate_normal(p, np.array([[0.002, 0], [0, 0.002]]), n)


def toy_DataLoder(n=100, batch_size=100, gaussian_num=5, shuffle=True):
    if gaussian_num == 5:
        d = np.linspace(0, 360, 6)[:-1]
        x = np.sin(d / 180. * np.pi)
        y = np.cos(d / 180. * np.pi)
        points = np.vstack((y, x)).T
        s0 = sample_data(points[0], n).astype(np.float)
        s1 = sample_data(points[1], n).astype(np.float)
        s2 = sample_data(points[2], n).astype(np.float)
        s3 = sample_data(points[3], n).astype(np.float)
        s4 = sample_data(points[4], n).astype(np.float)
        samples = np.vstack((s0, s1, s2, s3, s4))
    elif gaussian_num == 2:
        s0 = sample_data([1, 0], n).astype(np.float)
        s1 = sample_data([-1, 0], n).astype(np.float)
        samples = np.vstack((s0, s1))
    elif gaussian_num == 25:
        samples = np.empty((0, 2))
        for x in range(-2, 3, 1):
            for y in range(-2, 3, 1):
                samples = np.vstack((samples, sample_data([x, y], n).astype(np.float)))
    permutation = np.arange(gaussian_num * n)
    np.random.shuffle(permutation)
    samples = samples[permutation]
    data = TensorDataset(torch.from_numpy(samples))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return dataloader


def _noise_adder(img):
    return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img

def mnist_DataLoder(image_size, batch_size=128, shuffle=True, train=True, balancedbatch=False):
    root = '/data01/tf6/DATA/mnist/'
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        _noise_adder,
    ])
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if balancedbatch:
        train_loader = torch.utils.data.DataLoader(dataset,
                               batch_size=batch_size, shuffle=False, drop_last=True,
                               sampler=BalancedBatchSampler(dataset))
    else:
        train_loader = torch.utils.data.DataLoader(dataset, num_workers=8,
                                batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return train_loader

def imbalancedmnist_DataLoader(image_size, batch_size, shuffle=True, train=True):
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        _noise_adder,
    ])
    imbal_class_prop = np.array([0, 0, 0, 0, 0, 0.2, 0, 1, 0, 0])
    train_dataset_imbalanced = ImbalancedMNIST(
        imbal_class_prop, root='/data01/tf6/DATA/mnist/', train=train, download=True, transform=transform)

    _, train_class_counts = train_dataset_imbalanced.get_labels_and_class_counts()

    # Create new DataLoaders, since the datasets have changed
    train_loader_imbalanced = torch.utils.data.DataLoader(
        dataset=train_dataset_imbalanced, batch_size=batch_size, shuffle = shuffle, num_workers=8, drop_last = True)
    return train_loader_imbalanced


def cifar_DataLoder(image_size, batch_size=128, shuffle=True, train=True, balancedbatch=False):
    root = '/data01/tf6/DATA/cifar10/'
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        _noise_adder,
    ])
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if balancedbatch:
        train_loader = torch.utils.data.DataLoader(dataset,
                               batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8,
                               sampler=BalancedBatchSampler(dataset))
    else:
        train_loader = torch.utils.data.DataLoader(dataset,
                                batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=True)
    return train_loader


def stl_DataLoder(image_size, batch_size=128, shuffle=True):
    root = '/data01/tf6/DATA/stl10/'
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        _noise_adder,
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.STL10(root=root, split='unlabeled', download=True, transform=transform),
        batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=True)
    return train_loader


def lsunDataLoder(image_size, batch_size=128, shuffle=True, category='bedroom_train'):
    dataset = datasets.LSUN(root='/data01/tf6/DATA', classes=[category],
                            transform=transforms.Compose([
                                transforms.CenterCrop(image_size),
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                _noise_adder,
                            ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=True)
    return train_loader


def celebaDataLoader(image_size, batch_size=128, shuffle=True):
    dataset = datasets.ImageFolder(root='/data01/tf6/DATA',
                   transform=transforms.Compose([
                       transforms.Resize((image_size, image_size)),
                       transforms.CenterCrop(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       _noise_adder,
                   ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=True)
    return train_loader


def get_labels_and_class_counts(labels_list):
    '''
    Calculates the counts of all unique classes.
    '''
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts

class ImbalancedMNIST(Dataset):
    def __init__(self, imbal_class_prop, root, train, download, transform, seed=1):
        self.dataset = datasets.MNIST(
            root=root, train=train, download=download, transform=transform)
        self.train = train
        self.imbal_class_prop = imbal_class_prop
        self.nb_classes = len(imbal_class_prop)
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.idxs = self.resample()

    def get_labels_and_class_counts(self):
        return self.labels, self.imbal_class_counts

    def resample(self):
        '''
        Resample the indices to create an artificially imbalanced dataset.
        '''
        if self.train:
            targets, class_counts = get_labels_and_class_counts(
                self.dataset.train_labels)
        else:
            targets, class_counts = get_labels_and_class_counts(
                self.dataset.test_labels)
        # Get class indices for resampling
        class_indices = [np.where(targets == i)[0] for i in range(self.nb_classes)]
        # Reduce class count by proportion
        self.imbal_class_counts = [
            int(count * prop)
            for count, prop in zip(class_counts, self.imbal_class_prop)
        ]
        # Get class indices for reduced class count
        idxs = []
        for c in range(self.nb_classes):
            imbal_class_count = self.imbal_class_counts[c]
            idxs.append(class_indices[c][:imbal_class_count])
        idxs = np.hstack(idxs)
        random.shuffle(idxs)
        self.labels = targets[idxs]
        return idxs

    def __getitem__(self, index):
        img, target = self.dataset[self.idxs[index]]
        return img, target

    def __len__(self):
        return len(self.idxs)

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if (dataset_type is torchvision.datasets.MNIST ):
                return dataset.train_labels[idx].item()
            elif (dataset_type is torchvision.datasets.CIFAR10 ):
                return dataset.train_labels[idx]
            elif  dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


def getDataLoader(name, image_size, batch_size=64, shuffle=True, train=True, imbalancedataset=False, balancedbatch=False):
    if name == 'mnist':
        if imbalancedataset:
            return imbalancedmnist_DataLoader(image_size, batch_size, shuffle, train)
        else:
            return mnist_DataLoder(image_size, batch_size, shuffle, train, balancedbatch)
    elif name == 'cifar':
        return cifar_DataLoder(image_size, batch_size, shuffle, train, balancedbatch)
    elif name == 'stl':
        return stl_DataLoder(image_size, batch_size, shuffle)
    elif name == 'celeba':
        return celebaDataLoader(image_size, batch_size, shuffle)
    elif name == 'lsun':
        return lsunDataLoder(image_size, batch_size, shuffle)
    elif name == 'church_outdoor':
        return lsunDataLoder(image_size, batch_size, shuffle, category='church_outdoor_train')
    elif name == 'tower':
        return lsunDataLoder(image_size, batch_size, shuffle, category='tower_train')
