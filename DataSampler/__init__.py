from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .dog_dataset import DogDataset
from .cifar_dataset import CIFAR10

def get_trainval_datasets(tag, resize,path):
    if tag == 'aircraft':
        return AircraftDataset(path,phase='train', resize=resize), AircraftDataset(path,phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(path,phase='train', resize=resize), BirdDataset(path, phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(path,phase='train', resize=resize), CarDataset(path, phase='val', resize=resize)
    elif tag == 'dog':
        return DogDataset(phase='train', resize=resize), DogDataset(phase='val', resize=resize)
    elif tag=='cifar10':
        return CIFAR10(root=path, train=True,download=True,resize=resize),CIFAR10(root=path, train=False,download=True,resize=resize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))