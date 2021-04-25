from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .dog_dataset import DogDataset


def get_trainval_datasets(tag, resize,path):
    if tag == 'aircraft':
        return AircraftDataset(path,phase='train', resize=resize), AircraftDataset(path,phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(path,phase='train', resize=resize), BirdDataset(path, phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(path,phase='train', resize=resize), CarDataset(path, phase='val', resize=resize)
    elif tag == 'dog':
        return DogDataset(phase='train', resize=resize), DogDataset(phase='val', resize=resize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))