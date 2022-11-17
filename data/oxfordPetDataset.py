from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch


IMAGE_DIMENSIONS = (128, 128)   # sets the datasets image dimensions
MASK_TYPE = 3                   # sets the datasets mask preset (number of segmentation types - 2 or 3)

class OxfordPetDataset:
    """
    Oxford pets train and test datasets
    """
    def __init__(self, root, val_split = 0.1, batch_size=20):
        """
        Inits the train and test datasets with a corresponing data transformations
        """
        im_transform = transforms.Compose([
            transforms.Resize(IMAGE_DIMENSIONS),
            transforms.ToTensor()
        ])

        mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_DIMENSIONS),
            transforms.ToTensor(),
            transforms.Lambda(lambda y: preprocess_mask(y))
        ]) if MASK_TYPE == 2 else im_transform

        self.train = OxfordIIITPet(
            root, 
            download=True, 
            split='trainval', 
            target_types='segmentation', 
            transform=im_transform, 
            target_transform=mask_transform
        )

        self.test = OxfordIIITPet(
            root, 
            download=True, 
            split='test', 
            target_types='segmentation', 
            transform=im_transform, 
            target_transform=mask_transform
        )
        
        train_count = int(len(self.train) * (1 - val_split))
        val_count = len(self.train) - train_count

        train_ds, valid_ds = torch.utils.data.random_split(self.train, (train_count, val_count))
        self.train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)


def preprocess_mask(mask):
    """
    Preprocesses the segmentation masks. The preprocession removes the edge
    class and creates mask only with classes:
        1 - the animal to be segmented
        2 - surroundings

    Args:
        mask (PIL image): is the mask image

    Returns:
        PIL image: is the altered mask image
    """
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask

def display_image_grid(dataset, range, prediction=None):
    """
    Displays given images, their masks and given predictions.

    Args:
        dataset (Dataset): is the dataset to display
        range (Range): is a range object which establishes the images wanted to be displayed
        prediction (array of PIL images, optional): Array of predicted corresponding images. Defaults to None.
    """
    cols = 2 if prediction == None else 3
    rows = len(range)
    figure, ax = plt.subplots(nrows= rows, ncols= cols, figsize=(10, 24))

    transform = transforms.ToPILImage()

    for row, image in enumerate(range):

        ax[row, 0].imshow(transform(dataset[image][0]))
        ax[row, 1].imshow(transform(dataset[image][1]), interpolation='nearest')

        ax[row, 0].set_title('Image')
        ax[row, 1].set_title('Mask')

        ax[row, 0].set_axis_off()
        ax[row, 1].set_axis_off()

        if prediction != None:
            ax[row, 2].imshow(transform(prediction[row]))
            ax[row, 2].set_title('Prediction')
            ax[row, 2].set_axis_off()

    plt.show()


if __name__ == '__main__':
    dataset = OxfordPetDataset('.')
    display_image_grid(dataset.train, range(0,4))
