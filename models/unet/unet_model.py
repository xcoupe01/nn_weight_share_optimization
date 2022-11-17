import torch
from torch import nn

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down1 = down_unet_block(in_channels, 32, 7, 3)
        self.down2 = down_unet_block(32, 64, 3, 1)
        self.down3 = down_unet_block(64, 128, 3, 1)
        
        self.up3 = up_unet_block(128, 64, 3, 1)
        self.up2 = up_unet_block(2*64, 32, 3, 1)
        self.up1 = up_unet_block(2*32, out_channels, 3, 1)
    
    def __call__(self, x):

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        up3 = self.up3(down3)
        up2 = self.up2(torch.cat([up3, down2], 1))
        up1 = self.up1(torch.cat([up2, down1], 1))
        
        return up1

    def train_model(self, epochs, dataset, optimizer, loss_fn, acc_fn, device = 'cpu'):

        self.to(torch.device(device))

        train_loss, valid_loss = [], []

        best_acc = 0.0

        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}")

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.train(True)
                    dataloader = dataset.train_dl
                else:
                    self.train(False)
                    dataloader = dataset.valid_dl

                running_loss = 0.0
                running_acc = 0.0

                step = 0

                for x, y in dataloader:
                    x = x.to(torch.device(device))
                    y = y.to(torch.device(device))

                    step += 1

                    if phase == 'train':
                        
                        optimizer.zero_grad()
                        outputs = self(x)

                        loss = loss_fn(outputs, y)

                        loss.backward()
                        optimizer.step()

                    else:
                        with torch.no_grad():
                            outputs = self(x)
                            loss = loss_fn(outputs, y)

                acc = acc_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss*dataloader.batch_size

                if step % 100 == 0:
                    print('lol')

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print(epoch_loss, epoch_acc)

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

        return

def down_unet_block(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

def up_unet_block(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    )

            



