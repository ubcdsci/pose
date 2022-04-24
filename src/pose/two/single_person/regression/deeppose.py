"""
@article{
    1312.4659,
    Author = {Alexander Toshev and Christian Szegedy},
    Title = {DeepPose: Human Pose Estimation via Deep Neural Networks},
    Year = {2013},
    Eprint = {arXiv:1312.4659},
    Doi = {10.1109/CVPR.2014.214},
}

@inproceedings{
    inproceedings,
    author = {Krizhevsky, Alex and Sutskever, I and Hinton, G},
    year = {2012},
    month = {01},
    pages = {1097-1105},
    title = {Imagenet classification with deep convolutional neural networks}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from pose.data.mpii import load_mpii_data
from pose.two.single_person.dataset import SinglePose2DDataset
from pose.util import Human2D, BoundingBox
from pose.definitions import TORCH_DEVICE
from pose.two.estimator import Estimator2D


class DeepPoseJointRegressor(nn.Module):
    def __init__(self):
        self.isTrained = False
        """
        The input to the net is an image of 220 × 220 which via stride of 4 is fed into the network.

        Specifics of this implementation come from both:
        - DeepPose: Human Pose Estimation via Deep Neural Networks
        - Imagenet classification with deep convolutional neural networks

        The CNN architecture contains 7 layers which can be listed as : C(55×55×96) — LRN — P — C(27×27×256) — LRN — P — C(13×13×384) — C(13×13×384) — C(13×13×256) — P — F(4096) — F(4096)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4), padding=(4, 4))
        self.lrn = nn.LocalResponseNorm(size=5, alpha=10 ** -4, beta=0.75, k=2)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding="same")
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding="same")
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding="same")
        self.fc1 = nn.Linear(9216, 4096)
        # Output vector is comprised of x and y for each join in the human, so we multiple the size by two
        self.fc2 = nn.Linear(4096, 2 * Human2D.NUM_JOINTS)

    def test_from_dataset(self, dataset, train_test_ratio, batch_size):
        criterion = nn.MSELoss().to(TORCH_DEVICE)
        running_loss = 0
        for i in tqdm(range(int(len(dataset.data) * train_test_ratio))[::batch_size][:-1]):
            batch = dataset.create_batch(i, i + batch_size, size=(220, 220))
            outputs = self(batch.img_tensor)
            loss = criterion(outputs, batch.humans_tensor)
            running_loss += loss.item()
        return running_loss / len(dataset.data)

    def forward(self, x):
        x = self.lrn(torch.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.lrn(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_from_dataset(
            self,
            train: SinglePose2DDataset,
            batch_size=15,
            num_epochs=1,
            print_stride=10,
            train_test_ratio=0.7,
            plot=True
    ):

        criterion = nn.MSELoss().to(TORCH_DEVICE)
        optimizer = optim.Adam(self.parameters(), lr=0.00005)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4)

        losses = []
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            running_loss = 0.0
            iters = 0
            # Iterate through data points while ensuring we don't access out of bounds
            for i in tqdm(range(int(len(train.data) * train_test_ratio))[::batch_size][:-1]):
                batch = train.create_batch(i, i + batch_size, size=(220, 220))
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(batch.img_tensor)
                loss = criterion(outputs, batch.humans_tensor)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                iters += 1
                if iters % print_stride == 0:
                    running_loss = running_loss * batch_size / print_stride
                    print('[%d] loss: %.5f' % (epoch + 1, running_loss))
                    losses.append(running_loss)
                    running_loss = 0
                self.isTrained = True
            # scheduler.step(batch_eval_loss)


class DeepPose(Estimator2D):
    def __init__(self, regressor: DeepPoseJointRegressor):
        self.regressor = regressor

    def _initial_regression(self):
        pass

    def _refine(self):
        pass

    def _find_human_bbox(self) -> BoundingBox:
        pass

    def estimate(self, image, visualize=False) -> Human2D:
        pass

    def estimate_single(self, image, visualize=False) -> Human2D:
        img_tensor = torch.tensor(np.expand_dims(np.rollaxis(image, 2, 0), 0)).to(TORCH_DEVICE)
        output = self.regressor(img_tensor)
        return Human2D.from_vector(output)


if __name__ == "__main__":
    regressor = DeepPoseJointRegressor().to(TORCH_DEVICE)
    train_dtst = load_mpii_data();
    train_dtst.create_batch()
    regressor.train_from_dataset(train_dtst)
    torch.save(regressor.state_dict(), "model.pth")
