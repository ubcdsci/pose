"""
@article{
    1312.4659,
    Author = {Alexander Toshev and Christian Szegedy},
    Title = {DeepPose: Human Pose Estimation via Deep Neural Networks},
    Year = {2013},
    Eprint = {arXiv:1312.4659},
    Doi = {10.1109/CVPR.2014.214},
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd

from pose.util import Human2D, Point2D, BoundingBox
from pose.definitions import ROOT_PATH, TORCH_DEVICE
from pose.two.estimator import Estimator2D


class DeepPoseJointRegressor(nn.Module):
    def __init__(self):
        """
        The input to the net is an image of 220 × 220 which via stride of 4 is fed into the network.

        Specifics of this implementation come from both:
        - DeepPose: Human Pose Estimation via Deep Neural Networks
        - Imagenet classification with deep convolutional neural networks
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4))
        self.lrn = nn.LocalResponseNorm(size=5, alpha=10**-4, beta=0.75, k=2)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3))
        self.fc1 = nn.Linear(4096, 4096)
        # Output vector is comprised of x and y for each join in the human, so we multiple the size by two
        self.fc2 = nn.Linear(4096, 2 * Human2D.NUM_JOINTS)

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


    def train_from_images(self, train, validation, batch_size=256, num_epochs=2, plot=True):



        criterion = nn.MSELoss().to(TORCH_DEVICE)
        optimizer = optim.Adam(self.parameters(), lr=0.0005, weight_decay=0.004)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4)

        model_actions = []
        labels = []
        losses = []
        validation_acc = []
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            running_loss = 0.0
            # Iterate through data points while ensuring we don't access out of bounds
            for i in tqdm(range(len(transitions))[::batch_size][:-1]):
                batch_data = [[transitions[j].prev_state, transitions[j].action] for j in range(i, i + batch_size)]
                x, y = zip(*batch_data)
                x = torch.stack(x, dim=0).to(TORCH_DEVICE).float()
                y = torch.stack(y, dim=0).to(TORCH_DEVICE).float()
                y_class_idx = torch.max(y, 1)[1]
                labels.extend(y_class_idx.tolist())
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(x)
                model_actions.extend([x.argmax() for x in outputs.detach().cpu().numpy()])
                loss = criterion(outputs, y_class_idx)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            batch_eval_loss = running_loss * batch_size
            print('[%d] loss: %.5f' % (epoch + 1, batch_eval_loss))
            losses.append(batch_eval_loss)
            scheduler.step(batch_eval_loss)
            validation_acc.append(self.evaluate_on_transitions(validation_transitions))


class DeepPose(Estimator2D):
    def __init__(self):
        pass

    def _initial_regression(self):
        pass

    def _refine(self):
        pass

    def _normalize_joint(self, joint: Point2D, human_bbox: BoundingBox) -> Point2D:
        return Point2D(
            (joint.x - human_bbox.center.x) / human_bbox.w,
            (joint.y - human_bbox.center.y) / human_bbox.h,
        )

    def _denormalize_joint(self, joint: Point2D, human_bbox: BoundingBox) -> Point2D:
        return Point2D(
            joint.x * human_bbox.w + human_bbox.center.x,
            joint.y * human_bbox.h + human_bbox.center.y
        )

    def _find_human_bbox(self) -> BoundingBox:
        pass

    def estimate(self, image) -> Human2D:
        pass



