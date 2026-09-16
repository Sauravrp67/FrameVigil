"""Small image classifier for deciding whether an empty scene needs detection.

Layer names match the included checkpoint. No external model code is loaded.
See licenses/ALBIREO-MIT.txt for the checkpoint and method provenance.
"""
import cv2
import numpy as np
import torch
from torch import nn
from ..tracking.parameters import ERD_INPUT_W, ERD_INPUT_H
from ..project import ROOT

class ScenePresenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        channels = 3
        for width in (8, 16, 16, 32, 32):
            for _ in range(2):
                blocks.extend((nn.Conv2d(channels, width, 3, padding=1),
                               nn.BatchNorm2d(width), nn.ReLU(inplace=True)))
                channels = width
            blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        dense = []
        incoming = 32 * 7 * 7
        for width in (64, 64):
            dense.extend((nn.Linear(incoming, width), nn.ReLU(inplace=True), nn.Dropout(0.5)))
            incoming = width
        dense.append(nn.Linear(incoming, 2))
        self.classifier = nn.Sequential(*dense)

    def forward(self, images):
        pooled = self.avgpool(self.features(images))
        return self.classifier(pooled.view(pooled.shape[0], -1))


def load_scene_classifier(device='cpu'):
    checkpoint = ROOT / 'models/erd_bdd100k.pt'
    network = ScenePresenceNet()
    network.load_state_dict(torch.load(str(checkpoint), map_location=device, weights_only=True))
    return network.to(device).eval()


def classify_scene(network, image, device='cpu'):
    resized = cv2.resize(image, (ERD_INPUT_W, ERD_INPUT_H))
    channels = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    tensor = torch.from_numpy(channels.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        category = network(tensor).argmax(dim=1)
    return int(category.item())
