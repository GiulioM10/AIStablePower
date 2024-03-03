
import torch
import torch.nn as nn
from AIStablePower.models.blocktype import BlockType
from AIStablePower.models.blocks import build_block

class StableCNN(nn.Module):
    def __init__(self, blockType: BlockType, stages: int, blocks_per_stage: int, *args, **kwargs) -> None:
        super(StableCNN, self).__init__(*args, **kwargs)
        
        self.blocks = nn.ModuleList([])
        self.stages = stages
        self.blocks_per_stage = blocks_per_stage
        self.blockType = blockType
        
        stem_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(5, 1), stride=(5, 1), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=(5, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.blocks.append(stem_conv)
        
        inchannels = 32
        kernel_size = 7
        for s in range(self.stages):
            inchannels = inchannels*2
            kernel_size -= 2
            for _ in range(self.blocks_per_stage - 1):
                self.blocks.append(build_block(self.blockType, inchannels, inchannels, kernel_size, False))
            self.blocks.append(build_block(self.blockType, inchannels, 2*inchannels * (s != self.stages - 1) + inchannels * (s == self.stages - 1), kernel_size, (s != self.stages - 1)))
            
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannels, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.squeeze(x)