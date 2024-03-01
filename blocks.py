# GM 05/17/23
import numpy as np
from AIStablePower.blocktype import BlockType
import torch
from torch.nn import Module
import torch.nn as nn

class Block(Module):
  def __init__(self,
               inchannels: int,
               outchannels: int,
               kernel_size: int,
               downsample: bool = False
               ) -> None:
    """This is the block interface. Each type of block inherits form this class

    Args:
        inchannels (int): Number of input channels
        outchannels (int): Number of output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether this is a downsampling block. Defaults to False.
    """
    super(Block, self).__init__()
    self.inchannels = inchannels
    self.outchannels = outchannels
    self.downsample = downsample
    self.kernel_size = kernel_size
    self.padding = self.kernel_size // 2

    if downsample:
      self.stride = 2
    else:
      self.stride = 1

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''
    The forward method to be overloaded by the inheritors
    '''
    pass


class ConvNext(Block):
  def __init__(self,
               inchannels: int,
               outchannels: int,
               kernel_size: int,
               downsample=False,
               expansion=4) -> None:
    """A state of the art architecture capable of matching the performance
    of trasformers on image classification tasks.

    Args:
        inchannels (int): Input channels
        outchannels (int): Output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether to do downsample. Defaults to False.
        expansion (int, optional): Wether to do downsample. Defaults to 4.
    """
    super().__init__(inchannels, outchannels, kernel_size, downsample)
    # Save expansion hyper-parameter
    self.expansion = expansion
    midchannels = self.outchannels * self.expansion

    # Adapt the tensor
    if self.inchannels == self.outchannels and not self.downsample:
      self.adapt = nn.Sequential(nn.Identity())
    else:
      self.adapt = nn.Sequential(
          nn.GroupNorm(num_groups=1, num_channels=self.inchannels),
          nn.Conv2d(
              self.inchannels, self.outchannels,
              kernel_size=3, stride=self.stride, padding=1
          )
      )

    # ConvNext block
    self.block = nn.Sequential(
        nn.Conv2d(
            self.outchannels, self.outchannels,
            kernel_size=self.kernel_size, stride=1, padding=self.padding,
            groups=self.outchannels
        ),
        nn.GroupNorm(num_groups=1, num_channels=self.outchannels),
        nn.Conv2d(self.outchannels, midchannels, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(midchannels, self.outchannels, kernel_size=1)
    )


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.adapt(x)
    F = self.block(x)
    out = x + F
    
    return out
  

class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction),
                nn.ReLU(),
                nn.Linear(channel//reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    

class MobileNetv3(Block):
  def __init__(self, inchannels:int,
               outchannels,
               kernel_size: int,
               downsample=False,
               expansion=4) -> None:
    """An Imporved version of the MobileNetv2 block. It uses different activation
    functions and a 'squeeze and excite' layer

    Args:
        inchannels (int): Input channels
        outchannels (int): Output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether to do downsample. Defaults to False.
        expansion (int, optional): Wether to do downsample. Defaults to 3.
    """
    super().__init__(inchannels, outchannels, kernel_size, downsample)

    self.expansion = expansion
    midchannels = self.outchannels * self.expansion

    # Adapt the tensor
    if self.inchannels == self.outchannels and not self.downsample:
      self.adapt = nn.Sequential(nn.Identity())
    else:
      self.adapt = nn.Sequential(
          nn.GroupNorm(num_groups=1, num_channels=self.inchannels),
          nn.Conv2d(
              self.inchannels, self.outchannels,
              kernel_size=3, stride=self.stride, padding=1
          )
      )

    self.block = nn.Sequential(
        nn.Conv2d(self.outchannels, midchannels, kernel_size=1, stride = 1),
        nn.BatchNorm2d(midchannels),
        h_swish(),
        nn.Conv2d(
            midchannels, midchannels,
            kernel_size=self.kernel_size, stride = 1, padding=self.padding,
            groups = midchannels
        ),
        nn.BatchNorm2d(midchannels),
        SELayer(midchannels),
        h_swish(),
        nn.Conv2d(midchannels, self.outchannels, kernel_size=1, stride=1),
        nn.BatchNorm2d(self.outchannels)
    )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.adapt(x)
    out = self.block(out)

    if self.inchannels == self.outchannels and not self.downsample:
      out = out + x

    return out
  
class BlockFactory():
  def __init__(self) -> None:
     self.compatible_blocks = BlockType()
  
  def get_block(self, block_type: BlockType, inchannels: int, outchannels: int, kernel_size: int, downsample: bool) -> Block:
    if block_type not in self.compatible_blocks:
      raise RuntimeError(f"BlockType {block_type} is not yet supported")
    return self._build_block(block_type, inchannels, outchannels, kernel_size, downsample)
  
  @staticmethod
  def _build_block(block_type: BlockType, inchannels: int, outchannels: int, kernel_size: int, downsample: bool) -> Block:
    if block_type == BlockType.ConvNext:
      return ConvNext(inchannels, outchannels, kernel_size, downsample)
    else:
      return MobileNetv3(inchannels, outchannels, kernel_size, downsample)