"""ResNet-32 cho ảnh CIFAR 32x32, tự cài từ kiến trúc ResNet gốc.

Nguồn: He et al., CVPR 2016, mục 4.2: 6n+2 lớp, n=5, kênh 16/32/64,
shortcut option A không có tham số. Đây là backbone chuẩn, không phải
implementation ACIL của tác giả. Biến thể shortcut cần đối chiếu trước
khi tuyên bố khớp hoàn toàn môi trường benchmark ACIL.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Hai conv 3x3; shortcut identity hoặc giảm mẫu và đệm kênh bằng 0."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.channel_padding = (out_channels - in_channels) // 2

    def forward(self, inputs):
        """Nhận tensor NCHW và trả đặc trưng sau cộng residual rồi ReLU."""
        identity = inputs
        if self.stride != 1:
            identity = identity[:, :, ::self.stride, ::self.stride]
            identity = F.pad(identity, (0, 0, 0, 0, self.channel_padding, self.channel_padding))
        residual = F.relu(self.bn1(self.conv1(inputs)))
        residual = self.bn2(self.conv2(residual))
        return F.relu(residual + identity)


class ResNet32(nn.Module):
    """Backbone CIFAR gồm 31 conv và một FC; forward_features trả 64 chiều."""

    def __init__(self, n_classes=50):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._stage(16, 16, 1)
        self.stage2 = self._stage(16, 32, 2)
        self.stage3 = self._stage(32, 64, 2)
        self.classifier = nn.Linear(64, n_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _stage(in_channels, out_channels, stride):
        """Tạo một stage gồm 5 residual block; chỉ block đầu có thể giảm mẫu."""
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            *[ResidualBlock(out_channels, out_channels) for _ in range(4)],
        )

    def forward_features(self, inputs):
        """Nhận ảnh (N,3,32,32), trả đặc trưng (N,64) sau global average pooling."""
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.stage3(self.stage2(self.stage1(outputs)))
        return F.adaptive_avg_pool2d(outputs, 1).flatten(1)

    def forward(self, inputs):
        """Trả logits (N,n_classes); CrossEntropyLoss tự xử lý log-softmax."""
        return self.classifier(self.forward_features(inputs))


def freeze_backbone(model):
    """Cố định mọi tham số và chuyển eval, bao gồm running statistics BatchNorm."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
