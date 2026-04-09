import torch
import torch.nn as nn
import torchaudio.transforms
from torchvision import models


def normalized_log_mel(
    mel_transform: torchaudio.transforms.MelSpectrogram,
    waveform_bt: torch.Tensor,
) -> torch.Tensor:
    """Same pipeline as training / server: Mel → log → per-utterance mean/std.

    waveform_bt: [B, T] mono samples at `mel_transform.sample_rate`.
    Returns tensor [B, 1, n_mels, time] for the ResNet backbone.
    """
    x = waveform_bt.unsqueeze(1)
    mels = mel_transform(x)
    log_mels = torch.log(mels + 1e-6)
    mean = log_mels.mean(dim=(1, 2, 3), keepdim=True)
    std = log_mels.std(dim=(1, 2, 3), keepdim=True)
    return (log_mels - mean) / (std + 1e-6)


class ResNetKWS(nn.Module):
    """ResNet-18 на лог-Mel; в чекпоинте обычно только `backbone.*` (как в lab3_resnet)."""

    def __init__(
        self,
        num_classes: int,
        *,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] сырые сэмплы 16 kHz, T == sample_rate (1 с)
        return self.backbone(normalized_log_mel(self.mel, x))
