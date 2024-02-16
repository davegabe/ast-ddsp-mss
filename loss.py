import torch
import auraloss
import numpy as np
from config import *

class StemsMultiResolutionSTFTLoss(torch.nn.Module):
    """
    Compute the loss between the generated stems and the target stems.
    """
    def __init__(self):
        super().__init__()
        self.overlap = 0.75
        self.frame_length = [4096, 2048, 1024, 512, 256, 128, 64] # win_length
        self.frame_step = [int(fl * (1 - self.overlap)) for fl in self.frame_length] # hop_length or hop_size
        self.fft_lengths = [2 ** int(np.ceil(np.log2(fl))) for fl in self.frame_length] # n_fft or fft_size
        self.loss = auraloss.freq.MultiResolutionSTFTLoss(
            win_lengths=self.frame_length,
            hop_sizes=self.frame_step,
            fft_sizes=self.fft_lengths,
            # scale="mel",
            # n_bins=229,
            sample_rate=SR,
            perceptual_weighting=True
        )

    def forward(self, generated_stems: torch.Tensor, target_stems: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between the generated stems and the target stems.

        Args:
            generated_stems (torch.Tensor): The generated stems. [B, C, T]
            target_stems (torch.Tensor): The target stems. [B, C, T]

        Returns:
            torch.Tensor: The loss.
        """
        tot_loss = 0.0
        for i in range(generated_stems.shape[1]):
            gen_stem = generated_stems[:, i, :] # [B, T]
            tar_stem = target_stems[:, i, :] # [B, T]
            tot_loss += self.loss(gen_stem[:, None, :], tar_stem[:, None, :])
        return tot_loss