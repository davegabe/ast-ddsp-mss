from torch import nn
import torch
import torch.nn.functional as F
from src.models import ASTModel
from torchaudio.transforms import MelSpectrogram
import numpy as np
from ddsp.core import scale_function, remove_above_nyquist, upsample, frequencies_softmax
from ddsp.core import oscillator_bank, amp_to_impulse_response, fft_convolve, rnnSandwiched
from ddsp.model import Reverb
from config import *

class AST_DDSP(nn.Module):
    def __init__(self, harmonics_depth=64, n_amplitudes=100, n_bands=65, sr=SR, sample_length=LENGTH):
        super(AST_DDSP, self).__init__()
        # Parameters
        self.register_buffer("sr", torch.tensor(sr))
        self.register_buffer("sample_length", torch.tensor(sample_length))

        # For each time step, we need multiple harmonics, amps of those harmonics and noise
        self.depth = harmonics_depth
        self.n_harmonics = n_amplitudes * harmonics_depth
        self.n_amplitudes = n_amplitudes
        self.n_bands = n_bands

        # Compute t_dim minimum in order to have sixteenth notes
        bpm = 120 # average bpm
        secs_per_bar = 60 / bpm * 4 # seconds per bar
        n_bars = SECONDS / secs_per_bar # number of bars in the sample
        n_sixteenth = n_bars * 4 * 16 # 16th notes in the sample
        self.t_dim = int(n_sixteenth) # samples needed to have 16th notes
        print("t_dim: ", self.t_dim)

        # Mel spectrogram
        self.overlap = 0.75
        self.frame_length = 2048 # win_length
        self.frame_step = int(self.frame_length * (1 - self.overlap)) # hop_length or hop_size
        self.fft_lengths = 2 ** int(np.ceil(np.log2(self.frame_length))) # n_fft or fft_size
        self.mel = MelSpectrogram(
            sample_rate=sr,
            win_length=self.frame_length,
            hop_length=self.frame_step,
            n_fft=self.fft_lengths,
            normalized=True,
            f_min=20.0, # Inaudible frequencies are removed
            f_max=sr/2, # Nyquist frequency
        ).to("cuda")

        # Splits
        self.splits: dict[str, list[tuple[str, tuple]]] = {
            "harmonic": [
                ("frequencies", (self.t_dim, self.n_harmonics)),
                ("amplitudes", (self.t_dim, self.n_amplitudes)),
                ("total_amplitude", (1,)),
            ],
            "noise": [
                ("noise_magnitudes", (1, self.n_bands))
            ]
        }
        self.h_dim = sum([np.prod(shape)
                         for _, shape in self.splits["harmonic"]])
        self.n_dim = sum([np.prod(shape) for _, shape in self.splits["noise"]])

        # Sample length
        sample_mel = self.mel(torch.randn(1, sample_length).to("cuda"))
        self.f_dim = sample_mel.shape[-1]
        self.t_dim = sample_mel.shape[-2]

        # Sinusoidal encoder
        self.sinusoidal_encoder = ASTModel(
            label_dim=self.h_dim + self.n_dim,
            model_size=MODEL_SIZE,
            input_fdim=self.f_dim,
            input_tdim=self.t_dim,
            audioset_pretrain=True if MODEL_SIZE == "base384" else False
        )

        # # Reverb
        # self.reverb = Reverb(
        #     length=int(sr * 2),
        #     sampling_rate=sr,
        # )

        # # Audio normalize
        # self.normalize = nn.BatchNorm1d(1)

        # # Harmonic encoder
        # self.harmonic_encoder = rnnSandwiched()

    def tensor_to_dict(self, tensor: torch.Tensor) -> tuple[dict, dict]:
        """
        Convert tensor to dictionary of tensors.

        Args:
            tensor (torch.Tensor): The tensor to convert. [B, D]

        Returns:
            dict: The dictionary of tensors.
        """
        batch_size = tensor.shape[0]
        harmonic: dict[str, torch.Tensor] = {}
        noise: dict[str, torch.Tensor] = {}
        start = 0
        # Create dict for harmonic
        for key, shape in self.splits["harmonic"]:
            dim = np.prod(shape)
            harmonic[key] = tensor[:, start: start + dim]
            harmonic[key] = harmonic[key].view(batch_size, *shape)
            start += dim
        # Create dict for noise
        for key, shape in self.splits["noise"]:
            dim = np.prod(shape)
            noise[key] = tensor[:, start: start + dim]
            noise[key] = noise[key].view(batch_size, *shape)
            start += dim
        return harmonic, noise

    def audio_to_sin(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert audio to sinusoidal representation.

        Args:
            audio (torch.Tensor): The audio waveform. [B, T]

        Returns:
            torch.Tensor: The sinusoidal frequencies. [B, T, H]
            torch.Tensor: The sinusoidal amplitudes. [B, T, H]
            torch.Tensor: The noise magnitudes. [B, T, N]
            torch.Tensor: The synthesized sinusoidal audio. [B, T, 1]      
        """
        # Compute mel spectrogram
        mixture = self.mel(audio)  # [B, n_mels, T]
        mixture = mixture.transpose(1, 2)  # [B, T, n_mels]

        # Generate latent representation
        latent = self.sinusoidal_encoder(mixture)  # [B, D]

        # Output is in half type, convert to float
        latent = latent.float()

        # Split latent representation into harmonic and noise components
        harmonic_dict, noise_dict = self.tensor_to_dict(latent)

        ######################
        # * Sinusoidal part *#
        ######################
        sin_freqs = frequencies_softmax(harmonic_dict["frequencies"], depth=self.depth, hz_max=SR//2) # [B, self.t_dim, self.n_amplitudes]
        sin_amps = scale_function(harmonic_dict["amplitudes"]) # [B, self.t_dim, self.n_amplitudes]
        total_amplitude = F.sigmoid(harmonic_dict["total_amplitude"]) # [B, 1]
        total_amplitude = total_amplitude.unsqueeze(-1) # [B, 1, 1]


        # Upsample to desired length
        sin_freqs = upsample(sin_freqs, self.sample_length) # [B, T, self.n_amplitudes]
        sin_amps = upsample(sin_amps, self.sample_length, method="window") # [B, T, self.n_amplitudes]

        # Remove harmonics above nyquist frequency
        sin_amps = remove_above_nyquist(sin_amps, sin_freqs, self.sr) # [B, T, self.n_amplitudes]
        # Normalize amplitudes
        sin_amps /= sin_amps.sum(-1, keepdim=True)
        sin_amps *= total_amplitude # [B, T, self.n_amplitudes]
        
        # Synthesize sinusoidal part
        sin_audio = oscillator_bank(sin_freqs, sin_amps, self.sr)  # [B, T, 1]

        #################
        # * Noise part *#
        #################
        noise_magnitudes = scale_function(noise_dict["noise_magnitudes"])  # [B, 1, self.n_bands]

        # Convert to impulse response
        impulse = amp_to_impulse_response(noise_magnitudes, self.sample_length)
        sin_noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.sample_length,
        ).to(impulse) * 2 - 1

        # Convolve noise with impulse response
        sin_noise = fft_convolve(sin_noise, impulse).contiguous()
        sin_noise = sin_noise.reshape(sin_noise.shape[0], -1, 1) # [B, T, 1]

        # Sum signals
        sin_audio = sin_audio + sin_noise  # [B, T, 1]

        # Reverb part
        sin_audio = sin_audio.transpose(1, 2)  # [B, 1, T]

        return sin_freqs, sin_amps, noise_magnitudes, sin_audio

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            mixture (torch.Tensor): The mixture waveform. [B, T]

        Returns:
            torch.Tensor: The extracted stem waveform. [B, T]
        """
        # Audio to sinusoidal representation
        out_sin = self.audio_to_sin(mixture)
        sin_freqs, sin_amps, noise_magnitudes, sin_audio = out_sin

        return sin_audio


if __name__ == "__main__":
    model = AST_DDSP().to("cuda")
    x = torch.randn(BATCH_SIZE, LENGTH).to("cuda")
    y = model(x).to("cuda")
    print(torch.mean(y), torch.std(y), torch.max(y), torch.min(y), y.shape)
