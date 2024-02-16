import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import math


def _add_depth_axis(freqs: torch.Tensor, depth: int) -> torch.Tensor:
    """
    Turns [batch, time, sinusoids*depth] to [batch, time, sinusoids, depth]

    Args:
        freqs (torch.Tensor): The frequencies.
        depth (int): The depth.

    Returns:
        torch.Tensor: The frequencies with a depth axis.
    """
    freqs = freqs[..., None]  # [batch, time, sinusoids*depth, 1]
    # Unpack sinusoids dimension.
    n_batch, n_time, n_combined, _ = freqs.shape
    n_sinusoids = int(n_combined) // depth
    return freqs.reshape(n_batch, n_time, n_sinusoids, depth)


def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Safe log function to avoid log(0).

    Args:
        x (torch.Tensor): The input tensor.
        eps (float): The epsilon value to add to the input tensor.

    Returns:
        torch.Tensor: The log of the input tensor.
    """
    return torch.log(x + eps)


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps=1e-7) -> torch.Tensor:
    """
    Avoid dividing by zero by adding a small epsilon.

    Args:
        numerator (torch.Tensor): The numerator.
        denominator (torch.Tensor): The denominator.
        eps (float): The epsilon value to add to the denominator.

    Returns:
        torch.Tensor: The numerator divided by the denominator.
    """
    safe_denominator = torch.where(denominator > eps, denominator, eps)
    return numerator / safe_denominator


def logb(x: torch.Tensor, base: float = 2.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Logarithm with base as an argument.

    Args:
        x (torch.Tensor): The input tensor.
        base (float): The base of the logarithm.
        eps (float): The epsilon value to add to the input tensor.

    Returns:
        torch.Tensor: The logarithm of the input tensor.
    """
    base = torch.as_tensor(base)
    return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)


def overlap_and_add(signal: torch.Tensor, frame_step: int) -> torch.Tensor:
    """
    Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] `Tensor`. All dimensions may be
        unknown, and rank must be at least 2.
        frame_step: An integer or scalar `Tensor` denoting overlap offsets. Must be
        less than or equal to `frame_length`.
        name: An optional name for the operation.

    Returns:
        A `Tensor` with shape `[..., output_size]` containing the overlap-added
        frames of `signal`'s inner-most two dimensions.

    Raises:
        ValueError: If `signal`'s rank is less than 2, or `frame_step` is not a
        scalar integer.
    """
    with torch.autograd.profiler.record_function("overlap_and_add"):
        signal_shape = signal.shape

        # All dimensions that are not part of the overlap-and-add. Can be empty for
        # rank 2 inputs.
        outer_dimensions = signal_shape[:-2]
        outer_rank = len(outer_dimensions)

        frame_length = signal_shape[-1]
        frames = signal_shape[-2]

        # Compute output length.
        output_length = frame_length + frame_step * (frames - 1)

        # If frame_length is equal to frame_step, there's no overlap so just
        # reshape the tensor.
        if frame_step == signal.shape[-1]:
            output_shape = (outer_dimensions, output_length)
            return torch.reshape(signal, output_shape)

        # The following code is documented using this example:
        #
        # frame_step = 2
        # signal.shape = (3, 5)
        # a b c d e
        # f g h i j
        # k l m n o

        # Compute the number of segments, per frame.
        segments = -(-frame_length // frame_step)  # Divide and round up.

        # Pad the frame_length dimension to a multiple of the frame step.
        # Pad the frames dimension by `segments` so that signal.shape = (6, 6)
        # a b c d e 0
        # f g h i j 0
        # k l m n o 0
        # 0 0 0 0 0 0
        # 0 0 0 0 0 0
        # 0 0 0 0 0 0
        paddings = (0, segments * frame_step - frame_length, 0, segments)
        signal = F.pad(signal, paddings)

        # Reshape so that signal.shape = (3, 6, 2)
        # ab cd e0
        # fg hi j0
        # kl mn o0
        # 00 00 00
        # 00 00 00
        # 00 00 00
        shape = outer_dimensions + (frames + segments, segments, frame_step)
        signal = signal.reshape(shape)

        # Transpose dimensions so that signal.shape = (3, 6, 2)
        # ab fg kl 00 00 00
        # cd hi mn 00 00 00
        # e0 j0 o0 00 00 00
        perm = [i for i in range(outer_rank)] + \
            [outer_rank + 1, outer_rank, outer_rank + 2]
        signal = signal.permute(perm)

        # Reshape so that signal.shape = (18, 2)
        # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0 00 00 00
        shape = outer_dimensions + ((frames + segments) * segments, frame_step)
        signal = signal.reshape(shape)

        # Truncate so that signal.shape = (15, 2)
        # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0
        signal = signal[..., :(frames + segments - 1) * segments, :]

        # Reshape so that signal.shape = (3, 5, 2)
        # ab fg kl 00 00
        # 00 cd hi mn 00
        # 00 00 e0 j0 o0
        shape = outer_dimensions + \
            (segments, (frames + segments - 1), frame_step)
        signal = signal.reshape(shape)

        # Now, reduce over the columns, to achieve the desired sum.
        signal = torch.sum(signal, -3)

        # Flatten the array.
        shape = outer_dimensions + ((frames + segments - 1) * frame_step,)
        signal = signal.reshape(shape)

        # Truncate to final length.
        signal = signal[..., :output_length]

        return signal


def upsample_with_windows(inputs: torch.Tensor, n_samples: int, add_endpoint: bool = True) -> torch.Tensor:
    """
    Upsample a series of frames using overlapping Hann windows.

    Args:
        inputs (torch.Tensor): The input frames. [B, T, D]
        n_samples (int): The number of samples to upsample to.
        add_endpoint (bool): Whether to add an endpoint to the input frames.

    Returns:
        torch.Tensor: The upsampled frames. [B, n_samples, D]
    """
    if inputs.dim() != 3:
        raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
                         'not {}.'.format(inputs.shape))

    if add_endpoint:
        # Append the last timestep to mimic behavior of tf.image.resize
        inputs = torch.cat([inputs, inputs[:, -1:, :]], dim=1)

    n_frames = inputs.shape[1]
    n_intervals = (n_frames - 1)

    if n_frames >= n_samples:
        raise ValueError('Upsample with windows cannot be used for downsampling '
                         'More input frames ({}) than output timesteps ({})'.format(
                             n_frames, n_samples))

    if n_samples % n_intervals != 0.0:
        minus_one = '' if add_endpoint else ' - 1'
        raise ValueError(
            'For upsampling, the target number of timesteps must be divisible '
            'by the number of input frames{}. (timesteps:{}, frames:{}, '
            'add_endpoint={}).'.format(minus_one, n_samples, n_frames,
                                       add_endpoint))

    # Constant overlap-add, half overlapping windows.
    hop_size = n_samples // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length).to(
        inputs.device)  # [window_length]

    # Transpose for overlap_and_add
    x = inputs.permute(0, 2, 1)  # [batch_size, n_channels, n_frames]

    # Broadcast multiply
    # Add dimension for windows [batch_size, n_channels, n_frames, window]
    x = x.unsqueeze(3)
    window = window.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_windowed = x * window
    x = overlap_and_add(x_windowed, hop_size)

    # Transpose back.
    x = x.permute(0, 2, 1)  # [batch_size, n_samples, n_channels]

    # Trim the rise and fall of the first and last window
    return x[:, hop_size:-hop_size, :]


def upsample(signal: torch.Tensor, n_samples: int, method: str = "linear", add_endpoint: bool = True) -> torch.Tensor:
    """
    Upsample a signal to be n_samples long.

    Args:
        signal (torch.Tensor): The signal to upsample. [B, T, D]
        n_samples (int): The length of the output signal.
    """
    n = signal.shape[-2]
    if n == n_samples:
        return signal
    else:
        # Interpolate to [B, n_samples, D]
        if method == "window":
            signal = upsample_with_windows(signal, n_samples, add_endpoint)
        else:
            # Swap time and channel dimensions in order to interpolate time
            signal = signal.transpose(1, 2)
            signal = F.interpolate(signal, size=n_samples, mode=method)
            signal = signal.transpose(1, 2)
        return signal


def remove_above_nyquist(amplitudes: torch.Tensor, pitch: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """
    Remove harmonics above the nyquist frequency.

    Args:
        amplitudes (torch.Tensor): The amplitudes of the harmonics. [B, T, H]
        pitch (torch.Tensor): The pitch of the signal. [B, T, H]
        sampling_rate (int): The sampling rate of the signal.

    Returns:
        torch.Tensor: The amplitudes with harmonics above the nyquist frequency removed.
    """
    nyquist = sampling_rate / 2
    return torch.where(pitch > nyquist, torch.zeros_like(amplitudes), amplitudes)


def hz_to_midi(frequencies: float) -> torch.Tensor:
    """
    Convert frequencies to MIDI notes.

    Args:
        frequencies (torch.Tensor): The frequencies to convert. [B, T]

    Returns:
        torch.Tensor: The MIDI notes. [B, T]
    """
    frequencies = torch.as_tensor(frequencies)
    A4 = torch.as_tensor(440.0)
    notes = 12.0 * (logb(frequencies, 2.0) - logb(A4, 2.0)) + 69.0
    # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
    notes = torch.where(frequencies <= 0.0, torch.zeros_like(notes), notes)
    return notes


def midi_to_hz(notes: torch.Tensor, midi_zero_silence: bool = False) -> torch.Tensor:
    """
    Convert MIDI notes to frequencies.

    Args:
        notes: Tensor containing encoded pitch in MIDI scale.
        midi_zero_silence: Whether to output 0 hz for midi 0, which would be
            convenient when midi 0 represents silence. By defualt (False), midi 0.0
            corresponds to 8.18 Hz.

    Returns:
        hz: Frequency of MIDI in hz, same shape as input.
    """
    notes = notes.float()
    hz = 440.0 * (2.0 ** ((notes - 69.0) / 12.0))
    # Map MIDI 0 as 0 hz when MIDI 0 is silence.
    if midi_zero_silence:
        hz = torch.where(notes == 0.0, torch.zeros_like(hz), hz)
    return hz


def unit_to_midi(unit: torch.Tensor, midi_min: float = 20.0, midi_max: float = 90.0, clip: bool = False) -> torch.Tensor:
    """
    Map the unit interval [0, 1] to MIDI notes.

    Args:
        unit (torch.Tensor): The unit interval.
        midi_min (float): The minimum MIDI note.
        midi_max (float): The maximum MIDI note.
        clip (bool): Whether to clip the output to the MIDI range.

    Returns:
        torch.Tensor: The MIDI notes.    
    """
    unit = torch.clamp(unit, 0.0, 1.0) if clip else unit
    return midi_min + (midi_max - midi_min) * unit


def unit_to_hz(unit: torch.Tensor, hz_min: float, hz_max: float, clip: bool = False) -> torch.Tensor:
    """
    Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically.

    Args:
        unit (torch.Tensor): The unit interval.
        hz_min (float): The minimum frequency.
        hz_max (float): The maximum frequency.
        clip (bool): Whether to clip the output to the frequency range.

    Returns:
        torch.Tensor: The frequencies.
    """
    midi = unit_to_midi(
        unit, midi_min=hz_to_midi(hz_min),
        midi_max=hz_to_midi(hz_max), clip=clip
    )
    return midi_to_hz(midi)


def frequencies_softmax(freqs: torch.Tensor, depth: int = 64, hz_min: float = 20.0, hz_max: float = 8000.0) -> torch.Tensor:
    """
    Softmax to logarithmically scale network outputs to frequencies.

    Args:
        freqs: Neural network outputs [batch, time, n_sinusoids]
        depth: Number of frequency bins per octave
        hz_min: Lowest frequency to consider
        hz_max: Highest frequency to consider

    Returns:
        A tensor of frequencies in hertz [batch, time, n_sinusoids]
    """
    if len(freqs.shape) == 3:
        # Add depth axis: [B, T, N*D] -> [B, T, N, D]
        freqs = _add_depth_axis(freqs, depth)
    else:
        depth = int(freqs.shape[-1])

    # Probs [B, T, N, D]
    f_probs = torch.softmax(freqs, dim=-1)

    # [1, 1, 1, D]
    unit_bins = torch.linspace(0.0, 1.0, depth, dtype=freqs.dtype, device=freqs.device)
    unit_bins = unit_bins[None, None, None, :]

    # [B, T, N]
    f_unit = torch.sum(unit_bins * f_probs, axis=-1)
    return unit_to_hz(f_unit, hz_min=hz_min, hz_max=hz_max)


def scale_function(x: torch.Tensor) -> torch.Tensor:
    """
    Scale function for the amplitude.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The scaled tensor.
    """
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def oscillator_bank(frequencies: torch.Tensor, amplitudes: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """
    Synthesize a harmonic signal from pitch and amplitudes.

    Args:
        frequencies (torch.Tensor): The frequencies of the signal. [B, T, H]
        amplitudes (torch.Tensor): The amplitudes of the harmonics. [B, T, H]
        sampling_rate (int): The sampling rate of the signal.
    """
    # Angular frequency, Hz -> radians per sample
    omega = 2 * math.pi * frequencies / sampling_rate
    # Phase accumulation
    phase = torch.cumsum(omega, dim=1)
    # Oscillator bank
    signal = torch.sin(phase) * amplitudes  # [B, T, H]
    # Sum over harmonics
    signal = signal.sum(-1, keepdim=True)  # [B, T, 1]
    return signal


def amp_to_impulse_response(amp: torch.Tensor, target_size: torch.Tensor) -> torch.Tensor:
    """
    Convert amplitude to an impulse response.

    Args:
        amp (torch.Tensor): The amplitude of the impulse response. [B, T]
        target_size (torch.Tensor): The target size of the impulse response.

    Returns:
        torch.Tensor: The impulse response. [B, T]
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve two signals using FFT.

    Args:
        signal (torch.Tensor): The first signal. [B, T, 1]
        kernel (torch.Tensor): The second signal. [B, T, 1]

    Returns:
        torch.Tensor: The convolved signal. [B, T, 1]
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def mlp(in_size: int, hidden_size: int, n_layers: int) -> nn.Sequential:
    """
    Create an MLP with LeakyReLU activations and LayerNorm.

    Args:
        in_size (int): The input size.
        hidden_size (int): The hidden size.
        n_layers (int): The number of layers.

    Returns:
        nn.Sequential: The MLP.
    """
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input: int, hidden_size: int) -> nn.GRU:
    """
    Create a GRU.

    Args:
        n_input (int): The input size.
        hidden_size (int): The hidden size.

    Returns:
        nn.GRU: The GRU.
    """
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def rnnSandwiched(fc_stack_ch: int = 256, fc_stack_layers: int = 2, rnn_ch: int = 512) -> nn.Sequential:
    """
    RNN Sandwiched by two FC Stacks.

    Args:
        fc_stack_ch (int): The number of channels in the FC stack.
        fc_stack_layers (int): The number of layers in the FC stack.
        rnn_ch (int): The number of channels in the RNN.

    Returns:
        nn.Sequential: The RNN sandwiched by two FC stacks.
    """
    layers = [
        mlp(fc_stack_ch, fc_stack_ch, fc_stack_layers),
        gru(rnn_ch, rnn_ch),
        mlp(fc_stack_ch, fc_stack_ch, fc_stack_layers),
    ]
    return nn.Sequential(*layers)
