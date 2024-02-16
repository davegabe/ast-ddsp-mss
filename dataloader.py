import math
import os
import re
from typing import *

import av
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
import ast
from config import *


def get_duration_sec(file, cache=False):
    try:
        with open(file + ".dur", "r") as f:
            duration = float(f.readline().strip("\n"))
        return duration
    except:
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + ".dur", "w") as f:
                f.write(str(duration) + "\n")
        return duration


def load_audio(file, sr, offset, duration, resample=True, approx=False, time_base="samples", check_duration=True):
    resampler = None
    if time_base == "sec":
        offset = offset * sr
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    if not os.path.exists(file):
        return np.zeros((2, duration), dtype=np.float32), sr
    container = av.open(file, buffer_size=32768*32, timeout=10)
    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    if approx:

        if offset + duration > audio_duration * sr:
            # Move back one window. Cap at audio_duration
            offset = min(audio_duration * sr - duration, offset - duration)
    else:
        if check_duration:
            assert (
                offset + duration <= audio_duration * sr
            ), f"End {offset + duration} beyond duration {audio_duration*sr}"
    if resample:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
    else:
        assert sr == audio.sample_rate
    offset = int(
        offset / sr / float(audio.time_base)
    )  # int(offset / float(audio.time_base)) # Use units of time_base for seeking
    # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    duration = int(duration)
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    for frame in container.decode(audio=0):  # Only first audio stream
        if resample:
            frame.pts = None
            frame = resampler.resample(frame)
        # Convert to floats and not int16
        frame = frame[0].to_ndarray(format="fltp")
        read = frame.shape[-1]
        if total_read + read > duration:
            read = duration - total_read
        sig[:, total_read: total_read + read] = frame[:, :read]
        total_read += read
        if total_read == duration:
            break
    assert total_read <= duration, f"Expected {duration} frames, got {total_read}"
    return sig, sr


def _identity(x):
    return x


def is_silent(signal: torch.Tensor) -> bool:
    """
    Check if signal is silent.

    Args:
        signal (torch.Tensor): The signal to check.

    Returns:
        bool: True if signal is silent, False otherwise.
    """
    if torch.var(signal) < VARIANCE_THRESHOLD: # If has zero variance
        return True
    loudness = torch.sqrt(torch.mean(signal ** 2)) # RMS
    return loudness < SILENCE_THRESHOLD

def get_all_chunks(
    stems: list[str],
    duration: int,
    chunk_size: int
):
    # Load stems
    separated_track = []
    for stem in stems:
        data, sr = load_audio(stem, sr=SR, offset=0, duration=duration)
        data = torch.from_numpy(data)
        separated_track.append(data)
    separated_track = torch.cat(separated_track)

    # Get chunks with overlap
    overlap = chunk_size * (1 - OVERLAP)
    _, num_samples = separated_track.shape
    num_chunks = num_samples // overlap + \
        int(num_samples % overlap != 0)

    # Get available chunks
    available_chunks = []
    for i in range(int(num_chunks)):
        chunk = separated_track[:, i * overlap: i * overlap + chunk_size]
        _, chunk_samples = chunk.shape
        
        # Remove if it contains less than the minimum chunk size
        if chunk_samples < chunk_size:
            continue

        available_chunks.append(i)
    return available_chunks


def get_nonsilent_and_multi_instr_chunks(
    stems: list[str],
    duration: int,
    chunk_size: int
):
    # Load stems
    separated_track = []
    for stem in stems:
        data, sr = load_audio(stem, sr=SR, offset=0, duration=duration)
        data = torch.from_numpy(data)
        separated_track.append(data)
    separated_track = torch.cat(separated_track)

    # Get chunks
    _, num_samples = separated_track.shape
    num_chunks = num_samples // chunk_size + \
        int(num_samples % chunk_size != 0)

    # Get available chunks
    available_chunks = []
    for i in range(num_chunks):
        chunk = separated_track[:, i * chunk_size: (i + 1) * chunk_size]
        _, chunk_samples = chunk.shape
        
        # Remove if it contains less than the minimum chunk size
        if chunk_samples < chunk_size:
            continue

        # Remove if even one stem is silent
        silent = False
        for stem in range(chunk.shape[0]):
            if is_silent(chunk[stem]):
                silent = True
                break
        if silent:
            continue

        available_chunks.append(i)
    return available_chunks


class MultiSourceDataset(Dataset):
    def __init__(self, sr, channels, min_duration, max_duration, aug_shift, sample_length, audio_files_dir, stems, transform=None, split="train"):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration or math.ceil(sample_length / sr)
        self.max_duration = max_duration or math.inf
        self.sample_length = sample_length
        self.audio_files_dir = os.path.join(audio_files_dir, split)
        self.stems = stems
        self.split = split
        self.last_track = ""
        self.multitracks: dict[str, list[str]] = {} # Dict of track name to list of stems (including mix)
        self.chunks: list[tuple[str, int]] = [] # List of tuples of track name and chunk index
        assert (
            sample_length / sr < self.min_duration
        ), f"Sample length {sample_length} per sr {sr} ({sample_length / sr:.2f}) should be shorter than min duration {self.min_duration}"
        self.aug_shift = aug_shift
        self.transform = transform if transform is not None else _identity
        self.init_dataset()

    def load_metadata(self, tracks: list[str]):
        """
        Load metadata from dataset into a dictionary.
        Structure of dictionary:
            metadata (dict): Dictionary containing metadata for each track.
            instruments (list): List of all instruments in dataset.

        Args:
            tracks (list): List of track names to load metadata for.
        """
        # For each Track* folder in dataset
        metadata = {}
        for track in tracks:
            # Load metadata from metadata.json
            metadata_path = os.path.join(
                self.audio_files_dir, track, "metadata.yaml")
            with open(metadata_path, "r") as f:
                metadata[track] = yaml.load(f, Loader=yaml.FullLoader)

        # For each Track* folder in dataset
        multitracks: dict[str, list[str]] = {}
        for track in metadata:
            # Wavs to process are all stems with common instruments + mix
            mix_path = os.path.join(self.audio_files_dir, track, f"mix{DATA_TYPE}")
            multitracks[track] = [mix_path]
            for instrument in self.stems[1:]:
                # For each instrument append the corresponding stem (in order)
                for stem in metadata[track]["stems"]:
                    inst_name = metadata[track]["stems"][stem][INST_KEY].lower()
                    # Instrument is a regular expression
                    reg = re.compile(instrument.lower())
                    if reg.match(inst_name):
                        # If stem is a common instrument, add to wavs to process
                        stem_path = os.path.join(self.audio_files_dir, track,
                                                 "stems", stem + DATA_TYPE)
                        # Check if stem exists
                        if os.path.exists(stem_path):
                            multitracks[track].append(stem_path)
                        break
            # Remove track if it doesn't have all the instruments
            if len(multitracks[track]) != len(self.stems):
                del multitracks[track]
        # Save multitracks
        self.multitracks = multitracks

    def filter(self, tracks: list[str]) -> tuple[list[str], list[float], np.ndarray]:
        """
        Filter tracks based on duration and number of instruments.

        Args:
            tracks (list): List of track names to filter.

        Returns:
            tuple: Tuple containing:
                tracks (list): List of track names to keep.
                durations (list): List of track durations.
                cumsum (np.ndarray): Cumulative sum of durations.
        """
        # Load metadata
        self.load_metadata(tracks)

        # Remove files too short or too long
        keep = []
        durations = []

        for track in self.multitracks.keys():
            # Only take N_TRACKS
            if N_TRACKS > 0 and len(keep) >= N_TRACKS:
                break
            track_dir = os.path.join(self.audio_files_dir, track)
            files = librosa.util.find_files(f"{track_dir}", ext=[DATA_TYPE[1:]])

            # skip if there are no sources per track
            if not files:
                continue

            durations_track = np.array([get_duration_sec(
                file, cache=True) * self.sr for file in files])  # Could be approximate

            # skip if there is a source that is shorter than minimum track length
            if (durations_track / self.sr < self.min_duration).any():
                continue

            # skip if there is a source that is longer than maximum track length
            if (durations_track / self.sr >= self.max_duration).any():
                continue

            # skip if in the track the different sources have different lengths
            if not (durations_track == durations_track[0]).all():
                print(f"{track} skipped because sources are not aligned!")
                print(durations_track)
                continue
            keep.append(track)
            durations.append(durations_track[0])

        print(
            f"self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")
        tracks = keep
        durations = durations
        cumsum = np.cumsum(np.array(durations))
        return keep, durations, cumsum

    def init_dataset(self):
        # Load tracks and chunks from file if they exist
        if os.path.exists(os.path.join(self.audio_files_dir, "tracks.txt")) and \
            os.path.exists(os.path.join(self.audio_files_dir, "chunks.txt")):
            with open(os.path.join(self.audio_files_dir, "tracks.txt"), "r") as f:
                for line in f.readlines()[:N_TRACKS]:
                    track = line.split(" ")[0]
                    data = line.strip("\n")[len(track)+1:]
                    self.multitracks[track] = ast.literal_eval(data)

            if os.path.exists(os.path.join(self.audio_files_dir, "blacklist.txt")):
                with open(os.path.join(self.audio_files_dir, "blacklist.txt"), "r") as f:
                    for line in f.readlines():
                        track = line.strip()
                        if track in self.multitracks:
                            del self.multitracks[track]
            else:
                print("No blacklist found.")

            with open(os.path.join(self.audio_files_dir, "chunks.txt"), "r") as f:
                for line in f.readlines():
                    track, offset = line.strip("\n").split(" ")
                    if track in self.multitracks:
                        self.chunks.append((track, int(offset)))
            return
        
        # Load list of tracks and starts/durations
        tracks = os.listdir(self.audio_files_dir)
        print(f"Found {len(tracks)} tracks.")

        # Filter tracks based on duration and number of instruments
        tracks, durations, cumsum = self.filter(tracks)
        self.tracks = tracks
        self.durations = durations

        # For each track, get non-silent chunks for each instrument
        for i, track in enumerate(self.tracks):
            if self.split == "test":
                chunks = get_all_chunks(
                    self.multitracks[track],
                    self.durations[i],
                    self.sample_length
                )
            else:
                chunks = get_nonsilent_and_multi_instr_chunks(
                    self.multitracks[track],
                    self.durations[i],
                    self.sample_length
                )
                
            if chunks:
                self.chunks.extend([(track, chunk) for chunk in chunks])
                print(f"{track} has {len(chunks)} chunks")
            else:
                # Remove track if it doesn't have any chunks
                del self.multitracks[track]

        # Save tracks to file
        with open(os.path.join(self.audio_files_dir, "tracks.txt"), "w") as f:
            for track, data in self.multitracks.items():
                f.write(f"{track} {data}\n")

        # Save chunks to file
        with open(os.path.join(self.audio_files_dir, "chunks.txt"), "w") as f:
            for chunk in self.chunks:
                f.write(f"{chunk[0]} {chunk[1]}\n")
        
        # Print number of tracks
        print(f"Keeping {len(self.multitracks)} of {len(tracks)} tracks")

    def get_song_chunk(self, track, offset):
        data_list = []
        for i, stem in enumerate(self.stems):
            wav_path = self.multitracks[track][i]
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"'{stem}' not found at '{wav_path}'")
            data, sr = load_audio(wav_path, sr=self.sr, offset=offset,
                                  duration=self.sample_length, approx=True)
            data = 0.5 * data[0:1, :] + 0.5 * data[1:, :]
            assert data.shape == (
                self.channels,
                self.sample_length,
            ), f"Expected {(self.channels, self.sample_length)}, got {data.shape}"
            data_list.append(data)
        return np.concatenate(data_list, axis=0)
    
    def get_item(self, item):
        track, chunk = self.chunks[item]
        wav = self.get_song_chunk(track, chunk * self.sample_length)
        return self.transform(torch.from_numpy(wav))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, item):
        return self.get_item(item)
