import os
import torch
from torch.utils.data import DataLoader
from dataloader import MultiSourceDataset
from model import AST_DDSP
from train import train
from test import test
from config import *

def main_train():
    print('Training a audio spectrogram transformer model')
    train_loader = DataLoader(
        MultiSourceDataset(
            sr=SR, channels=CHANNELS, min_duration=MIN_DURATION,
            max_duration=MAX_DURATION, aug_shift=AUG_SHIFT, split='train',
            sample_length=LENGTH, audio_files_dir=DATASET_PATH, stems=STEMS
        ), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    val_loader = DataLoader(
        MultiSourceDataset(
            sr=SR, channels=CHANNELS, min_duration=MIN_DURATION,
            max_duration=MAX_DURATION, aug_shift=AUG_SHIFT, split='validation',
            sample_length=LENGTH, audio_files_dir=DATASET_PATH, stems=STEMS
        ), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    audio_model = AST_DDSP()

    print("\nCreating experiment directory: %s" % EXP_DIR)
    os.makedirs("%s/models" % EXP_DIR, exist_ok=True)

    print('Now starting training for {:d} epochs'.format(N_EPOCHS))

    train(audio_model, train_loader, val_loader)


def main_test():
    print('Testing audio spectrogram transformer model')
    test_loader = DataLoader(
        MultiSourceDataset(
            sr=SR, channels=CHANNELS, min_duration=MIN_DURATION,
            max_duration=MAX_DURATION, aug_shift=AUG_SHIFT, split='test',
            sample_length=LENGTH, audio_files_dir=DATASET_PATH, stems=STEMS
        ), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    audio_model = AST_DDSP()

    print("\nCreating experiment directory: %s" % EXP_DIR)
    os.makedirs("%s/models" % EXP_DIR, exist_ok=True)

    test(audio_model, test_loader)
    
if __name__ == '__main__':
    main_train()
    main_test()
