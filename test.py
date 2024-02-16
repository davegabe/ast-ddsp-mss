import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))  # nopep8
from src.utilities import *
import time
import torch
from torch import nn
import numpy as np
from loss import StemsMultiResolutionSTFTLoss
from tqdm import tqdm
from train import save_audio
from config import *

def test(audio_model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    audio_model = audio_model.to(device)
    exp_dir = EXP_DIR
    
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    # Load the model
    model_checkpoint, opt_checkpoint = last_checkpoint()
    if model_checkpoint != "" and opt_checkpoint != "":
        audio_model.load_state_dict(torch.load(model_checkpoint))
        epoch = int(model_checkpoint.split(".")[1])
        global_step = epoch * len(test_loader)
    else:
        raise FileNotFoundError("No checkpoints found.")
        
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start testing...")

    loss_fn = StemsMultiResolutionSTFTLoss()
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_loss = []
    with torch.no_grad():
        for i, tracks in tqdm(enumerate(test_loader)):
            B = tracks.size(0)
            tracks = tracks.to(device, non_blocking=True)
            mixture = tracks[:, 0, :]
            stems = tracks[:, 1:, :]

            # compute output
            generated_stems = audio_model(mixture)

            # compute the loss
            loss = loss_fn(generated_stems, stems)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

            # Save the audio
            save_audio(
                mixture.to('cpu').detach().numpy(),
                generated_stems.to('cpu').detach().numpy(),
                stems.to('cpu').detach().numpy(), f"{exp_dir}/audio/test/{i}"
            )

        loss = np.mean(A_loss)
    return loss
