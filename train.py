import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))  # nopep8
from src.utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
from loss import StemsMultiResolutionSTFTLoss
from tqdm import tqdm
import soundfile as sf
import wandb
from config import *


def train(audio_model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)
    wandb.init(project="ast-ddsp-mss")
    wandb.watch(audio_model, log_freq=100)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = EXP_DIR

    def _save_progress():
        progress.append([epoch, global_step, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(
        trainables, LR, weight_decay=5e-7, betas=(0.95, 0.999))
    
    # Load the model
    model_checkpoint, opt_checkpoint = last_checkpoint()
    if model_checkpoint != "" and opt_checkpoint != "":
        audio_model.load_state_dict(torch.load(model_checkpoint))
        optimizer.load_state_dict(torch.load(opt_checkpoint))
        epoch = int(model_checkpoint.split(".")[1])
        global_step = epoch * len(train_loader)

    warmup = WARMUP
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(
        range(LR_SCHEDULER_START, 1000, LR_SCHEDULER_STEP)), gamma=LR_SCHEDULER_DECAY, last_epoch=epoch if epoch > 1 else -1)
    loss_fn = StemsMultiResolutionSTFTLoss()
    print('now training with loss function: {:s}, learning rate scheduler: {:s}'.format(str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(
        LR_SCHEDULER_START, LR_SCHEDULER_DECAY, LR_SCHEDULER_STEP))

    epoch += 1
    # # for amp
    # scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    audio_model.train()
    while epoch < N_EPOCHS + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, tracks in tqdm(enumerate(train_loader)):
            B = tracks.size(0)
            tracks = tracks.to(device, non_blocking=True)
            mixture = tracks[:, 0, :]
            stems = tracks[:, 1:, :]

            data_time.update(time.time() - end_time)
            per_sample_data_time.update(
                (time.time() - end_time) / mixture.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print(
                    'warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                generated_stems = audio_model(mixture)
                loss = loss_fn(generated_stems, stems)

            # optimization if amp is not used
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # optimiztion if amp is used
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/mixture.shape[0])
            per_sample_dnn_time.update(
                (time.time() - dnn_start_time)/mixture.shape[0])

            print_step = global_step % N_PRINT_STEPS == 0
            early_print_step = epoch == 0 and global_step % (
                N_PRINT_STEPS/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                      'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                      'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                      'Train Loss {loss_meter.avg:.4f}\t'.format(
                          epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                          per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    raise Exception("Training diverged...")

            end_time = time.time()
            global_step += 1
        
        # Save the model
        if epoch % SAVE_EPOCHS == 0:
            torch.save(audio_model.state_dict(),
                    "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(),
                    "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
            # Remove old checkpoints
            checkpoints = os.listdir("%s/models" % exp_dir)
            checkpoints = [int(checkpoint.split(".")[1]) for checkpoint in checkpoints]
            checkpoints = list(set(checkpoints))
            checkpoints.sort()
            if len(checkpoints) > MAX_CHECKPOINTS:
                for checkpoint in checkpoints[:-MAX_CHECKPOINTS]:
                    os.remove("%s/models/audio_model.%d.pth" % (exp_dir, checkpoint))
                    os.remove("%s/models/optim_state.%d.pth" % (exp_dir, checkpoint))

        # Validation
        if epoch % VALIDATION_EPOCHS == 0:
            print('start validation')
            valid_loss = validate(audio_model, test_loader, epoch, loss_fn, exp_dir)

            print("valid_loss: {:.6f}".format(valid_loss))
            wandb.log({"valid_loss": valid_loss}, step=epoch)
            print('validation finished')

        print("train_loss: {:.6f}".format(loss_meter.avg))
        wandb.log({"loss": loss_meter.avg}, step=epoch)

        if LR_SCHEDULER:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch,
              optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(
            epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()


def validate(audio_model, val_loader, epoch, loss_fn, exp_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_loss = []
    with torch.no_grad():
        for i, tracks in tqdm(enumerate(val_loader)):
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
            if loss < 1.5 and epoch % SAVE_EPOCHS == 0:
                save_audio(
                    mixture.to('cpu').detach().numpy(),
                    generated_stems.to('cpu').detach().numpy(),
                    stems.to('cpu').detach().numpy(), f"{exp_dir}/audio/val_{epoch}/{i}"
                )

        loss = np.mean(A_loss)
    return loss

def save_audio(mixture, generted_stems, target_stems, file_name):
    """
    Save the audio files from the generated stems
    
    Args:
        generted_stems: the generated stems from the model
        target_stems: the target stems
        file_name: the file name of the generated stems
    """
    os.makedirs(file_name, exist_ok=True)
    mix = np.ravel(mixture)
    sf.write(file_name + '/mixture.wav', mix, SR)
    for i in range(generted_stems.shape[1]):
        gen = np.ravel(generted_stems[:, i, :])
        tar = np.ravel(target_stems[:, i, :])
        sf.write(file_name + '/target_' + str(i) + '.wav', tar, SR)
        sf.write(file_name + '/stem_' + str(i) + '.wav', gen, SR)