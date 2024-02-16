from config import *
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def last_checkpoint():
    """
    Get path to last checkpoint and optimizer state

    Returns:
        (str, str): path to last checkpoint, path to last optimizer state
    """
    path = EXP_DIR + "/models/"
    # Get latest checkpoint
    checkpoints = os.listdir(path)
    checkpoints = [int(checkpoint.split(".")[1]) for checkpoint in checkpoints]
    checkpoints.sort()

    # If no checkpoints found, return empty string
    if len(checkpoints) == 0:
        print("No checkpoints found")
        return "", ""

    # Get path to last checkpoint
    model_name = "audio_model." + str(checkpoints[-1]) + ".pth"
    opt_name = "optim_state." + str(checkpoints[-1]) + ".pth"
    print("Found checkpoint: ", model_name, opt_name)
    return path + model_name, path + opt_name
