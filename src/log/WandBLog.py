import wandb

class WandBLog():
    def __init__(self, model):
        wandb.init()
        wandb.watch(model)

    def getWandB(self):
        return wandb
