import wandb

class WandBLog():
    def __init__(self, model):
        wandb.init()
        wandb.watch(model)

    def getWandB(self):
        return wandb

    # def log(self, name, parametr):
    #     self({name : })
    #     wandb.log({"mean val loss:": numpy.mean(val_loss)})
