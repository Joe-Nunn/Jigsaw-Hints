from trainer import Trainer
import os

cwd = os.getcwd()
jigsaw_trainer = Trainer(os.path.join(cwd, "JigsawDataset"), "data.csv")
jigsaw_trainer.train()
