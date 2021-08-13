import wandb
import torch
import os

from loggers.exp_logger import ExperimentLogger
import numpy as np

import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt



class Logger(ExperimentLogger):
    """Characterizes a Tensorboard logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)
        self.table_dict = {}
        wandb.run.name = exp_name

    def log(self, name, value):
        wandb.log({name: value})

    def log_scalar(self, task, iter, name, value, group=None, curtime=None, log_iter=False):
        if not log_iter:
            wandb.log({"task_%s/%s/%s" % (task, group, name): value})
        if log_iter:
            wandb.log({"task_%s/%s/%s" % (task, group, name): value, "iter/task_%s/%s/%s" % (task, group, name): iter})

    def log_figure(self, name, iter, figure, curtime=None, log_iter=False):
        if not log_iter:
            wandb.log({"%s" % (name): wandb.Image(figure)})
        if log_iter:
            wandb.log({"%s" % (name): wandb.Image(figure), "iter/%s" % (name): iter})

    def log_args(self, args):
        wandb.config.update(args.__dict__)

    def log_result(self, array, name, step):
        if array.ndim == 1:
            wandb.log({"results/%s" % (name): array[step]})

        elif array.ndim == 2:
            if name not in self.table_dict:
                columns = ["Task"]
                columns.extend(["T" + str(i) for i in range(array.shape[1])])
                columns.extend(["Average"])
                self.table_dict[name] = pd.DataFrame(columns=columns)

            args = {}
            i = step
            args["Task"] = i
            for j in range(array.shape[1]):
                args["T"+str(j)] = '{:.3f}'.format(100 * array[i, j])

            args["Average"] = '0'

            if name.startswith("fwt") and i < len(array) - 1:
                args["Average"] = '{:5.3f}'.format(100 * array[i, i+1:].mean()) # equiv. to /N-i
            elif (name.startswith("bwt") or name.startswith("forg")) and i > 0:
                args["Average"] = '{:5.3f}'.format(100 * array[i, :i].mean()) # equiv. to /i-1
            elif name[0:4] not in ["bwt", "fwt", "for"]:
                args["Average"] = '{:5.3f}'.format(100 * array[i, :i+1].mean()) # equiv. to /i

            self.table_dict[name] = self.table_dict[name].append(args, ignore_index=True)

            fig =  ff.create_table(self.table_dict[name])
            fn = os.path.join(self.exp_path, "tmp_image_%s.png" % (name))
            fig.write_image(fn)
            
            wandb.log({"results/%s" % (name): [wandb.Image(plt.imread(fn))]})
            
    def save_model(self, state_dict, task):
        filename = os.path.join(self.exp_path, "models", "task{}.ckpt".format(task))
        torch.save(state_dict, filename)
        wandb.save(filename)

    def save_array(self, array, filename):
        fn = os.path.join(self.exp_path, filename)
        np.savetxt(fn, array, delimiter=",", fmt="%.3e")
        wandb.save(fn)

    def __del__(self):
        pass
