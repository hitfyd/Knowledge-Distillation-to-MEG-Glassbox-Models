import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, min_train_epochs=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
            min_train_epochs (int): Minimum train epochs before counting starts.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_model_parameters = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.min_epochs = min_train_epochs

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.update_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 满足最小训练轮数后再进行早停计数
            if epoch > self.min_epochs:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # print('early_stop epoch', epoch, 'best_epoch:', self.best_epoch, 'best_score:', self.best_score)
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.update_checkpoint(val_loss, model)
            self.counter = 0

    def update_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_parameters = model.state_dict()
        self.val_loss_min = val_loss
