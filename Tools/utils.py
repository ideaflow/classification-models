import numpy as np
import torch

__all__ = ['AverageMeter','TopKAccuracyMetric','Callback','ModelCheckpoint']

class AverageMeter(object):
    def __init__(self, name='Loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(object):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            try:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0)
                self.corrects[i] += correct_k.item()
            except Exception as e:
                print(e)

        return self.corrects * 100. / self.num_samples


##################################
# Callback
##################################
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)
            print('model saved at score: %f'%current_score)