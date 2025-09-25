
from yolox.core import Trainer

class StackedTrainer(Trainer):

    def __init__(self, exp, args):
        super(StackedTrainer, self).__init__(exp, args)
        self.before_data_dir = exp.before_data_dir

    def resume_train(self, model):

        # if ckpt_file starts with 'yolox', load model.original_model
        if hasattr(model, 'original_model'):
            super(StackedTrainer, self).resume_train(model.original_model)
        else:
            super(StackedTrainer, self).resume_train(model)

        return model