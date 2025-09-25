
from .trainer import Trainer

class StackedTrainer(Trainer):

    def __init__(self, exp, args):
        super(StackedTrainer, self).__init__(exp, args)
        self.before_data_dir = exp.before_data_dir

    def resume_train(self, model):

        if self.args.resume:
            super().resume_train(model)
        else:
            if self.args.ckpt is not None:
                if self.args.ckpt[:6] == "yolox":
                    super().resume_train(model.original_model)
                else:
                    super().resume_train(model)
            self.start_epoch = 0

        return model