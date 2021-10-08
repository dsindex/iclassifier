# ------------------------------------------------------------------------------ #
# source from https://github.com/tbung/naive-bayes-layer
# ------------------------------------------------------------------------------ #

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNaiveBayes(nn.Module):
    def __init__(self, features, classes, fix_variance=False):
        super(self.__class__, self).__init__()

        self.features = features
        self.classes = classes

        # We need mean and variance per feature and class
        self.register_buffer(
            "means",
            Variable(torch.Tensor(self.classes, self.features))
        )
        if not fix_variance:
            self.register_parameter(
                "variances",
                nn.Parameter(torch.Tensor(self.classes, self.features))
            )
        else:
            self.register_buffer(
                "variances",
                Variable(torch.Tensor(self.classes, self.features))
            )

        # We need the class priors
        self.register_parameter(
            "class_priors",
            nn.Parameter(torch.Tensor(self.classes))
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.means.data = torch.eye(self.classes, self.features)
        self.variances.data.fill_(1)
        # self.variances.data = torch.eye(self.classes, self.features)
        self.class_priors.data.uniform_()

    def forward(self, x):
        x = x[:,np.newaxis,:]
        return (torch.sum(- 0.5 * torch.log(2 * pi * torch.abs(self.variances))
                - (x - self.means)**2 / torch.abs(self.variances) / 2, dim=-1)
                + torch.log(self.class_priors))

