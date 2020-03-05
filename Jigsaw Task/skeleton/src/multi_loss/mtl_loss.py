import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
	"""Implementation of multi task loss from the paper"""

	def __init__(self, classifier_loss, ae_loss):
		super(MultiTaskLoss, self).__init__()
		self.cl_loss = classifier_loss
		self.ae_loss = ae_loss

	def forward(self, output, log_vars):
		l1 = self.cl_loss(output)
		l2 = self.ae_loss(output)
		loss = [l1, l2]

		total_loss = l1 * torch.exp(- log_vars[0]) + log_vars[0]
		total_loss += l2 * torch.exp(- log_vars[1]) + log_vars[1]
		return total_loss, log_vars, loss