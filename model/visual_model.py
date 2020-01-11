import torch
import torch.nn as nn


class VNN(nn.Module):
	def __init__(self, hdim):
		super(VNN, self).__init__()
		cnn_cfg = [4, 4]
		bias_cfg = False
		self.fc = nn.Linear(hdim, 2)
		self.fc1 = nn.Linear(hdim, 16)
		self.fc2 = nn.Linear(20, 2)
		self.vnl_cntr = nn.Sequential(# method 1.
			nn.AvgPool2d(kernel_size=(4, 4)),
			nn.Conv2d(in_channels=1, out_channels=cnn_cfg[0], kernel_size=3, stride=3, bias=bias_cfg),
			nn.BatchNorm2d(cnn_cfg[0]), nn.ReLU(True), nn.MaxPool2d(2, 2),
			nn.Conv2d(in_channels=cnn_cfg[0], out_channels=1, kernel_size=3, stride=3, bias=bias_cfg),
			nn.BatchNorm2d(1), nn.ReLU(True), nn.AvgPool2d(kernel_size=(2, 2)), )
	
	def forward(self, feat, cntr):
		use_cntr = False
		if not use_cntr:
			x = self.fc(feat)
		else:
			feat = torch.tanh(self.fc1(feat))
			cntr = torch.tanh(self.vnl_cntr(cntr))
			cntr = cntr.view(cntr.size()[0], -1)
			x = torch.cat([feat, cntr], 1)
			x = self.fc2(x)
		return x
