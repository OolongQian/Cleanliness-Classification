import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class COOL(nn.Module):
	def __init__(self, hdim, nfeat, nhid, dropout):
		super(COOL, self).__init__()
		# gcn
		layers = []
		layers.append(FGraphConvolution(nfeat, nhid[0]))
		for i in range(len(nhid) - 1):
			layers.append(FGraphConvolution(nhid[i], nhid[i + 1]))
		self.gc = nn.ModuleList(layers)
		self.dropout = dropout
		
		self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 16), stride=1)
		self.fc = nn.Linear(16, 2)
		
		# visual
		bias_cfg = False
		cnn_cfg = [4, 4]
		self.fc1 = nn.Linear(hdim, 16)
		self.fc2 = nn.Linear(20, 2)
		self.vnl_cntr = nn.Sequential(nn.AvgPool2d(kernel_size=(4, 4)),
			nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=3, bias=bias_cfg), nn.BatchNorm2d(4),
			nn.ReLU(True), nn.MaxPool2d(2, 2),
			nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=3, bias=bias_cfg), nn.BatchNorm2d(1),
			nn.ReLU(True), nn.AvgPool2d(kernel_size=(2, 2)), )
		
		self.mixfc = nn.Linear(36, 12)
		self.coolfc = nn.Linear(12, 2)
		
		self.precnn_fc = nn.Linear(hdim, 2)
	
	def forward(self, feat, cntr, att, adj, edge):
		pre_cnn = True
		if pre_cnn == True:
			tmp = self.precnn_fc(feat)
			return tmp, tmp, tmp
		else:
			# gcn
			end_layer = len(self.gc)
			for i in range(end_layer):
				att = F.tanh(self.gc[i](att, edge, adj))  # att = F.dropout(att, self.dropout, training=self.training)
			fx = torch.unsqueeze(att, axis=1)
			fx = torch.relu(self.conv(fx))  # LR for nodewise feature
			fx, _ = torch.max(fx, 2)  # pooling over node dimension.
			fx = torch.squeeze(fx, 2)
			furx = self.fc(fx)
			
			# visual
			feat = torch.tanh(self.fc1(feat))
			cntr = torch.tanh(self.vnl_cntr(cntr))
			cntr = cntr.view(cntr.size()[0], -1)
			vx = torch.cat([feat, cntr], 1)
			visx = self.fc2(vx)
			
			x = torch.cat([fx, vx], 1)
			x = torch.relu(self.mixfc(x))
			x = self.coolfc(x)
			return x, furx, visx


class FGraphConvolution(Module):
	def __init__(self, in_features, out_features):
		super(FGraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.aggt = nn.Linear(in_features * 2 + 3, out_features, bias=False)
	
	def forward(self, att, edge, adj):
		assert adj.size()[1] == adj.size()[2]
		max_neighbor_num = adj.size()[1]
		a1 = torch.unsqueeze(att, 1)
		a1 = a1.repeat([1, max_neighbor_num, 1, 1])
		a2 = torch.unsqueeze(att, 2)
		a2 = a2.repeat([1, 1, max_neighbor_num, 1])
		pw = torch.cat([a2, a1, edge], 3)  # pairwise
		pw = self.aggt(pw)
		vadj = torch.unsqueeze(adj, 3)
		pw = pw * vadj
		nrml = torch.sum(vadj, 2).repeat([1, 1, self.out_features])
		pw = torch.sum(pw, 2)
		pw = pw / (nrml + 0.001)
		return pw
