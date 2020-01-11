import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import utils


class CoolDataset(Dataset):
	def __init__(self, img, feat, cntr, data, adj, edge, label, arr_lb, vis_lb):
		"""
		Wrap numpy into a dataset.
		"""
		self.imgs = img
		self.feats = feat
		self.cntrs = cntr
		self.data = data
		self.adj = adj
		self.edge = edge
		self.labels = label
		self.arr_lbs = arr_lb
		self.vis_lbs = vis_lb
	
	def __len__(self):
		return self.imgs.shape[0]
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		samples = (
		self.feats[idx, ...], self.cntrs[idx, ...], self.data[idx, ...], self.adj[idx, ...], self.edge[idx, ...])
		labels = (self.labels[idx, ...], self.arr_lbs[idx, ...], self.vis_lbs[idx, ...])
		return samples, labels
