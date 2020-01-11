import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import utils


class VisualDataset(Dataset):
	def __init__(self, img, feat, cntr, label):
		"""
		Wrap numpy into a dataset.
		"""
		self.imgs = img
		self.feats = feat
		self.cntrs = cntr
		self.labels = label
	
	def __len__(self):
		return self.imgs.shape[0]
	
	def get_feature(self):
		return self.feats
	
	def get_label(self):
		return self.labels
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		samples = (self.feats[idx, ...], self.cntrs[idx, ...])
		return samples, self.labels[idx, ...]
