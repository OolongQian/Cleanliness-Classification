import torch
import torch.nn as nn
from keras.applications.xception import Xception
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from config.config import cfg
from dataset.cool_dataset import CoolDataset
from models.cool_model import COOL
from utils.utils import *

if __name__ == '__main__':
	device = torch.device('cuda:0')
	writer = SummaryWriter("./runs/{}".format(cfg.MODEL))
	
	X, X_cntr, _ = get_train_visual_data()
	data, adj_mat, edge_mat, _ = get_train_furniture_data()
	Y, Y_arr, Y_vis = get_multi_label()
	
	pos_num = np.sum(Y == 1)
	neg_num = np.sum(Y == 0)
	
	print('begin computing pre-trained Xception feature...')
	base_model = Xception(include_top=False, weights='imagenet', pooling='max')
	X_rawfeat = base_model.predict(X)
	pca = PCA(0.95)
	pca.fit(X_rawfeat)  # fit on training set
	X_feat = pca.transform(X_rawfeat)
	print('done')
	
	X = torch.from_numpy(X).float()
	X_cntr = torch.from_numpy(X_cntr).float()
	X_feat = torch.from_numpy(X_feat).float()
	data = torch.from_numpy(data).float()
	data = data[:, :, :3]  # get attributes out of X!
	X_adj = torch.from_numpy(adj_mat).float()
	X_edge = torch.from_numpy(edge_mat).float()
	Y = torch.from_numpy(Y).long()
	Y_arr = torch.from_numpy(Y_arr).long()
	Y_vis = torch.from_numpy(Y_vis).long()
	
	net = COOL(X_feat.size()[1], 3, [16, 16], 0.9).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR)
	
	if cfg.TRAIN_MODE == 'valid':
		train_dataset = CoolDataset(X, X_feat, X_cntr, data, X_adj, X_edge, Y, Y_arr, Y_vis)
		train_size = int(len(train_dataset) * cfg.TRAIN_TEST_SPLIT)
		valid_size = len(train_dataset) - train_size
		lens = [train_size, valid_size]
		train_dataset, valid_dataset = random_split(train_dataset, lens)
		train_weights = np.array(
			[pos_num / (pos_num + neg_num) if label == 0 else neg_num / (pos_num + neg_num) for data, label in
			 train_dataset])
		train_weights = torch.from_numpy(train_weights)
		train_sampler = WeightedRandomSampler(train_weights.type('torch.DoubleTensor'), len(train_weights))
		train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=train_sampler,
		                              num_workers=cfg.NUM_WORKERS)
		valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.TEST_BATCH_SIZE, shuffle=cfg.SHUFFLE,
		                              num_workers=cfg.NUM_WORKERS)
	elif cfg.TRAIN_MODE == 'overfit':
		train_dataset = CoolDataset(X, X_feat, X_cntr, data, X_adj, X_edge, Y, Y_arr, Y_vis)
		train_weights = np.array(
			[pos_num / (pos_num + neg_num) if label == 0 else neg_num / (pos_num + neg_num) for data, label in
			 train_dataset])
		train_weights = torch.from_numpy(train_weights)
		train_sampler = WeightedRandomSampler(train_weights.type('torch.DoubleTensor'), len(train_weights))
		train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=train_sampler,
		                              num_workers=cfg.NUM_WORKERS)
	else:
		raise NotImplementedError
	
	for epoch in range(cfg.EPOCHS):
		train_loss = 0.0
		
		optimizer.zero_grad()
		for i_batch, ((X_feat_batch, X_cntr_batch, X_node_batch, X_adj_batch, X_edge_batch),
		              (Y_batch, Y_arr_batch, Y_vis_batch)) in enumerate(train_dataloader):
			X_feat_batch = X_feat_batch.to(device)
			X_cntr_batch = X_cntr_batch.to(device)
			X_node_batch = X_node_batch.to(device)
			X_adj_batch = X_adj_batch.to(device)
			X_edge_batch = X_edge_batch.to(device)
			Y_batch = Y_batch.to(device)
			Y_arr_batch = Y_arr_batch.to(device)
			Y_vis_batch = Y_vis_batch.to(device)
			
			arr_idx = (Y_arr_batch != 2)
			vis_idx = (Y_vis_batch != 2)
			
			pred, fur_pred, vis_pred = net(X_feat_batch, X_cntr_batch, X_node_batch, X_adj_batch, X_edge_batch)
			loss = criterion(pred, Y_batch)
			loss_arr = criterion(fur_pred[arr_idx], Y_arr_batch[arr_idx])
			loss_vis = criterion(vis_pred[vis_idx], Y_vis_batch[vis_idx])
			train_loss += loss.item() + cfg.AUX_COEF * loss_arr.item() + cfg.AUX_COEF * loss_vis.item()  # record training loss.
			loss.backward()
		optimizer.step()
		
		# evaluate traversing the entire dataset.
		tot = 0.0
		acc = 0.0
		test_loss = 0.0
		cnt = 0
		eval_dataloader = train_dataloader if cfg.TRAIN_MODE == 'overfit' else valid_dataloader
		for ((X_feat_batch, X_cntr_batch, X_node_batch, X_adj_batch, X_edge_batch),
		     (Y_batch, Y_arr_batch, Y_vis_batch)) in eval_dataloader:
			cnt += 1
			X_feat_batch = X_feat_batch.to(device)
			X_cntr_batch = X_cntr_batch.to(device)
			X_node_batch = X_node_batch.to(device)
			X_adj_batch = X_adj_batch.to(device)
			X_edge_batch = X_edge_batch.to(device)
			Y_batch = Y_batch.to(device)
			
			pred, _, _ = net(X_feat_batch, X_cntr_batch, X_node_batch, X_adj_batch, X_edge_batch)
			test_loss += criterion(pred, Y_batch).item()
			
			pred, _, _ = net(X_feat_batch, X_cntr_batch, X_node_batch, X_adj_batch, X_edge_batch)
			for i in range(pred.shape[0]):
				print('pred {} label {}'.format(pred[i], Y_batch[i]))
			
			_, predicted = torch.max(pred.data, 1)
			print(predicted, predicted.shape)
			print(Y_batch, Y_batch.shape)
			tot += Y_batch.size(0)
			acc += (predicted == Y_batch).sum()
		
		if cfg.TRAIN_MODE == 'valid':
			print('train_loss {} test_loss {} total {} correct {} acc {}'.format(train_loss, test_loss, tot, acc,
			                                                                     float(acc / tot)))
			writer.add_scalar('Train_Loss', train_loss, epoch)
			writer.add_scalar('Test_Loss', test_loss, epoch)
			writer.add_scalar('accuracy', acc / tot, epoch)
		elif cfg.TRAIN_MODE == 'overfit':
			print('loss {} tot {} acc {} ratio {}'.format(test_loss, tot, acc, acc / tot))
			writer.add_scalar('Loss', test_loss, epoch)
			writer.add_scalar('accuracy', acc / tot, epoch)
		else:
			raise NotImplementedError
