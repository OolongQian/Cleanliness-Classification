import itertools
import random

import numpy as np
import scipy
import scipy.io
from PIL import Image
from scipy.spatial import Delaunay


def train_test_split(X, Y, train_ratio):
	tot = X.shape[0]
	num_train = int(tot * train_ratio)
	num_test = tot - num_train
	assert num_train > 0
	assert num_test > 0
	train_ids = random.sample(list(range(tot)), num_train)
	test_ids = [idx for idx in range(tot) if idx not in train_ids]
	
	X_train = X[train_ids]
	X_test = X[test_ids]
	Y_train = Y[train_ids]
	Y_test = Y[test_ids]
	
	return X_train, Y_train, X_test, Y_test


def read_metadata():
	return scipy.io.loadmat('./data/OFFICIAL_SUNRGBD/SUNRGBDMeta')['SUNRGBDMeta']


def get_spatial(id, meta):
	id -= 1  # note matlab id is 1 greater than python index.
	gt3DBB_struct = meta[0][id][1]
	
	try:
		gt3DBBs = gt3DBB_struct[0]
	except IndexError:
		gt3DBBs = []
	
	num_3DBB = len(gt3DBBs)
	
	centroids = []
	basises = []
	coefficients = []
	orientations = []
	items = []
	if len(gt3DBBs) != 0:
		for i in range(num_3DBB):
			centroids.append(gt3DBBs[i][2][0])  # (1, 3) shape centroid.
			basises.append(gt3DBBs[i][0][0])  # (3, 3)
			coefficients.append(gt3DBBs[i][1][0])  # (3, 3)
			orientations.append(gt3DBBs[i][6][0])
			items.append(gt3DBBs[i][3][0])
	
	return basises, coefficients, centroids, orientations, items


def delaunay_triangulate(P: np.ndarray):
	n = P.shape[0]
	if n <= 5:
		A = fully_connect(P)
	else:
		d = Delaunay(P)
		A = np.zeros((n, n))
		for simplex in d.simplices:
			for pair in itertools.permutations(simplex, 2):
				A[pair] = 1
	return A


def fully_connect(P: np.ndarray):
	n = P.shape[0]
	A = np.ones((n, n)) - np.eye(n)
	return A


def get_multi_label():
	lbs = {}
	arr_lbs = {}
	vis_lbs = {}
	lb_cnt = [0, 0]
	with open('./label/multi_label.txt', 'r') as f:
		for line in f.readlines():
			id, lb, arr_lb, vis_lb = line.split(' ')
			id, lb, arr_lb, vis_lb = int(id), int(lb), int(arr_lb), int(vis_lb)
			if id == 1572:
				continue
			lbs[id] = lb
			arr_lbs[id] = arr_lb
			vis_lbs[id] = vis_lb
			lb_cnt[lb] += 1
	
	ids = list(lbs.keys())
	ids.sort()
	num_instance = len(ids)
	Y = np.zeros(shape=(num_instance,))
	Y_arr = np.zeros(shape=(num_instance,))
	Y_vis = np.zeros(shape=(num_instance,))
	
	for i, id in enumerate(ids):
		Y[i] = int(lbs[id])
		Y_arr[i] = int(arr_lbs[id])
		Y_vis[i] = int(vis_lbs[id])
	return Y, Y_arr, Y_vis


def get_train_visual_data():
	lbs = {}
	lb_cnt = [0, 0]
	with open('./label/label.txt', 'r') as f:
		for line in f.readlines():
			id, lb = line.split(' ')
			id, lb = int(id), int(lb)
			if id == 1572:
				continue
			lbs[id] = lb
			lb_cnt[lb] += 1
	
	ids = list(lbs.keys())
	ids.sort()
	
	num_instance = len(ids)
	width = 400
	height = 400
	
	X = np.zeros(shape=(num_instance, height, width, 3))
	X_cntr = np.zeros(shape=(num_instance, height, width))
	Y = np.zeros(shape=(num_instance,))
	
	print('begin loading image visual data...')
	for i, id in enumerate(ids):
		img = Image.open('./data/total_images/{}.png'.format(id))
		img = np.array(img.resize((width, height), Image.BICUBIC))
		X[i] = img / 64.
		cntr = Image.open('./data/contour/{}.png'.format(id))
		cntr = np.array(cntr.resize((width, height), Image.BICUBIC))
		X_cntr[i] = cntr / 64.
		Y[i] = int(lbs[id])
	
	X_cntr = X_cntr[:, np.newaxis, :, :]
	print('done')
	return X, X_cntr, Y


def get_train_furniture_data():
	print('begin loading furniture data...')
	meta = read_metadata()
	lbs = {}
	lb_cnt = [0, 0]
	with open('./label/label.txt', 'r') as f:
		for line in f.readlines():
			id, lb = line.split(' ')
			id, lb = int(id), int(lb)
			if id == 1572:
				continue
			lb_cnt[lb] += 1
			lbs[id] = lb
	ids = list(lbs.keys())
	ids.sort()
	
	num_instance = len(ids)
	# chair, table, desk, garbage_bin, cabinet, sofa, sofa_chair, endtable, dining_table, coffee_table.
	cat2objs = {0: ['chair', 'sofa_chair', 'sofa', 'garbage_bin', 'bench', 'stool'],
	            1: ['table', 'desk', 'cabinet', 'endtable', 'dining_table', 'coffee_table']}
	obj2cat = {}
	for key, val in cat2objs.items():
		for obj in val:
			obj2cat[obj] = key
	
	max_num_object = 50
	data = np.zeros((num_instance, max_num_object, 2 + 1 + 3 + 3))  # one-hot category, centroids, orientations, size.
	adj_mat = np.zeros((num_instance, max_num_object, max_num_object))
	edge_mat = np.zeros((num_instance, max_num_object, max_num_object, 3))
	Y = np.zeros(shape=(num_instance,))
	
	for i in range(num_instance):
		id = ids[i]
		Y[i] = float(lbs[id])
		basises, coefficients, centroids, orientations, items = get_spatial(id, meta)
		
		inval_idx = [j for j, item in enumerate(items) if item not in obj2cat.keys()]
		inval_idx.sort()
		for j in reversed(inval_idx):
			del basises[j]
			del coefficients[j]
			del centroids[j]
			del orientations[j]
			del items[j]
		
		coefficients = np.array(coefficients)
		centroids = np.array(centroids)
		orientations = np.array(orientations)
		
		for j, item in enumerate(items):
			cat_hot = np.zeros(shape=(2,))
			cat_hot[obj2cat[item]] = 1  # fill up one-hot category.
			centroid = centroids[j]
			orientation = orientations[j]
			size = np.sqrt(np.sum(np.square(coefficients[j])))[np.newaxis]
			feature = np.concatenate([cat_hot, size, centroid, orientation])
			data[i][j] = feature
		
		adj = delaunay_triangulate(centroids)
		for u in range(adj.shape[0]):
			for v in range(adj.shape[1]):
				if adj[u][v] == 1:
					disp = centroids[v] - centroids[u]
					length = np.linalg.norm(disp)  # displacement from neighbor j to current i.
					disp_angle = np.deg2rad(np.rad2deg(np.arctan2(disp[1], disp[0])) - np.rad2deg(
						np.arctan2(orientations[u][1], orientations[u][0])))
					orit_angle = np.deg2rad(np.rad2deg(np.arctan2(orientations[v][1], orientations[v][0])) - np.rad2deg(
						np.arctan2(orientations[u][1], orientations[u][0])))
					edge_mat[i][u][v][0] = length
					edge_mat[i][u][v][1] = disp_angle
					edge_mat[i][u][v][2] = orit_angle
					adj_mat[i][u][v] = 1
	print('done')
	return data, adj_mat, edge_mat, Y
