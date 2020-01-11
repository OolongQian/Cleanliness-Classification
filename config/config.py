from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.EPOCHS = 1000
__C.BATCH_SIZE = 120
__C.TEST_BATCH_SIZE = 10
__C.LR = 0.001
__C.TRAIN_TEST_SPLIT = 0.9
__C.SHUFFLE = True
__C.NUM_WORKERS = 1
__C.MODEL = 'semcnn'  # cnn, fc, cnnfc, highlevelcnn, semcnn
__C.TRAIN_MODE = 'valid'  # valid, overfit.
__C.GCN_MODE = 'fgcn'  # gcn, fgcn
__C.AUX_COEF = 0.1
