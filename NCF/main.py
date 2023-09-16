import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils

torch.manual_seed(1)
torch.cuda.manual_seed(1)
# lb: 8 seed=0

# 0.0005
parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.0001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=40,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99999, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()

# for i in train_data:
# 	print(i)

# drop_num = int(0.2*user_num)
# idx_perm = np.random.permutation(user_num)

# idx_drop = idx_perm[:drop_num]
# idx_nondrop = idx_perm[drop_num:]
# idx_nondrop.sort()
# idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

# train_mat_aug = train_mat[idx_nondrop, :][:, idx_nondrop]
# train_data_aug_ = train_mat_aug.nonzero()
# train_data_aug = []
# for i in range(train_data_aug_[0].size):
# 	train_data_aug.append((train_data_aug_[0][i], train_data_aug_[1][i]))

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

# for i in test_loader:
# 	print(i)

# train_dataset_aug = data_utils.NCFData(
# 		train_data_aug, item_num, train_mat_aug, args.num_ng, True)
# train_loader_aug = data.DataLoader(train_dataset_aug,
# 		batch_size=args.batch_size, shuffle=True, num_workers=4)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()
# loss_function = nn.CrossEntropyLoss()

if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)


# writer = SummaryWriter() # for visualization

torch.set_printoptions(profile='full')

def loss_cl(x1, x2):
	T = 0.5
	batch_size, _ = x1.size()
	
	# batch_size *= 2
	# x1, x2 = torch.cat((x1, x2), dim=0), torch.cat((x2, x1), dim=0)

	x1_abs = x1.norm(dim=1)
	x2_abs = x2.norm(dim=1)

	'''
	sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
	sim_matrix = torch.exp(sim_matrix / T)
	pos_sim = sim_matrix[range(batch_size), range(batch_size)]
	self_sim = sim_matrix[range(batch_size), list(range(int(batch_size/2), batch_size))+list(range(int(batch_size/2)))]
	loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim - self_sim)
	loss = - torch.log(loss).mean()
	'''
	
	sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
	sim_matrix = torch.exp(sim_matrix / T)
	pos_sim = sim_matrix[range(batch_size), range(batch_size)]
	loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
	loss = - torch.log(loss).mean()
	
	return loss
########################### TRAINING #####################################
def train():
	count, best_f1 = 0, 0
	# b = torch.from_numpy(np.load('../bert/uc.npy')).cuda()
	for epoch in range(args.epochs):
		model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample()
		# train_loader_aug.dataset.ng_sample()

		w = []
		for user, item, label, weight in train_loader:
			# print(user, item, label)
			user = user.cuda()
			item = item.cuda()
			# user = b[user]
			# item = b[item]
			label = label.float().cuda()
			# label = label.cuda()

			model.zero_grad()
			# prediction = model(user, item)
			prediction, l1, pcl = model(user, item)
			# loss = loss_function(prediction, label.view(-1))
			loss = loss_function(prediction, label) + l1*0
			loss.backward()
			optimizer.step()
			# writer.add_scalar('data/loss', loss.item(), count)
			count += 1
			pred = (prediction>0).float()
			check = (pred==label).float()
			for i in range(user.size(0)):
				w.append([user[i].item(), item[i].item(), int(label[i].item()), int(check[i].item())])



		# for t1, t2 in zip(train_loader, train_loader_aug):
		# 	user, item, label, weight = t1[0], t1[1], t1[2], t1[3]
		# 	user = user.cuda()
		# 	item = item.cuda()
		# 	label = label.float().cuda()

		# 	model.zero_grad()
		# 	prediction, l1, pcl = model(user, item)

		# 	user1, item1, label1, weight1 = t2[0], t2[1], t2[2], t2[3]
		# 	user1 = user.cuda()
		# 	item1 = item.cuda()
		# 	label1 = label.float().cuda()

		# 	model.zero_grad()
		# 	prediction1, l11, pcl1 = model(user1, item1)

		# 	loss = loss_function(prediction, label) + l1*0
		# 	loss.backward()
		# 	optimizer.step()

		# 	loss = loss_function(prediction1, label1) + l11*0
		# 	# loss = 0*loss_cl(pcl, pcl1) + (l+ll) * 1
		# 	# loss.backward()
		# 	loss.backward()
		# 	optimizer.step()
		# 	# writer.add_scalar('data/loss', loss.item(), count)
		# 	count += 1
		# 	# pred = (prediction>0).float()
		# 	# check = (pred==label).float()
		# 	# for i in range(user.size(0)):
		# 	# 	w.append([user[i].item(), item[i].item(), int(label[i].item()), int(check[i].item())])


		model.eval()
		# HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
		f1_macro, logits, outputs = evaluate.eval_preq(model, test_loader)

		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		# print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

		if f1_macro > best_f1:
			best_f1, best_epoch = f1_macro, epoch
			# with open('w_uc.txt', 'w') as fw:
			# 	for line in w:
			# 		fw.write('\t'.join([str(i) for i in line]) + '\n')
			if args.out:
				if not os.path.exists(config.model_path):
					os.mkdir(config.model_path)
				torch.save(model, 
					'{}{}.uc_mix.pth'.format(config.model_path, config.model))

	# print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
	# 									best_epoch, best_hr, best_ndcg))

def eval():
	model = torch.load('{}{}.uc_mix.pth'.format(config.model_path, config.model))
	model.eval()
	f1_macro, logits, outputs = evaluate.eval_preq(model, test_loader)

	# model1 = torch.load('{}{}.uc_e0_seed6.pth'.format(config.model_path, config.model))
	# model1.eval()

	# assert user_num == item_num
	
	i2c = {}
	with open('../bert/uc_i2c.txt') as f:
		for line in f.readlines():
			i, c = line.strip().split('\t')
			i2c[int(i)] = c
	item_num = len(i2c)

	for c1 in range(item_num):
		test_examples = []
		for c2 in range(item_num):
			test_examples.append([c1, c2, 0])
		test_dataset_row = data_utils.NCFData(
			test_examples, item_num, train_mat, 0, False)
		test_loader_row = data.DataLoader(test_dataset_row,
			batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)
		f1_macro, logits, outputs = evaluate.eval_preq(model, test_loader_row)
		with open ('output_uc_mix.txt', 'a') as fw:
			for i in range(len(test_examples)):
				c1, c2, _ = test_examples[i]
				# fw.write(i2c[c1] + '\t' + i2c[c2] + '\t' + str(logits[i][0]) + '\t' + str(logits[i][1]) + '\t' + str(outputs[i]) + '\n')
				fw.write(i2c[c1] + '\t' + i2c[c2] + '\t' + str(1-logits[i]) + '\t' + str(logits[i]) + '\t' + str(outputs[i]) + '\n')

	# for c1 in range(item_num):
	# 	test_examples = []
	# 	for c2 in range(item_num):
	# 		test_examples.append([c1, c2, 0])
	# 	test_dataset_row = data_utils.NCFData(
	# 		test_examples, item_num, train_mat, 0, False)
	# 	test_loader_row = data.DataLoader(test_dataset_row,
	# 		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)
	# 	f1_macro, logits, outputs = evaluate.eval_preq(model, test_loader_row)
	# 	f1_macro1, logits1, outputs1 = evaluate.eval_preq(model1, test_loader_row)
		# with open ('output_uc_decouple.txt', 'a') as fw:
		# 	for i in range(len(test_examples)):
		# 		c1, c2, _ = test_examples[i]
		# 		# fw.write(i2c[c1] + '\t' + i2c[c2] + '\t' + str(logits[i][0]) + '\t' + str(logits[i][1]) + '\t' + str(outputs[i]) + '\n')
		# 		# if outputs[i] != outputs1[i]:
		# 		if outputs[i] == 1 and outputs1[i] == 0:
		# 			if logits[i] > 1 - logits1[i]:
		# 				fw.write(i2c[c1] + '\t' + i2c[c2] + '\t' + str(1-logits[i]) + '\t' + str(logits[i]) + '\t' + str(outputs[i]) + '\n')
		# 			else:
		# 				fw.write(i2c[c1] + '\t' + i2c[c2] + '\t' + str(1-logits1[i]) + '\t' + str(logits1[i]) + '\t' + str(outputs1[i]) + '\n')
		# 		elif outputs[i] == 0 and outputs1[i] == 1:
		# 			if logits[i] < 1 - logits1[i]:
		# 				fw.write(i2c[c1] + '\t' + i2c[c2] + '\t' + str(1-logits[i]) + '\t' + str(logits[i]) + '\t' + str(outputs[i]) + '\n')
		# 			else:
		# 				fw.write(i2c[c1] + '\t' + i2c[c2] + '\t' + str(1-logits1[i]) + '\t' + str(logits1[i]) + '\t' + str(outputs1[i]) + '\n')

# train()
eval()
