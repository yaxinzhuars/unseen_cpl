import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)

def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs == labels)/outputs.shape[0]

def eval_preq(model, test_loader):
	# b = torch.from_numpy(np.load('../bert/uc.npy')).cuda()
	for user, item, label, weight in test_loader:
		user = user.cuda()
		item = item.cuda()
		# user = b[user]
		# item = b[item]

		prediction, l1, pcl = model(user, item)
		# prediction = torch.nn.functional.softmax(prediction)
		# ma = prediction.max()
		# mi = prediction.min()
		# mu = prediction.mean()
		# std = prediction.std()
		# # prediction = (prediction - mi)/std
		# prediction = (prediction - mi)/(ma - mi)
		# mu = prediction.mean()
		# print(user, item, prediction)
		# print(mu)
		# p = prediction.cpu().detach().numpy()
		# for i in range(p.shape[0]):
		# 	p[i] = 1 if p[i] > mu else 0
		# gt = label.numpy()
		# print(f1_score(p, gt, average='macro'))

		logits = prediction.detach().cpu().numpy()
		# outputs = np.argmax(logits, axis=1)
		outputs = np.int32(logits>0)
		p = torch.sigmoid(prediction)
		p = p.detach().cpu().numpy()
		# print(p)

		label = label.cpu().numpy()
		eval_accuracy = np.sum(outputs == label)/outputs.shape[0]
		f1_macro = f1_score(outputs, label, average='macro')
		p_macro = precision_score(outputs, label, average='macro')
		r_macro = recall_score(outputs, label, average='macro')
		print(eval_accuracy, f1_macro, p_macro, r_macro)
		# print(outputs)
		return f1_macro, p, outputs
