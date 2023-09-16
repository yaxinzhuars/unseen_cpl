# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'MLP'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = '/gypsum/home/yaxinzhu/concept_g/neural_collaborative_filtering/Data/'

# train_rating = main_path + '{}.train.rating'.format(dataset)
# test_rating = main_path + '{}.test.rating'.format(dataset)
# test_negative = main_path + '{}.test.negative'.format(dataset)
# train_rating = '/home/yaxinzhu/concept_g/mooc17/ML/train_mooc_unseen.index'
# test_rating = '/home/yaxinzhu/concept_g/mooc17/ML/test_mooc_unseen.index'
train_rating = '/gypsum/home/yaxinzhu/concept_g/datasets/name/uc_train_mix_boost.index'
test_rating = '/gypsum/home/yaxinzhu/concept_g/datasets/name/uc_test_mix.index'

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
