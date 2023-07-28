import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import logging

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='Corrformer for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
# parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='PointFormer',
                    help='model name, options: [Corrformer, PointFormer]')

# data loader
parser.add_argument('--data', type=str, default='moving_mnist', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/moving_mnist/', help='root path of the data file')
# parser.add_argument('--pos_filename', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='china_demo.npy', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--test_features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=int, default=0, help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--verbose', type=int, default=0, help='location of model checkpoints')
parser.add_argument('--print_every', type=int, default=100, help='print every x itr in one epoch, can be set to -1')
parser.add_argument('--log_file', type=str, default='log/bash.log', help='location of model checkpoints')
# ablation study
parser.add_argument('--wPT', type=int, default=1, help='location of model checkpoints')
parser.add_argument('--wGC', type=int, default=1, help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=10, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')

# model define
parser.add_argument('--consistency_lambda', type=float, default=1.0, help='consistency loss weight')
parser.add_argument('--c_in', type=int, default=1, help='encoder input size')
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--height', type=int, default=64, help='output size')
parser.add_argument('--width', type=int, default=64, help='output size')
# parser.add_argument('--node_num', type=int, default=100, help='number of nodes')
# parser.add_argument('--node_list', type=str, default='23,37,40', help='number of nodes for a tree')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--enc_tcn_layers', type=int, default=1, help='num of enc tcn layers')
# parser.add_argument('--dec_tcn_layers', type=int, default=3, help='num of dec tcn layers')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# parser.add_argument('--factor', type=int, default=1, help='attn factor')
# parser.add_argument('--factor_temporal', type=int, default=1, help='attn factor')
# parser.add_argument('--factor_spatial', type=int, default=1, help='attn factor')
# parser.add_argument('--distil', action='store_false',
#                     help='whether to use distilling in encoder, using this argument means not using distilling',
#                     default=True)
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--neighbor_r', type=int, default=5, help='num of neighbors = 3.14*r^2')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--temporal_type', type=str, default='index',
                    help='')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
# parser.add_argument('--pretrained_model', type=str, default='', help='pretrained model path')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multi gpu')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()
# args.node_list = [int(x) for x in args.node_list.split(',')]


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

LOG_FILE = args.log_file

class Logger:
    def __init__(self, logger=None, level=logging.INFO, setting='x'):
        self.logger = logging.getLogger(logger)
        self.logger.propagate = False  # 防止终端重复打印
        self.logger.setLevel(level)
        if not os.path.exists(os.path.dirname(LOG_FILE)):
            os.makedirs(os.path.dirname(LOG_FILE))
        fh = logging.FileHandler(LOG_FILE, 'a', encoding='utf-8')
        log_file_2 = os.path.join('./valid_results', setting, 'log.log')
        if not os.path.exists(os.path.dirname(log_file_2)):
            os.makedirs(os.path.dirname(log_file_2))
        if not os.path.exists(os.path.dirname(os.path.dirname(log_file_2))):
            os.makedirs(os.path.dirname(os.path.dirname(log_file_2)))
        fh2 = logging.FileHandler(log_file_2, 'a', encoding='utf-8')
        fh.setLevel(level)
        fh2.setLevel(level)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        fh2.setFormatter(formatter)
        sh.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(fh2)
        self.logger.addHandler(sh)
        fh.close()
        fh2.close()
        sh.close()

    def get_log(self):
        return self.logger

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_bs{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_fct{}_fcs{}_eb{}_dt{}_{}_{}'.format(
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_bs{}_dm{}_nh{}_el{}_dl{}_lr{}_wPT{}_wGC{}_gloC_mlpFullAtt_{}'.format(
            # args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            # args.node_num,
            # args.node_list,
            args.batch_size,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.learning_rate,
            args.wPT,
            args.wGC,
            # args.d_ff,
            # args.factor,
            # args.factor_temporal,
            # args.factor_spatial,
            # args.embed,
            # args.distil,
            # args.des,
            ii
        )

        logger = Logger(__name__, setting=setting).get_log()
        print('Args in experiment:')
        logger.info('Args in experiment:')
        print(args)
        logger.info(args)

        exp = Exp(args, logger)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        # if args.do_predict:
        #     logger.info('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     exp.predict(setting, True)

        torch.cuda.empty_cache()
# else:
#     ii = 0
#     setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_node{}_node{}_bs{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_fct{}_fcs{}_eb{}_dt{}_{}_{}'.format(
#         args.model_id,
#         args.model,
#         args.data,
#         args.features,
#         args.seq_len,
#         args.label_len,
#         args.pred_len,
#         args.node_num,
#         args.node_list,
#         args.batch_size,
#         args.d_model,
#         args.n_heads,
#         args.e_layers,
#         args.d_layers,
#         args.d_ff,
#         args.factor,
#         args.factor_temporal,
#         args.factor_spatial,
#         args.embed,
#         args.distil,
#         args.des, ii)
#
#     exp = Exp(args)  # set experiments
#     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#     exp.test(setting, test=1)
#     torch.cuda.empty_cache()
