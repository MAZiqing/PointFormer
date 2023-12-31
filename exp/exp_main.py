from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
from models import Corrformer
from models import PointFormer
from models import cuboid_transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, simple_metric
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from einops import repeat, rearrange
from datetime import datetime

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

import os
import logging


LOG_FILE = 'log/my.log'
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE))


# """
# $ Usage:
#     from init_logger import Logger
#     logger = Logger(__name__).get_log()
# """


class Exp_Main():
    def __init__(self, args, logger):
        # super(Exp_Main, self).__init__(args)
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.logger = logger

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
                device = torch.device('cuda') #:{}'.format(self.args.devices))
                print('Use multiple GPU: [{}]'.format(self.args.devices))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: 【{}】'.format(device))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_dict = {
            'Corrformer': Corrformer,
            'PointFormer': PointFormer,
            'EarthFormer': cuboid_transformer,
        }
        self.args.device = self.device
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train_vali_test(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = repeat(batch_x.mean(dim=1), 'b h w d -> b t h w d', t=self.args.pred_len)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # metric
        f_dim = -1 if self.args.features == 'MS' else 0

        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion, setting, epoch):
        folder_path = './valid_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # break
                outputs, batch_y = self.train_vali_test(batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                # gt = np.concatenate((input[0], true[0]), axis=0)
                # pd = np.concatenate((input[0], pred[0]), axis=0)
                    folder_path_e = os.path.join(folder_path, str(epoch))
                    if not os.path.exists(folder_path_e):
                        os.makedirs(folder_path_e)
                    visual(inp[0], true[0].numpy(), pred[0].numpy(),
                           os.path.join(folder_path_e, 'epoch' + str(epoch) + '-' + str(i) + '.pdf'))
        total_loss = np.average(total_loss)
        self.model.train()
        if self.args.data == 'moving_mnist':
            total_loss = total_loss * 64 * 64
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        self.logger.info('train_steps={}'.format(train_steps))
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # break
                iter_count += 1
                model_optim.zero_grad()
                if i == 0:
                    self.logger.info('input shape: batch_x={}, batch_y={}, batch_x_mark={}, batch_y_mark={}'.format(
                        batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape
                    ))

                outputs, batch_y = self.train_vali_test(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                print_every = train_steps // 5 if self.args.print_every == -1 else self.args.print_every
                if (i + 1) % print_every == 0:
                    self.logger.info("\t | iters: {}/{}, epoch: {} | loss: {:.7f} | total_loss {:.3f}".format(
                        # str(datetime.now()),
                        i + 1,
                        train_steps,
                        epoch + 1,
                        loss.item(),
                        np.average(train_loss)))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (train_steps - i)
                    left_iter = (train_steps - i)
                    self.logger.info('\tspeed: {:.4f}s/iter; left iter: {:.4f} iter; left minutes: {:.1f}'.format(speed,
                                                                                                       left_iter,
                                                                                                       left_time // 60))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                a = 1
            # torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if self.args.data == 'moving_mnist':
                train_loss = train_loss * 64 * 64
            vali_loss = self.vali(vali_data, vali_loader, criterion, setting=setting, epoch=epoch)
            test_loss = self.vali(test_data, test_loader, criterion, setting=setting, epoch=epoch)

            self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            self.logger.info('[Learning_rate now = {}]'.format(model_optim.param_groups[0]['lr']))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info('loading model')
            if self.args.pretrained_model == '':
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            else:
                self.model.load_state_dict(
                    torch.load(os.path.join('./checkpoints/' + self.args.pretrained_model, 'checkpoint.pth')))
            self.logger.info('loading model finished')

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae = 0.0
        mse = 0.0
        batch_num = 0
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # if i > 3:
                #     break
                outputs, batch_y = self.train_vali_test(batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                # if self.args.test_features == 'S_station':
                #     pred = pred[:, :, self.args.target:(self.args.target + 1)]
                #     true = true[:, :, self.args.target:(self.args.target + 1)]
                tmp_mae, tmp_mse = simple_metric(pred, true)
                mse += tmp_mse
                mae += tmp_mae
                batch_num += 1
                # visual
                input = batch_x.detach().cpu().numpy()
                if i % 10 == 0:
                    # gt = np.concatenate((input[0], true[0]), axis=0)
                    # pd = np.concatenate((input[0], pred[0]), axis=0)
                    visual(input[0], true[0], pred[0], os.path.join(folder_path, str(i) + '.pdf'))
                if i % 10 == 0:
                    self.logger.info("batch: " + str(i))

        mse = mse / float(batch_num)
        mae = mae / float(batch_num)
        self.logger.info('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
    #
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #
    #     preds = []
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     # if self.args.output_attention:
    #                     #     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     # else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 # if self.args.output_attention:
    #                 #     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                 # else:
    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)
    #
    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     np.save(folder_path + 'real_prediction.npy', preds)
    #
    #     return
