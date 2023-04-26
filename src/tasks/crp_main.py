import os
import torch
import torch.nn as nn
import collections
import numpy as np
from torch.utils.data.dataloader import DataLoader
import src.tasks.Decoder as Decoder
# import function_change_color as function
from src.param import args
from src.tasks.crp_model import CRPModel
from src.tasks.data_pre import CRPTorchDataset, CRPEvaluator
from torch.optim.lr_scheduler import StepLR
import itertools
import time

from prefetch_generator import BackgroundGenerator
from src.visualize import get_plot

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class DataloaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

DataTuple = collections.namedtuple("DataTuple", 'loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    data_set = CRPTorchDataset(splits)
    evaluator = CRPEvaluator(data_set)
    data_loader = DataloaderX(
        data_set, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(loader=data_loader, evaluator=evaluator)


class CRP:
    def __init__(self):
        # 返回数据集loader
        if args.train == None:
            self.train_tuple = None
        else:
            self.train_tuple = get_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True)


        if args.valid == None:
            self.valid_tuple = None
        else:
            self.valid_tuple = get_tuple(args.valid, bs=args.batch_size,shuffle=False, drop_last=False)

        self.model = CRPModel(args.nums_class).to(device)#加入类别
        self.decoder = Decoder.Decoder().to(device)
        self.mlp =Decoder.MLP(args.nums_class).to(device)

        # Losses and optimizer
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            from src.lxrt.optimization import BertAdam

            self.optim = BertAdam(self.model.parameters(),lr=args.lr,warmup=0.1,)

        else:
            self.optim_mlp = args.optimizer(self.mlp.parameters(),args.lr)
            self.optim_model = args.optimizer(self.model.parameters(),args.lr)
            self.optim_decoder = args.optimizer(self.decoder.parameters(),args.lr)

        self.scheduler_mlp = StepLR(self.optim_mlp, step_size=10, gamma=0.5) #5
        self.scheduler_model = StepLR(self.optim_model,step_size=10,gamma=0.5)
        self.scheduler_decoder = StepLR(self.optim_decoder,step_size=10,gamma=0.5)

        self.output = args.output
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
        self.dropout = nn.Dropout(0.5)

    def train(self, train_tuple, eval_tuple):
        train_loader, _ = train_tuple
        valid_loader, _ = eval_tuple
        epoch_loss_train = []
        epoch_acc_train = []
        epoch_loss_val = []
        epoch_acc_val = []
        best_accuracy = 0.
        print("Begin Training....")
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            loss_all = 0.
            loss_r = 0.
            loss_v = 0.
            accuracy = 0.
            cnt = 0.

            for i, data in enumerate(train_loader):
                self.model.train()
                self.decoder.train()
                self.mlp.train()

                self.optim_mlp.zero_grad()
                self.optim_decoder.zero_grad()

                vis_feature = data[0].to(device)
                rad_feature = data[1].to(device)
                targets = data[2].to(device)

                # --------fusion--------------
                feat_r, feat_v, indices_r, indices_v, ori_r, ori_v = self.model(rad_feature, vis_feature)
                feat = torch.cat((feat_r, feat_v), dim=2)
                feat_r_c = feat_r.view(feat_r.size()[0], -1)
                feat_v_c = feat_v.view(feat_v.size()[0], -1)
                feat_c = torch.cat((feat_r_c, feat_v_c), dim=1)

                predictions = self.mlp(feat_c)

                batch_loss_G_r, batch_loss_G_v = self.decoder(feat, indices_r, indices_v, ori_r, ori_v)
                loss = self.mce_loss(predictions, targets)

                batch_loss_all = 0.98 * loss + 0.001 * batch_loss_G_r + 0.001 * batch_loss_G_v
                batch_loss_all.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.)

                self.optim_mlp.step()
                self.optim_decoder.step()
                self.optim_model.step()

                with torch.no_grad():
                    loss_all += batch_loss_all.sum().item()
                    loss_r += batch_loss_G_r.sum().item()
                    loss_v += batch_loss_G_r.sum().item()
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)

            self.scheduler_mlp.step()
            self.scheduler_model.step()
            self.scheduler_decoder.step()

            loss_all /= cnt
            loss_r /= cnt
            loss_v /= cnt
            accuracy *= 100. / cnt
            print(f"Epoch: {epoch + 1}, Train accuracy: {accuracy:6.2f} %, Train loss_all: {loss_all:8.5f}, Train loss_r: {loss_r:8.5f}, Train loss_v: {loss_v:8.5f}")
            epoch_loss_train.append(loss_all)
            epoch_acc_train.append(accuracy)

#----------------eval---------------
            self.model.eval()
            self.decoder.eval()
            self.mlp.eval()
            accuracy = 0.
            cnt = 0.
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    vis_feature = data[0].to(device)
                    rad_feature = data[1].to(device)
                    targets = data[2].to(device)

                    # --------fusion--------------
                    feat_r, feat_v, indices_r, indices_v, ori_r, ori_v = self.model(rad_feature, vis_feature)
                    feat = torch.cat((feat_r, feat_v), dim=2)
                    feat_r_c = feat_r.view(feat_r.size()[0], -1)
                    feat_v_c = feat_v.view(feat_v.size()[0], -1)
                    feat_c = torch.cat((feat_r_c, feat_v_c), dim=1)

                    predictions = self.mlp(feat_c)

                    batch_loss_G_r, batch_loss_G_v = self.decoder(feat, indices_r, indices_v, ori_r, ori_v)
                    loss = self.mce_loss(predictions, targets)

                    batch_loss_all = 0.98 * loss + 0.001 * batch_loss_G_r + 0.001 * batch_loss_G_v
                    with torch.no_grad():
                        loss_all += batch_loss_all.sum().item()
                        loss_r += batch_loss_G_r.sum().item()
                        loss_v += batch_loss_G_r.sum().item()
                        accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                    cnt += len(targets)

                loss_all /= cnt
                loss_r /= cnt
                loss_v /= cnt
                accuracy *= 100. / cnt

                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    torch.save(self.modesl.state_dict(), args.output + args.expid_name + '_best_ckpt.pt')
                    print("Check point " + args.output + args.expid_name + '_best_ckpt.pt' + ' Saved!')

            print(f"Epoch: {epoch + 1}, Test accuracy: {accuracy:6.2f}, Test loss: {loss_all:8.5f}")


            epoch_loss_val.append(loss_all)
            epoch_acc_val.append(accuracy)

        print(f"Best test accuracy: {best_accuracy}")
        print("TRAINING COMPLETED :)")

         # Save visualization
        get_plot(args.output, epoch_acc_train, epoch_acc_val, 'Accuracy-' +  args.expid_name, 'Train Accuracy', 'Val Accuracy',
                 'Epochs', 'Acc')
        get_plot(args.output, epoch_loss_train, epoch_loss_val, 'Loss-' + args.expid_name, 'Train Loss', 'Val Loss', 'Epochs', 'Loss')


if __name__ == "__main__":
    # Build Class
    crp = CRP()

    # Load Model
    if args.load is not None:
        # print("args.load", args.load)
        crp.load_ed(args.load)

    print('Splits in Train data:', args.train)
    if crp.valid_tuple is not None:
        print('Splits in Valid data:', args.valid)
    else:
        print("DO NOT USE VALIDATION")
    crp.train(crp.train_tuple,crp.valid_tuple)


