import argparse
import os, sys
from pathlib import Path
import pickle
import json
import time

import numpy as np
import tqdm

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.common import *
from utils.vis import *
from utils.tsa import *

random_seed = 0
np.random.seed(random_seed)

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, precision_recall_curve

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torchmetrics
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

TIMESERIES_LEN = 48
ENCODE_DIM = 6
ENCODE_ELEM = ['mean', 'var', 'encoded1', 'encoded2', 'encoded3', 'encoded4']

class PatientDatasetRaw(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        df_sample = self.data[idx]

        mortality = df_sample['mortality'].iloc[-1]
        x_label = np.array([df_sample['LOS'].iloc[-1]])
        if mortality == 1:
            x_label = torch.Tensor([4])
        else:
            if x_label < 5*24:
                x_label = torch.Tensor([0])
            elif (x_label >= 5*24) and (x_label < 10*24):
                x_label = torch.Tensor([1])
            elif (x_label >= 10*24) and (x_label < 20*24):
                x_label = torch.Tensor([2])
            else:
                x_label = torch.Tensor([3])

        x_info = torch.tensor([df_sample[col].iloc[0] for col in col_info_numeric], dtype=torch.float)
        for col in col_info_text:
            ohk = F.one_hot(torch.tensor(df_sample[col].iloc[0].astype(int)), num_classes=len(info_text_vals[col]))
            x_info = torch.cat((x_info, ohk), dim=0)


        x_data = torch.tensor(df_sample[col_chart_numeric].to_numpy(dtype=float), dtype=torch.float)
        for col in col_chart_text:
            num_classes = len(item_text_vals[col])
            ohk = torch.zeros((df_sample.shape[0], len(item_text_vals[col]))).long()
            ohk[~pd.isna(df_sample[col])] = F.one_hot(
                torch.tensor(df_sample[~pd.isna(df_sample[col])][col].astype(float).to_numpy(), dtype=torch.long),
                num_classes=num_classes
            )
            x_data = torch.cat((x_data, ohk.float()), dim=1)


        return {
            'data': {
                'static': x_info,
                'time_series': x_data,
            },
            'label': x_label,
        }


class PatientDatasetEncoded(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        df_sample = self.data[idx].astype(float)

        mortality = df_sample['mortality'].iloc[-1]
        x_label = np.array([df_sample['LOS'].iloc[-1]])
        if mortality == 1:
            x_label = torch.Tensor([4])
        else:
            if x_label < 5*24:
                x_label = torch.Tensor([0])
            elif (x_label >= 5*24) and (x_label < 10*24):
                x_label = torch.Tensor([1])
            elif (x_label >= 10*24) and (x_label < 20*24):
                x_label = torch.Tensor([2])
            else:
                x_label = torch.Tensor([3])

        x_info = torch.tensor([df_sample[col].iloc[0] for col in col_info_numeric], dtype=torch.float)
        for col in col_info_text:
            ohk = F.one_hot(torch.tensor(df_sample[col].iloc[0].astype(int)), num_classes=len(info_text_vals[col]))
            x_info = torch.cat((x_info, ohk), dim=0)

        x_data = torch.empty(TIMESERIES_LEN, 0)
        for col in col_chart_numeric:
            col_encoded = [col+'_'+elem for elem in ENCODE_ELEM]
            x_data = torch.cat((x_data, torch.tensor(df_sample[col_encoded].to_numpy(dtype=float), dtype=torch.float)), dim=1)

        for col in col_chart_text:
            num_classes = len(item_text_vals[col])
            ohk = torch.zeros((df_sample.shape[0], len(item_text_vals[col]))).long()
            ohk[~pd.isna(df_sample[col])] = F.one_hot(
                torch.tensor(df_sample[~pd.isna(df_sample[col])][col].astype(float).to_numpy(), dtype=torch.long),
                num_classes=num_classes
            )
            x_data = torch.cat((x_data, ohk.float()), dim=1)

        return {
            'data': {
                'static': x_info,
                'time_series': x_data,
            },
            'label': x_label,
        }


class CNN_CLF(LightningModule):
    def __init__(self, config, data, data_type, device=0):
        super().__init__()
        # self.automatic_optimization = False

        self.data_type = data_type
        if data_type == 'raw':
            self.dataset = PatientDatasetRaw
        elif data_type == 'encoded':
            self.dataset = PatientDatasetEncoded
        else:
            raise ValueError(f'Unsupported data_type: {data_type}')

        self.config = config
        self.data = data

        self.cnn_layer1 = nn.Conv1d(
            in_channels=config['cnn_in_channels'][0],
            out_channels=config['cnn_out_channels'][0],
            kernel_size=config['cnn_kernel_size'][0],
            stride=config['cnn_stride_size'][0],
            groups=config['cnn_group_size'][0],
        ).cuda(device)
        self.cnn_layer2 = nn.Conv1d(
            in_channels=config['cnn_in_channels'][1],
            out_channels=config['cnn_out_channels'][1],
            kernel_size=config['cnn_kernel_size'][1],
            stride=config['cnn_stride_size'][1],
            groups=config['cnn_group_size'][1],
        ).cuda(device)
        for i in range(config['n_cnn_layer']):
            setattr(
                self, f'avg_pool{i+1}',
                nn.AvgPool1d(
                    kernel_size=config['avgpool_kernel_size'][i],
                    stride=config['avgpool_stride_size'][i],
                ).cuda(device)
            )

        self.fc1 = nn.LazyLinear(out_features=256).cuda(device)
        self.fc2 = nn.LazyLinear(out_features=64).cuda(device)
        self.fc3 = nn.LazyLinear(out_features=5).cuda(device)

        self.loss = nn.CrossEntropyLoss().cuda(device)

        self.METRICS = {
            'acc_micro': torchmetrics.Accuracy(multiclass=True, average='micro', num_classes=5).cpu(),
            'acc': torchmetrics.Accuracy(multiclass=True, average='macro', num_classes=5).cpu(),
            'recall': torchmetrics.Recall(multiclass=True, average='macro', num_classes=5).cpu(),
            'precision': torchmetrics.Precision(multiclass=True, average='macro', num_classes=5).cpu(),
            'specification': torchmetrics.Specificity(multiclass=True, average='macro', num_classes=5).cpu(),
            'roc': torchmetrics.AUROC(num_classes=5).cpu(),
            # 'prc': torchmetrics.PrecisionRecallCurve(pos_label=1).cpu(),
            # 'prc': precision_recall_curve,
            'f1': torchmetrics.F1(multiclass=True, average='macro', num_classes=5).cpu(),
        }

    def forward(self, x):
        # x_static = x_static.float()
        # x_sequence = x_seq.float().permute(0,2,1)
        x_static = x['static'].float()
        x_ts = x['time_series'].float()
        batch_size = x_static.shape[0]
        if self.data_type == 'raw':
            x_ts_num = x_ts[:, :, :38]
            x_ts_cat = x_ts[:, :, 38:]
        elif self.data_type == 'encoded':
            x_ts_num = x_ts[:, :, :228]
            x_ts_cat = x_ts[:, :, 228:]

        x_ts_num = x_ts_num.permute(0, 2, 1)
        x_ts_cat = x_ts_cat.permute(0, 2, 1)

        for i in range(1, self.config['n_cnn_layer']+1):
            x_ts_num = F.relu(getattr(self, f'cnn_layer{i}')(x_ts_num))
            x_ts_cat = getattr(self, f'avg_pool{i}')(x_ts_cat)

        x = torch.concat((x_static.reshape(batch_size, -1),
                          x_ts_num.reshape(batch_size, -1),
                          x_ts_cat.reshape(batch_size, -1)),
                         dim=1)

        x = F.dropout(F.relu(self.fc1(x)), p=self.config['dropout'])
        x = F.dropout(F.relu(self.fc2(x)), p=self.config['dropout'])
        logit = self.fc3(x)
        prob = F.softmax(logit)

        return logit, prob


    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(), lr=self.config['lr'])
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, 'min')
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(adam, T_0=5, T_mult=2, eta_min=1e-6,)
        # return [adam], [lr_scheduler]
        return adam

    # def training_epoch_end(self, outputs):
    #     sch = self.lr_schedulers()
    #     # If the selected scheduler is a ReduceLROnPlateau scheduler.
    #     if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #         sch.step(self.trainer.callback_metrics["loss"])

    def training_step(self, batch, batch_idx):
        # lr_sch = self.lr_schedulers()
        # lr_sch.step()
        x = batch['data']
        y = batch['label'].squeeze()
        x_logit, pred = self.forward(x)
        loss = self.loss(x_logit, y.long())

        torch.cuda.empty_cache()

        outputs = {'train_loss': loss}
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            # if metric == 'prc':
            #     precision, recall, _ = precision_recall_curve(y.int().cpu().detach().numpy(), pred.cpu().detach().numpy())
            #     auprc = auc(recall, precision)
            #     outputs["train_"+metric] = auprc
            # else:
            outputs["train_"+metric] = self.METRICS[metric](pred.detach().cpu(), y.long().detach().cpu())
            self.log("train_" + metric, outputs["train_"+metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # with profile(activities=[ ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        x = batch['data']
        y = batch['label'].squeeze()
        # print(x['static'].device)
        x_logit, pred = self.forward(x)
        loss = self.loss(x_logit, y.long())

        torch.cuda.empty_cache()

        outputs = {'val_loss': loss}
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            # if metric == 'prc':
            #     precision, recall, _ = precision_recall_curve(y.int().cpu().detach().numpy(), pred.cpu().detach().numpy())
            #     auprc = auc(recall, precision)
            #     outputs["val_"+metric] = auprc
            # else:
            outputs["val_"+metric] = self.METRICS[metric](pred.detach().cpu(), y.long().detach().cpu())
            self.log("val_"+metric, outputs["val_"+metric], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return outputs

    def test_step(self, batch, batch_idx):
        x = batch['data']
        y = batch['label'].squeeze()
        x_logit, pred = self.forward(x)
        loss = self.loss(x_logit, y.long())

        torch.cuda.empty_cache()

        outputs = {'test_loss': loss}
        self.log("test_loss", loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.METRICS:
            # if metric == 'prc':
            #     precision, recall, _ = precision_recall_curve(y.int().cpu().detach().numpy(), pred.cpu().detach().numpy())
            #     auprc = auc(recall, precision)
            #     outputs["test_"+metric] = auprc
            # else:
            outputs["test_" + metric] = self.METRICS[metric](pred.detach().cpu(), y.long().detach().cpu())
            self.log("test_" + metric, outputs["test_" + metric], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)
        return outputs

    def train_dataloader(self):
        data_train = self.data['train']
        ds_train = self.dataset(data_train)
        loader_train = DataLoader(ds_train, batch_size=self.config['batch_size'], num_workers=32)
        return loader_train

    def test_dataloader(self):
        data_test = self.data['test']
        ds_test = self.dataset(data_test)
        loader_test = DataLoader(ds_test, batch_size=self.config['batch_size'], num_workers=32)
        return loader_test

    def val_dataloader(self):
        data_val = self.data['val']
        ds_val = self.dataset(data_val)
        loader_val = DataLoader(ds_val, batch_size=self.config['batch_size'], num_workers=32)
        return loader_val







if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--datatype",
        nargs="*",
        type=str,
        default=['raw'],
    )
    CLI.add_argument(
        "--gpu",
        nargs="*",
        type=int,
        default=[5],
    )
    CLI.add_argument(
        "--seed",
        nargs="*",
        type=int,
        default=[999],
    )
    args = CLI.parse_args()

    data_type = args.datatype[0]
    device = args.gpu[0]
    seed = args.seed[0]

    torch.manual_seed(seed)


    # training_data_raw = pickle.load(open('/home/kai/workspace/data/mimic/training_data_raw.pkl', 'rb'))
    # training_data_encoded = pickle.load(open('/home/kai/workspace/data/mimic/training_data_encoded.pkl', 'rb'))
    # training_data_raw_zero_imp = []
    # training_data_encoded_zero_imp = []
    #
    # zero_imp = {}
    # for col in zero_imp_numeric:
    #     zero_imp[col] = zero_imp_numeric[col]
    #     zero_imp[col+'_mean'] = zero_imp[col]
    #     zero_imp[col+'_var'] = norm_param_numeric[col]['std']**2
    #
    # zero_imp = dict(zero_imp, **zero_imp_info)
    # for i in tqdm.tqdm(range(len(training_data_encoded))):
    #     df_sample = training_data_raw[i].astype(float).copy()
        # df_sample.fillna(value=zero_imp, inplace=True)
        # training_data_raw_zero_imp.append(df_sample)

        # df_sample = training_data_encoded[i].astype(float).copy()
        # df_sample.fillna(value=zero_imp, inplace=True)
        # training_data_encoded_zero_imp.append(df_sample)
    #
    # pickle.dump(training_data_raw_zero_imp, open('/home/kai/workspace/data/mimic/training_data_raw_0imp.pkl', 'wb'))
    # pickle.dump(training_data_encoded_zero_imp, open('/home/kai/workspace/data/mimic/training_data_encoded_0imp.pkl', 'wb'))


    data_file_raw = '/home/kai/workspace/data/mimic/training_data_raw_0imp_with_mortality.pkl'
    data_file_encoded = '/home/kai/workspace/data/mimic/training_data_encoded_0imp_with_mortality.pkl'
    data_folder = os.path.abspath('../data_all/')

    print(type(data_type), data_type, type(device), device)
    # data_type = 'raw'
    if data_type == 'raw':
        training_data = pickle.load(open(data_file_raw, 'rb'))
    elif data_type == 'encoded':
        training_data = pickle.load(open(data_file_encoded, 'rb'))

    data_train_all, data_test = train_test_split(training_data, test_size=.2, random_state=random_seed, shuffle=True)
    num_splits = 5
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    kf_splits = [k for k in kf.split(data_train_all)]

    # model setup
    n_cnn_layer = 2
    if data_type == 'raw':
        cnn_in_channels = [38, 380]
    elif data_type == 'encoded':
        cnn_in_channels = [228, 380]
    cnn_out_channels = [380, 64]
    cnn_kernel_size = [8, 5]
    cnn_stride_size = [2, 2]
    cnn_group_size = [38, 1]
    avgpool_kernel_size = [8, 5]
    avgpool_stride_size = [2, 2]
    config = {
        "n_cnn_layer": n_cnn_layer,
        "cnn_in_channels": cnn_in_channels,
        "cnn_out_channels": cnn_out_channels,
        "cnn_kernel_size": cnn_kernel_size,
        "cnn_stride_size": cnn_stride_size,
        "cnn_group_size": cnn_group_size,
        "avgpool_kernel_size": avgpool_kernel_size,
        "avgpool_stride_size": avgpool_stride_size,
        "dropout": .5,
        "lr": 1e-4,
        "batch_size": 64,
    }

    model_name = f"{data_type}" + f"_2layer_multiclass"
    model_folder = f'../models/CNN_LOS_PRED/{model_name}'
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    model_version = os.listdir(model_folder)
    version = 0
    versions = [int(v.split('_')[1]) for v in model_version if 'version_' in v]
    if len(versions)>0:
        version = sorted(versions)[-1] + 1

    model_folder = os.path.join(model_folder, f'version_{version}')
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(model_folder, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    for kfold in range(num_splits):
        train_idx, val_idx = kf_splits[kfold]
        data_train = [data_train_all[i] for i in train_idx]
        data_val = [data_train_all[i] for i in val_idx]

        data = {
            'train': data_train,
            'val': data_val,
            'test': data_test,
        }

        modelfilename = f'model_cv{kfold}'

        callbacks = [
            ModelCheckpoint(
                monitor='val_roc',
                mode='max',
                save_top_k=1,
                dirpath=model_folder,
                filename=modelfilename+'epoch{epoch:02d}-val_roc{val_roc:.8f}',
                auto_insert_metric_name=False
            ),
            ModelCheckpoint(
                monitor='val_f1',
                mode='max',
                save_top_k=1,
                dirpath=model_folder,
                filename=modelfilename+'epoch{epoch:02d}-val_f1{val_f1:.8f}',
                auto_insert_metric_name=False
            ),
            ModelCheckpoint(
                monitor='val_recall',
                mode='max',
                save_top_k=1,
                dirpath=model_folder,
                filename=modelfilename+'epoch{epoch:02d}-val_recall{val_recall:.8f}',
                auto_insert_metric_name=False
            ),
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=30,
            )
        ]

        model = CNN_CLF(config, data, data_type, device=device)

        logger = TensorBoardLogger(model_folder,
                                   name=modelfilename,
                                   version=version)

        trainer = Trainer(
            max_epochs=1000,
            gpus=[device],
            callbacks=callbacks,
            logger=logger,
            # profiler="simple",
            # resume_from_checkpoint=os.path.join(checkpoint_dir, "checkpoint"),
        )
        train_time_start = time.time()
        trainer.fit(model)
        train_time_total = time.time() - train_time_start

        with open(os.path.join(model_folder, 'train_time.txt'), 'a') as train_time_file:
            train_time_file.write(f'{modelfilename}: {train_time_total}\n')




    print(111)