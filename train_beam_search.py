import json
import gzip
#import nvidia_smi
import os
import pytorch_lightning as pl
import random
import sacrebleu
import torch
import torch.nn.functional as F

#from google.colab import drive

from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from typing import Dict
from typing import List
from typing import Tuple
from sacrebleu.metrics import BLEU

from sklearn.metrics import f1_score
import numpy as np
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

def get_dataset(dataset):
    output = []
    for key, val in dataset.items():
        if val.get('abstract', False) and len(val['abstract']) > 0 and val.get('sections', False) and len(val['sections']) > 0:
            x = key + ' [SEP] ' + val['abstract']
            y = ' [SEP] '.join(val['sections'])
            output.append((x, y))

    return output


class MyDataset(Dataset):
    def __init__(self, text_pairs: List[Tuple[str]], tokenizer,
                 source_max_length: int = 32, target_max_length: int = 32):
        self.tokenizer = tokenizer
        self.text_pairs = text_pairs
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        source, target = self.text_pairs[idx]

        # task_prefix = 'translate English to Portuguese: '
        source_tokenizer_output = self.tokenizer(source, truncation=True,
                                                 padding='max_length',
                                                 max_length=self.source_max_length,
                                                 return_tensors='pt')
        target_tokenizer_output = self.tokenizer(target, truncation=True,
                                                 padding='max_length',
                                                 max_length=self.target_max_length,
                                                 return_tensors='pt')

        source_token_ids = source_tokenizer_output['input_ids'].squeeze(0)
        target_token_ids = target_tokenizer_output['input_ids'].squeeze(0)

        source_mask = source_tokenizer_output['attention_mask'].squeeze(0)
        target_mask = target_tokenizer_output['attention_mask'].squeeze(0)

        original_source = source
        original_target = target

        return (source_token_ids, source_mask, target_token_ids, target_mask,
                original_source, original_target)


def compute_f1_score(predicted_list: list, target_list: list) -> float:
    """
    Given the lists of target and predicted sequences, it returns the F1-Score
    :param predicted_list: list of predicted sequence
    :param target_list: list of target sequence
    :return: f1_score
    """

    scores = []
    for predicted, target in zip(predicted_list, target_list):
        predicted = [w.strip() for w in predicted.split('[SEP]')]
        target = [w.strip() for w in target.split('[SEP]')]

        # let target and predicted sequences with the same size
        diff_len = len(predicted) - len(target)
        if diff_len > 0:
            target += diff_len * [""]
        elif diff_len < 0:
            predicted += abs(diff_len) * [""]

        scores.append(f1_score(target, predicted, average='macro'))

    return np.array(scores).mean()


class T5Finetuner(pl.LightningModule):

    def __init__(self, model_name, learning_rate, source_max_length,
                 target_max_length, batch_size):
        super(T5Finetuner, self).__init__()

        if 't5' in model_name:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
        elif 'bert-' in model_name:
            from transformers import BertTokenizer, TFBertForPreTraining
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = TFBertForPreTraining.from_pretrained(model_name)
        else:
            raise NotImplementedError()

        self.learning_rate = learning_rate
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model

        self.bleu = BLEU()

        self.log_examples = True

        self.save_hyperparameters()

        self.wandb_table = wandb.Table(
            columns=['Epoch', 'Source', 'Target', 'Predicted'])

    def forward(self, source_token_ids, source_mask, target_token_ids=None,
                target_mask=None):

        if self.training:
            loss = self.model(input_ids=source_token_ids,
                              attention_mask=source_mask,
                              labels=target_token_ids).loss
            return loss
        else:
            generated_ids = self.model.generate(input_ids=source_token_ids,
                                                attention_mask=source_mask,
                                                max_length=self.target_max_length, num_beams=3, early_stopping=True, no_repeat_ngram_size=2)
            return generated_ids

    def training_step(self, batch, batch_nb):
        source_token_ids, source_mask, target_token_ids, target_mask, _, _ = batch

        # fwd
        loss = self(source_token_ids, source_mask, target_token_ids,
                    target_mask)

        # logs
        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True,
                 logger=True)

        tensorboard_logs = {'train_loss': loss.detach()}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        avg_bleu, f1 = self.get_scores(batch, batch_nb, True)
        loss = self.get_loss(batch, batch_nb)
        return {'val_bleu': avg_bleu, 'val_loss': loss, 'val_f1': f1}

    def test_step(self, batch, batch_nb):
        avg_bleu, f1 = self.get_scores(batch, batch_nb, False)
        loss = self.get_loss(batch, batch_nb)
        return {'test_bleu': avg_bleu, 'test_loss': loss, 'test_f1': f1}

    def get_scores(self, batch, batch_nb, is_test):
        source_token_ids, source_mask, target_token_ids, target_mask, original_source, original_target = batch

        generated_ids = self(source_token_ids, source_mask, target_token_ids,
                             target_mask)

        output_seq = self.tokenizer.batch_decode(generated_ids,
                                                 skip_special_tokens=True)

        avg_bleu = self.bleu.corpus_score(output_seq, [original_target]).score
        f1 = compute_f1_score(output_seq, original_target)

        if self.log_examples & is_test:
            self.wandb_table.add_data(self.current_epoch, original_source[:1],
                                      original_target[:1], output_seq[:1])

        return avg_bleu, f1

    def get_loss(self, batch, batch_nb):
        source_token_ids, source_mask, target_token_ids, target_mask, original_source, original_target = batch

        loss = self.model(input_ids=source_token_ids,
                          attention_mask=source_mask,
                          labels=target_token_ids).loss
        return loss

    def validation_epoch_end(self, outputs):
        avg_bleu = sum([x['val_bleu'] for x in outputs]) / len(outputs)
        avg_f1 = sum([x['val_f1'] for x in outputs]) / len(outputs)
        avg_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)

        self.log("avg_val_bleu", avg_bleu, prog_bar=True)
        self.log("avg_val_f1", avg_f1, prog_bar=True)
        self.log("avg_val_loss", avg_loss.detach(), prog_bar=True)

    def test_epoch_end(self, outputs):
        avg_bleu = sum([x['test_bleu'] for x in outputs]) / len(outputs)
        avg_f1 = sum([x['test_f1'] for x in outputs]) / len(outputs)
        avg_loss = sum([x['test_loss'] for x in outputs]) / len(outputs)

        self.log("avg_test_bleu", avg_bleu, prog_bar=True)
        self.log("avg_test_f1", avg_f1, prog_bar=True)
        self.log("avg_test_loss", avg_loss.detach(), prog_bar=True)

        wandb.log({'validation_samples': self.wandb_table})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate, eps=1e-08)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000,
                                                    gamma=1.0)  # This is the same as no LR decay.

        return {'optimizer': optimizer, 'lr_scheduler': scheduler,
                'monitor': 'avg_val_bleu'}

    def train_dataloader(self):
        # TODO place dataset into own module
        dataset_train = MyDataset(text_pairs=x_train,
                                  tokenizer=self.tokenizer,
                                  source_max_length=self.source_max_length,
                                  target_max_length=self.target_max_length)
        train_dataloader = DataLoader(dataset_train,
                                      batch_size=self.batch_size,
                                      shuffle=True, num_workers=0)
        return train_dataloader

    def val_dataloader(self):
        dataset_val = MyDataset(text_pairs=x_val,
                                tokenizer=self.tokenizer,
                                source_max_length=self.source_max_length,
                                target_max_length=self.target_max_length)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)

        return val_dataloader

    def test_dataloader(self):
        # TODO change to real test
        dataset_val = MyDataset(text_pairs=x_val,
                                tokenizer=self.tokenizer,
                                source_max_length=self.source_max_length,
                                target_max_length=self.target_max_length)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)
        return val_dataloader

if __name__ == '__main__':

    available_gpus = [torch.cuda.get_device_name(i) for i in
                      range(torch.cuda.device_count())]
    print("GPUs")
    print(available_gpus)
    print("-"*100)

    seed = 123
    random.seed(seed)
    # np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    path = '/work/gabriel.santos/autowiki/'
    with open(path + 'train.json', 'r') as f:
        data = json.load(f)

    dataset = get_dataset(data)
    random.shuffle(dataset)

    train_len = int(len(dataset)*.8)
    x_train = dataset[:train_len]
    x_val = dataset[train_len:]

    # Configurações gerais
    model_name = "t5-small"
    batch_size = 64
    accumulate_grad_batches = 2
    source_max_length = 128
    target_max_length = 128
    learning_rate = 1e-3

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    dataset_train = MyDataset(text_pairs=x_train,
                              tokenizer=tokenizer,
                              source_max_length=source_max_length,
                              target_max_length=target_max_length)

    dataset_val = MyDataset(text_pairs=x_val,
                            tokenizer=tokenizer,
                            source_max_length=source_max_length,
                            target_max_length=target_max_length)

    # dataset_test = MyDataset(text_pairs=x_test,
    #                          tokenizer=tokenizer,
    #                          source_max_length=source_max_length,
    #                          target_max_length=target_max_length)

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=15)

    val_dataloader = DataLoader(dataset_val, batch_size=batch_size,
                                shuffle=False,
                                num_workers=15)

    # test_dataloader = DataLoader(dataset_test, batch_size=batch_size,
    #                              shuffle=False, num_workers=0)

    # TODO change to real test
    test_dataloader = val_dataloader

    max_epochs = 100

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_monitor = EarlyStopping(monitor="avg_val_loss", min_delta=0.00,
                                  patience=0, mode="min")
    wandb_logger = WandbLogger(project="autowiki_final")

    checkpoint_path = r"./checkpoints/t5_128_beam_search/checkpoints.ckpt"
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'Files in {checkpoint_dir}: {os.listdir(checkpoint_dir)}')
    print(f'Saving checkpoints to {checkpoint_dir}')
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          save_top_k=3, monitor='avg_val_loss',
                                          mode='min')

    resume_from_checkpoint = None
    if os.path.exists(checkpoint_path):
        print(f'Restoring checkpoint: {checkpoint_path}')
        resume_from_checkpoint = checkpoint_path

    trainer = pl.Trainer(gpus=-1,
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=1,
                         accumulate_grad_batches=accumulate_grad_batches,
                         callbacks=[checkpoint_callback, lr_monitor,
                                    early_monitor],
                         progress_bar_refresh_rate=50,
                         resume_from_checkpoint=resume_from_checkpoint,
                         logger=wandb_logger,
                         accelerator='dp')

    model = T5Finetuner(model_name=model_name,
                        learning_rate=learning_rate,
                        source_max_length=source_max_length,
                        target_max_length=target_max_length,
                        batch_size=batch_size)

    trainer.fit(model)
    trainer.test(ckpt_path='best')

    wandb.finish()
