import torch
import torch.nn as nn
from transformers import BertModel
import pytorch_lightning as pl

class BertClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, classifications, sentences = batch

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, classifications)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, classifications, sentences = batch

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, classifications)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer