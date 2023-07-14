import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AdamW


class HateXPlainTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def forward(self, input):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = input
        return self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask
        )
    
    def training_step(self, batch, batch_idx):
        _, _, decoder_input_ids, _ = batch
        outputs = self(batch)
        logits = outputs[0]
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), decoder_input_ids.view(-1))
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, _, decoder_input_ids, _ = batch
        outputs = self(batch)
        logits = outputs[0]
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), decoder_input_ids.view(-1))
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr = 0.001)