import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AdamW

class HateXPlainTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def forward(self, **inputs):
        print(inputs)
        raise Exception
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr = 0.001)