import torch
import pytorch_lightning as pl

from dataset import HateXplainDataModule
from model import BertClassifier
    
def classify(list) -> str:
    if list.tolist() == [1, 0, 0]:
        return "Normal"
    elif list.tolist() == [0, 1, 0]:
        return "Offensive"
    elif list.tolist() == [0, 0, 1]:
        return "Hate Speech"
    else:
        raise Exception("Invalid format of list: ", list)
    
def print_some_samples_from_dataset():
    data_module = HateXplainDataModule()
    data_module.setup()

    test_dataloader = data_module.test_dataloader()

    count = 0
    for batch in test_dataloader:
        
        input_ids, attention_mask, classifications, sentences = batch

        for i in range(len(sentences)):
            print("Sample", count + 1)
            print("Sentence:", sentences[i])

            print("Classification:", classify(classifications[i]))
            print()

            count += 1
            if count == 5:
                break
        
        if count == 5:
            break

def test_classifier():

    # Initialize the data module
    data_module = HateXplainDataModule(batch_size=32)
    data_module.setup()

    # Initialize the model
    model = BertClassifier(num_classes=3)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=3)

    # Train the model
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

def main():
    # print_some_samples_from_dataset()
    test_classifier()
        
if __name__ == "__main__":
    main()