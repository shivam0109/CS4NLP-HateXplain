# CS4NLP-HateXplain
Project Repo of Computational Semantics for NLP - Hateful Speech Detection with explanations

# Current workflow
This table summarizes the current worflow followed for hate-speech detection: 

## Preprocessing 
* A sample consists of a list of words, multiple annotations whether the given the sentence is hate speech as well as a list of words that were used to arrive at this annotation
* Such an annotation can either be 0 for Hate Speech, 1 for normal and 2 for Hate Speech
* Given these annotations we calculate an avg_score as the mean of all annotations
* If the average score is greater or equal to 0.666, the sample is labeled as "Hate Speech", if the score is between 0.333 and 0.666 the sample is labeled as offensive, else the sample is labeled as normal
* Having derived these labels, we continue with deriving the rationales by concatenating the list of words used to a standard phrase for each of the three labeled classes:
  * **Hate/Offensive speech (and word rationales available):** Rationale: This sentence was classified as {Hate/Offensive speech} because it contains harmful words such as: {list of words}.
  * **Hate/Offensive speech (and no word rationales available):** Rationale: This sentence has a harmful tone and was thus classified as {Hate/Offensive speech}.
  * **Normal Speech:** Rationale: This sentence was classified as normal because it contains no harmful words.
* The words representing the sentence are merged into a sentence and then tokenized using t5-small


## Training

Project Repo of Computational Semantics for NLP - Hateful Speech Detection with explanations. 

To reproduce the results, access is required for the following links:  [this](https://drive.google.com/drive/folders/1U_L-GvtMUyER5DInpKh-lonXVkjDh4mF?usp=sharing) and [this](https://drive.google.com/drive/folders/1Q0fhtHM3sM4AHegkOHcEgEBunyb6SWDC?usp=sharing)  

Datasets and trained models for BERT can be accessed through [this link](https://drive.google.com/drive/folders/1U_L-GvtMUyER5DInpKh-lonXVkjDh4mF?usp=sharing)    

Datasets and trained models for T5 can be accessed through [this link](https://drive.google.com/drive/folders/1Q0fhtHM3sM4AHegkOHcEgEBunyb6SWDC?usp=sharing)    

To reproduce results from T5 models and baseline results, run: code\t5\t5_evaluation.ipynb
To reproduce results from BERT models, run the three notebooks in the folder: code\bert\

To reproduce results from T5 models and baseline results, run: code\t5\t5_evaluation.ipynb
To reproduce results from BERT models, run the three notebooks in the folder: code\bert\

To reproduce results from T5 models and baseline results, run: code\t5\t5_evaluation.ipynb
To reproduce results from T5 models and baseline results, run: code\t5\t5_evaluation.ipynb   

To reproduce results from BERT models, run the three notebooks in the folder: code\bert\