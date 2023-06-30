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
  * **Hate/Offensive speech (and word rationales available):** This sentence has a 
  * **Hate/Offensive speech (and no word rationales available):**
  * **Normal Speech:**


## Training
