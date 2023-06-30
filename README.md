# CS4NLP-HateXplain
Project Repo of Computational Semantics for NLP - Hateful Speech Detection with explanations

# Current workflow
This table summarizes the current worflow followed for hate-speech detection: 

## Preprocessing 
* A sample consists of a list of words and multiple annotations whether the given the sentence is hate speech
* Such an annotation can either be 0 for Hate Speech, 1 for normal and 2 for Hate Speech
* Given these annotations we calculate an avg_score as the mean of all annotations


## Training
