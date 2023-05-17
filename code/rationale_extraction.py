"""
Code to extract rationales from Open Assistant (Huggingface Chat) for each row. 
Prompt for HuggingChat -> Is the following statement hate speech, why or why not? 
API: https://github.com/Soulter/hugging-chat-api 

TO DO: Can change the hyperparameters for chatbot: temperature, etc. 
"""

import pandas as pd 
import numpy as np 
from hugchat import hugchat
import pickle 

## INPUT FILES 
train_path = '../data/load_and_preprocess/df_train_processed.csv'
val_path = '../data/load_and_preprocess/df_val_processed.csv'
test_path = '../data/load_and_preprocess/df_test_processed.csv'

## OUTPUT FILES 
train_rationale_path = '../data/rationale_extraction/df_train_with_rationales.csv'
val_rationale_path = '../data/rationale_extraction/df_val_with_rationales.csv'
test_rationale_path = '../data/rationale_extraction/df_test_with_rationales.csv'
texts_not_processed_path = '../data/texts_not_processed.pkl'

class ExtractRationales():
    def __init__(self):
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        self.chatbot = hugchat.ChatBot()

    ## Helper function to get Hugchat Responses
    def get_hugchat_response_helper(self, text):
        self.not_processed = [] 
        prompt_start = "Is the following statement hate speech? Why or why not? - "
        prompt = prompt_start + text 
        try:
            response = self.chatbot.chat(prompt)
        except:
            response = "" 
            self.not_processed.append(text)
        return response 

    # Function to split df into 10 chunks - easier to extract and write on disk. 
    def chunk_data(self, df):
        if df.shape[0]>=10000:
            return np.array_split(df, 1000)
        else:
            return np.array_split(df, 100)

    # Function to save generated rationales for each df chunk. 
    def save_rationales(self, df, path, mode='a'):
        df['hugchat_masked_sentence_response'] = df['masked_sentence'].apply(lambda x: self.get_hugchat_response_helper(x))
        df.to_csv(path, index=False, mode=mode)

    # Function to loop over df_train_list or df_val_list or df_test_list
    def loop_over_chunk(self, df_list, path):
        for i in range(len(df_list)):
            print("Chunk: ", i)
            if i==0:
                self.save_rationales(df_list[i], path, 'w')
            else:
                self.save_rationales(df_list[i], path)

    ## Function to get HugChat Responses for train, val and test datasets. 
    def get_hugchat_response(self):
        # Chunk data and create a list of dfs 
        df_train_list = self.chunk_data(self.df_train)
        df_val_list = self.chunk_data(self.df_val)
        df_test_list = self.chunk_data(self.df_test)
        
        # Get Rationales for all dataframes 
        print("Training Data: \n")
        self.loop_over_chunk(df_train_list, train_rationale_path)
        print("Validation Data: \n")
        self.loop_over_chunk(df_val_list, val_rationale_path)
        print("Test Data: \n")
        self.loop_over_chunk(df_test_list, test_rationale_path)
        
        # Save texts that were not processed 
        with open(texts_not_processed_path, 'wb') as f:
            pickle.dump(self.not_processed, f)
        

if __name__ == "__main__":
    er = ExtractRationales()
    er.get_hugchat_response()