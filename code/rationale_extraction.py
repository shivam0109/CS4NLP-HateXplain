"""
Code to extract rationales from Open Assistant (Huggingface Chat) for each row.  
API: https://github.com/Soulter/hugging-chat-api 

TO DO: Can change the hyperparameters for chatbot: temperature, etc. 
"""

from email import header
import pandas as pd 
import numpy as np 
from hugchat import hugchat
import pickle 
from hugchat.login import Login

# login
email = "shivam.adarsh@uzh.ch"
passwd = "" # Enter your password for Huggingface here
sign = Login(email, passwd)
#cookies = sign.login()
cookies = sign.loadCookies() # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.
sign.saveCookies()


## INPUT FILES 
train_path = '../data/load_and_preprocess/df_train_processed.csv'
val_path = '../data/load_and_preprocess/df_val_processed.csv'
test_path = '../data/load_and_preprocess/df_test_processed.csv'

## OUTPUT FILES 
train_rationale_path = '../data/rationale_extraction/df_train_with_rationales.csv'
val_rationale_path = '../data/rationale_extraction/df_val_with_rationales.csv'
test_rationale_path = '../data/rationale_extraction/df_test_with_rationales.csv'
texts_not_processed_path = '../data/texts_not_processed.pkl'

# Prompt 
prompt_start = """
Your task is to classify a sentence into hate speech, offensive speech or normal speech, and explain your answer. In addition, you would need to provide the communities being targeted and the keywords for your answer present in the sentence.

Produce the result in four parts - your first part should give the classification. 
The second part should give the community being targeted from the following list: ['African', 'Islam', 'Jewish', 'Homosexual', 'Women', 'Refugee', 'Arab', 'Caucasian', 'Asian', 'Hispanic', 'Other community']. You can select multiple communities if it's relevant. 
The third part should give an explanation of your answer. 
The fourth part should extract the key words only from the sentence due to which the sentence might be offensive or hate-speech. Don't include words not present in the sentence.

Here is an example: 
Input Sentence: Muslims should be banned from praying on the streets 

Output Start : 
Classification: Offensive
Community targeted: Islam 
Explanation: Using the word "ban" is authoritarian, and hurts the sentiments of people following that religion. The phrasing lacks justification or evidence for why this particular group should face restrictions and creates tension between majority cultural values and religious freedoms enjoyed by all citizens. Therefore, this statement reflects a narrow perspective about diversity and social cohesion within our societies.
Keywords: banned, prayers
Output End

Now, try to produce the output in a similar format for the following sentence: 
Sentence: """


class ExtractRationales():
    def __init__(self):
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        self.prompt_start = prompt_start
        self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

    ## Helper function to get Hugchat Responses
    def get_hugchat_response_helper(self, text):
        self.not_processed = [] 
        prompt = self.prompt_start + text 
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
        df['hugchat_response'] = df['unmasked_sentence'].apply(lambda x: self.get_hugchat_response_helper(x))
        if mode=='a':
            df.to_csv(path, index=False, mode=mode, header=False)
        else:
            df.to_csv(path, index=False, mode=mode)

    # Function to loop over df_train_list or df_val_list or df_test_list
    def loop_over_chunk(self, df_list, path):
        for i in range(len(df_list)):
            print("Chunk: ", i)
            if i==0:
                self.save_rationales(df_list[i], path, 'w')
            else:
                self.save_rationales(df_list[i], path, 'a')

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