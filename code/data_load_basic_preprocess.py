"""
Code to load hateXplain dataset from huggingface, replace unknown sentences, and form a sentence from list of tokens. 
"""

from dataclasses import replace
import pandas as pd 
import numpy as np 
from datasets import load_dataset
import random 
import randomtimestamp 
import names 
import datetime 


# OUTPUT FILE LOCATIONS
train_processed_path = '../data/df_train_processed.csv'
val_processed_path = '../data/df_val_processed.csv'
test_processed_path = '../data/df_test_processed.csv'


class DataLoadPreprocess():
    # Load dataset 
    def __init__(self): 
        self.dataset = load_dataset("hatexplain")
        self.df_train = pd.DataFrame(self.dataset['train'])
        self.df_val =  pd.DataFrame(self.dataset['validation'])
        self.df_test =  pd.DataFrame(self.dataset['test'])

        # Train shape 
        print(self.df_train.shape)
        # Validation shape 
        print(self.df_val.shape)
        # Test shape 
        print(self.df_test.shape)

        # DF columns 
        print("Columns: \n", self.df_train.columns)
        print(self.df_train.head())

    # Function to check masked tokens in token list for a row  
    def get_masked_tokens_helper(self, token_list):
        masked_tokens = [] 
        for token in token_list:
            if '<' in token: 
                masked_tokens.append(token)
        return masked_tokens 

    # Check masked tokens 
    def get_masked_tokens(self):
        self.df_train["masked_tokens"] = self.df_train["post_tokens"].apply(lambda x: self.get_masked_tokens_helper(x))
        self.df_val["masked_tokens"] = self.df_val["post_tokens"].apply(lambda x: self.get_masked_tokens_helper(x))
        self.df_test["masked_tokens"] = self.df_test["post_tokens"].apply(lambda x: self.get_masked_tokens_helper(x))

    # Get unique masked tokens for a df[colname]
    def get_unique_masked_tokens_helper(self, df, colname='masked_tokens'):
        return set([a for b in df[colname].tolist() for a in b])

    # Get unique masked tokens from train, val and test. 
    def get_unique_masked_tokens(self):
        masked_token_set_train = self.get_unique_masked_tokens_helper(self.df_train)
        masked_token_set_val = self.get_unique_masked_tokens_helper(self.df_val)
        masked_token_set_test = self.get_unique_masked_tokens_helper(self.df_test)
        unique_masked_tokens = masked_token_set_train | masked_token_set_val | masked_token_set_test
        print("Unique masked tokens: ", unique_masked_tokens)
        # Unique masked tokens: 
        # {'<percent>', '<money>', '<number>', '<wink>', '<happy>', 
        # '<phone>', '<will>', '<time>', '<user>', '<kiss>', '<tong>',
        # '<date>', '<url>', '<email>', '<angel>', '<annoyed>', '<surprise>', 
        # '<laugh>', '<censored>', '<sad>'}
        
        return unique_masked_tokens 

    # Function to get replacements for masked tokens and convert to sentence   
    def unmask_tokens_create_sentence_helper(self, unique_masked_tokens, token_list):
        replacement_dict = dict()
        for token in token_list:
            if token in unique_masked_tokens:
                if token == '<percent>':
                    replace_with = str(random.uniform(1,100)) + "%"
                elif token == '<money>':
                    replace_with = str(random.uniform(1,10)) 
                elif token == '<number>':
                    replace_with = str(random.randint(1,10))
                elif token == '<wink>':
                    replace_with = 'wink_emoji'
                elif token == '<happy>':
                    replace_with = 'happy_emoji'
                elif token == '<phone>': # check where <phone> occurs 
                    replace_with = ''
                elif token == '<will>': # check where <will> occurs 
                    replace_with = '' 
                elif token == '<time>':
                    replace_with = randomtimestamp.random_time().strftime('%H:%M')
                elif token == '<user>':
                    replace_with = names.get_first_name() 
                elif token == '<kiss>':
                    replace_with = 'kiss_emoji'
                elif token == '<tong>':
                    replace_with = 'tongue_emoji'
                elif token == '<date>':
                    start_date = datetime.date(2018, 1, 1)
                    end_date = datetime.date.today()
                    replace_with = randomtimestamp.random_date(start=start_date, end = end_date).strftime("%d-%m-%Y")
                elif token == '<url>':  # check where URL occurs 
                    replace_with = ''
                elif token == '<email>':
                    domain = ['@gmail.com', '@yahoo.com', '@hotmail.com', '@rediffmail.com']
                    replace_with = names.get_first_name() + random.choice(domain)
                elif token == '<angel>': # check where angel occurs 
                    replace_with = 'angel_emoji' 
                elif token == '<annoyed>':
                    replace_with = 'annoyed_emoji'
                elif token == '<surprise>':
                    replace_with = 'surprise_emoji'
                elif token == '<laugh>':
                    replace_with = 'laugh_emoji'
                elif token == '<censored>': # check where censored occurs 
                    replace_with = '<censored>'
                elif token == '<sad>':
                    replace_with = 'sad_emoji'
                else: # no other masked tokens, just for completeness 
                    replace_with = ''
                replacement_dict[token] = replace_with
        # Unmask tokens and return sentence 
        unmasked_token_list = [x if x not in replacement_dict else replacement_dict[x] for x in token_list]
        return " ".join(unmasked_token_list)


    # Function to get unmasked sentences for train, val and test 
    def get_unmasked_sentence(self, unique_masked_tokens):
        self.df_train['unmasked_sentence'] = self.df_train["post_tokens"].apply(lambda x: self.unmask_tokens_create_sentence_helper(unique_masked_tokens, x))
        self.df_val['unmasked_sentence'] = self.df_val["post_tokens"].apply(lambda x: self.unmask_tokens_create_sentence_helper(unique_masked_tokens, x))
        self.df_test['unmasked_sentence'] = self.df_test["post_tokens"].apply(lambda x: self.unmask_tokens_create_sentence_helper(unique_masked_tokens, x))


    # Function to get masked sentence for train, val and test 
    def get_masked_sentence(self):
        self.df_train["masked_sentence"] = self.df_train["post_tokens"].apply(lambda x: " ".join(x))
        self.df_val["masked_sentence"] = self.df_val["post_tokens"].apply(lambda x: " ".join(x))
        self.df_test["masked_sentence"] = self.df_test["post_tokens"].apply(lambda x: " ".join(x))

    # Save Data 
    def save_data(self):
        self.df_train.to_csv(train_processed_path, index=False)
        self.df_val.to_csv(val_processed_path, index=False)
        self.df_test.to_csv(test_processed_path, index=False)

    # Run all 
    def run(self):
        self.get_masked_tokens()
        unique_masked_tokens = self.get_unique_masked_tokens() 
        self.get_unmasked_sentence(unique_masked_tokens)
        self.get_masked_sentence()
        self.save_data()

if __name__ == "__main__":
    dlp = DataLoadPreprocess()
    dlp.run()
    
