"""
Post-Processing of Generated output from Hugchat 
1. Separate label, community, rationale and keywords 
2. Need to check if we should mask offensive/normal/hate in rationales.
"""

import pandas as pd
import numpy as np
import difflib
import string
from nltk.corpus import stopwords


#### INPUT 
train_rationale_path = '../data/rationale_extraction/df_train_with_rationales.csv'
val_rationale_path = '../data/rationale_extraction/df_val_with_rationales.csv'
test_rationale_path = '../data/rationale_extraction/df_test_with_rationales.csv'
columns = ['id', 'annotators', 'rationales', 'post_tokens', 'masked_tokens', 'unmasked_sentence', 'masked_sentence', 'hugchat_response']

### Output 
train_rationale_post_processed_path = '../data/rationale_extraction/df_train_rationale_post_processed.csv'
val_rationale_post_processed_path = '../data/rationale_extraction/df_val_rationale_post_processed.csv'
test_rationale_post_processed_path = '../data/rationale_extraction/df_test_rationale_post_processed.csv'


class RationalePostProcess():
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df.columns = columns
        print("Input shape: ", self.df.shape)
        self.df.dropna(subset=['hugchat_response'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("Shape after dropping NAs: ", self.df.shape)


    # Function to extract classification label from hugchat responses
    # Split the text response by '\n'. Split sentences by ':'. Check for current phrase in the list to be similar to 'classification'. 
    # Return the next phrase
    def extract_label(self, hugchat_response):
        text = hugchat_response.lower()
        text_list = text.split('\n')
        for sentence in text_list:
            phrases = sentence.split(":")
            for i in range(len(phrases)):
                if phrases[i].strip() in ['classification']:
                    if i+1<len(phrases):
                        return phrases[i+1]
                elif phrases[i].strip() in ['output part1', 'output part 1', 'output 1', 'output part i', 'part1', 'first part']:
                    # check next word is not 'classification'
                    if i+1<len(phrases):
                        s = difflib.SequenceMatcher(None, "classification", phrases[i+1].strip())
                        if s.ratio() >= 0.80 and (i+2)<len(phrases):
                            return phrases[i+2]
                        else:
                            return phrases[i+1]
                else:
                    s = difflib.SequenceMatcher(None, "classification", phrases[i].strip())
                    if (s.ratio()>=0.70) and (i+1)<len(phrases):
                        return phrases[i+1]
                    else:
                        s = difflib.SequenceMatcher(None, "output start", phrases[i].strip())
                        t = difflib.SequenceMatcher(None, "start", phrases[i].strip())
                        if (s.ratio()>0.9) and (i+1)<len(phrases):
                            if phrases[i+1].strip() in ['normal', 'normal speech', 'hateful', 'hateful speech', 'hate speech', 'offensive', 'offensive speech']:
                                return phrases[i+1]
                        if (t.ratio()>0.9) and (i+1)<len(phrases):
                            if phrases[i+1].strip() in ['normal', 'normal speech', 'hateful', 'hateful speech', 'hate speech', 'offensive', 'offensive speech']:
                                return phrases[i+1]
                
        return "-1"

    

    # Function to extract targeted communities from hugchat responses
    # Split the text response by '\n'. Split sentences by ':'. Check for current phrase in the list to be similar to 'communities targeted'. 
    # Return the next phrase
    def extract_comm_targeted(self, hugchat_response):
        text = hugchat_response.lower()
        text_list = text.split('\n')
        for sentence in text_list:
            phrases = sentence.split(":")
            for i in range(len(phrases)):
                if phrases[i].strip() in ['communities targeted', 'community targeted', 'targeted community', 'targeted communities', 'targeted Community(ies)']:
                    if i+1<len(phrases):
                        return phrases[i+1]
                elif phrases[i].strip() in ['output part2', 'output part 2', 'output 2', 'output part ii', 'part2', 'second part']:
                    # check next word is not communities targeted
                    if i+1<len(phrases):
                        s = difflib.SequenceMatcher(None, "communities targeted", phrases[i+1].strip())
                        if s.ratio() >= 0.60 and (i+2)<len(phrases):
                            return phrases[i+2]
                        else:
                            return phrases[i+1]
                else:
                    s = difflib.SequenceMatcher(None, "communities targeted", phrases[i].strip())
                    t = difflib.SequenceMatcher(None, "targeted communities", phrases[i].strip())
                    u = difflib.SequenceMatcher(None, "community targeted", phrases[i].strip())
                    v = difflib.SequenceMatcher(None, "targeted community", phrases[i].strip())
                    if (max(s.ratio(),t.ratio(), u.ratio(), v.ratio())>=0.60) and (i+1)<len(phrases):
                        return phrases[i+1]
        return "-1"

    
    # Function to extract targeted communities from hugchat responses
    # Split the text response by '\n'. Split sentences by ':'. Check for current phrase in the list to be similar to 'explanation'. 
    # Return the next phrase
    def extract_rationale(self, hugchat_response):
        text = hugchat_response.lower()
        text_list = text.split('\n')
        #print("Text list: \n", text_list)
        for sentence in text_list:
            #print("Sentence: ", sentence)
            phrases = sentence.split(":")
            for i in range(len(phrases)):
                #print("current phrase: ", phrases[i])
                if phrases[i].strip() in ['explanation', 'justification', '* explanation', '### explanation ###', 'explancement', 'explaining my answer']:
                    if i+1<len(phrases):
                        return phrases[i+1]
                elif phrases[i].strip() in ['output part3', 'output part 3', 'output 3', 'output part iii', 'part3', 'third part']:
                    # check next word is not explanation
                    if i+1<len(phrases):
                        s = difflib.SequenceMatcher(None, "explanation", phrases[i+1].strip())
                        if s.ratio() >= 0.80 and (i+2)<len(phrases):
                            return phrases[i+2]
                        else:
                            return phrases[i+1]
                elif 'explanation' in phrases[i].strip():
                    if (i+1)<len(phrases):
                        if len(phrases[i+1].split()) > 20:
                            return phrases[i+1]
                else:
                    s = difflib.SequenceMatcher(None, "explanation", phrases[i].strip())
                    t = difflib.SequenceMatcher(None, "justification", phrases[i].strip())
                    if (max(s.ratio(),t.ratio())>=0.80) and (i+1)<len(phrases):
                        return phrases[i+1]
        return "-1"

    # Function to extract keywords from hugchat responses
    # Split the text response by '\n'. Split sentences by ':'. Check for current phrase in the list to be similar to 'keywords'. 
    # Return the next phrase
    def extract_keywords(self, hugchat_response):
        text = hugchat_response.lower()
        text_list = text.split('\n')
        #print("Text list: \n", text_list)
        for sentence in text_list:
            #print("Sentence: ", sentence)
            phrases = sentence.split(":")
            for i in range(len(phrases)):
                #print("current phrase: ", phrases[i])
                if phrases[i].strip() in ['keywords', 'key word', 'keywords', 'key words']:
                    if i+1<len(phrases):
                        return phrases[i+1]
                elif phrases[i].strip() in ['output part4', 'output part 4', 'output 4', 'output part iv', 'part4', 'part 4', 'fourth part']:
                    # check next word is not 'keywords'
                    if i+1<len(phrases):
                        s = difflib.SequenceMatcher(None, "keywords", phrases[i+1].strip())
                        if s.ratio() >= 0.80 and (i+2)<len(phrases):
                            return phrases[i+2]
                        else:
                            return phrases[i+1]
                elif 'key' in phrases[i].strip():
                    if (i+1)<len(phrases):
                        if len(phrases[i+1].split()) < 20:
                            return phrases[i+1]
                else:
                    s = difflib.SequenceMatcher(None, "keyword", phrases[i].strip())
                    if (s.ratio()>=0.80) and (i+1)<len(phrases):
                        return phrases[i+1]
        return "-1"

    # Function to remove all entries which have "-1" in any of the field 
    def drop_no_output_rows(self,df):
        print("Intital shape: ", df.shape)
        df_copy = df.copy()
        print("label -1 rows : ", df_copy[df_copy['hugchat_label']=="-1"].shape[0])
        df_copy = df_copy[df_copy['hugchat_label']!="-1"].copy()
        print("comm targeted -1 : ", df_copy[df_copy['hugchat_comm_targeted']=="-1"].shape[0])
        df_copy = df_copy[df_copy['hugchat_comm_targeted']!="-1"].copy()
        print("keywords -1 : ", df_copy[df_copy['hugchat_keywords']=="-1"].shape[0])
        df_copy = df_copy[df_copy['hugchat_keywords']!="-1"].copy()
        print("explanation -1 : ", df_copy[df_copy['hugchat_explanation']=="-1"].shape[0])
        df_copy = df_copy[df_copy['hugchat_explanation']!="-1"].copy()
        print("Final shape: ",df_copy.shape)
        return df_copy

    # Function to convert hugchat labels into "hate", "offensive", "normal". 
    # Some responses have hate/offensive. Labels are overindexed on hate speech, prioritizing offensive over hate and normal.
    # Priority Order: hate < normal < offensive
    def convert_hugchat_label(self,hugchat_label):
        label = ''
        hugchat_label = hugchat_label.strip()
        if "hate" in hugchat_label:
            label = "hate_speech"
        if "normal" in hugchat_label:
            label = "normal_speech"
        if "offensive" in hugchat_label:
            label = "offensive_speech"
        return label 


    # Function to convert hugchat comm targeted into ['African', 'Islam', 'Jewish', 'Homosexual', 'Women', 'Refugee', 'Arab', 'Caucasian', 'Asian', 'Hispanic', 'None'] 
    def convert_comm_targeted(self, hugchat_comm_targeted):
        hugchat_comm_targeted = hugchat_comm_targeted.strip()
        hugchat_comm_targeted = hugchat_comm_targeted.translate(str.maketrans('', '', string.punctuation))
        comm_target = []
        none_list = ['none', 'na', 'not applicable', 'not specified']
        islam_list = ['islam', 'muslim', 'moslem', 'muslims', 'moslems']
        homosexual_list = ['homo','gay','lgbtq', 'lesbian','queer']
        women_list = ['women', 'female', 'feminist']
        refugee_list = ['refugee', 'migrant', 'immigrant']
        african_list = ['african','black']
        asian_list = ['asian', 'india', 'pakistan', 'bangla']
        hispanic_list = ['latin','hispanic']
        
        for item in none_list:
            if item in hugchat_comm_targeted:
                comm_target.append('none')
        
        for item in african_list:
            if item in hugchat_comm_targeted:
                comm_target.append('african')
        
        for item in islam_list:
            if item in hugchat_comm_targeted:
                comm_target.append('islam')
        
        if 'jew' in hugchat_comm_targeted:
            comm_target.append('jewish')
        
        if 'arab' in hugchat_comm_targeted:
            comm_target.append('arab')
        
        if 'white' in hugchat_comm_targeted:
            comm_target.append('caucasian')
        
        for item in asian_list:
            if item in hugchat_comm_targeted:
                comm_target.append('asian')
        
        for item in hispanic_list:
            if item in hugchat_comm_targeted:
                comm_target.append('hispanic')
        
        for item in women_list:
            if item in hugchat_comm_targeted:
                comm_target.append('women')
        
        for item in homosexual_list:
            if item in hugchat_comm_targeted:
                comm_target.append('homosexual')
        
        for item in refugee_list:
            if item in hugchat_comm_targeted:
                comm_target.append('refugee')
        
        if ('none' in comm_target) and (len(comm_target)>1):
            comm_target = [x for x in comm_target if x!='none']
        
        return list(set(comm_target))
    
    # Function to check to return hugchat keywords present in the sentence. 
    def convert_hugchat_keywords(self, sentence, hugchat_keywords):
        kw_list = []
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        words = sentence.split()
        words = [word for word in words if word not in stopwords.words('english')]
        kwords = hugchat_keywords.split()
        for kw in kwords:
            kw = kw.strip()
            for word in words:
                word = word.strip()
                s = difflib.SequenceMatcher(None, word, kw)
                if s.ratio()>=0.7:
                    kw_list.append(word)
        return list(set(kw_list))

    # Run function
    def run(self):
        # Get outputs from Hugchat Response
        self.df['hugchat_label'] = self.df['hugchat_response'].apply(lambda x: self.extract_label(x))
        self.df['hugchat_comm_targeted'] = self.df['hugchat_response'].apply(lambda x: self.extract_comm_targeted(x))
        self.df['hugchat_explanation'] = self.df['hugchat_response'].apply(lambda x: self.extract_rationale(x))
        self.df['hugchat_keywords'] = self.df['hugchat_response'].apply(lambda x: self.extract_keywords(x))

        ## Process these outputs
        df_filtered = self.drop_no_output_rows(self.df)
        df_filtered['hugchat_label_processed'] = df_filtered['hugchat_label'].apply(lambda x: self.convert_hugchat_label(x))
        df_filtered['hugchat_comm_targeted_processed'] = df_filtered['hugchat_comm_targeted'].apply(lambda x: self.convert_comm_targeted(x))
        df_filtered['hugchat_keywords_processed'] = df_filtered[['unmasked_sentence','hugchat_keywords']].apply(lambda row: self.convert_hugchat_keywords(row['unmasked_sentence'], row['hugchat_keywords']), axis=1)

        print("There will be some rows without an entry. You can use them or discard at modeling time.")
        print("Final Shape: ", df_filtered.shape)

        return df_filtered


if __name__ == "__main__":
    print("Train dataset")
    rat_pp_train = RationalePostProcess(train_rationale_path)
    df_filtered_train = rat_pp_train.run()
    df_filtered_train.to_csv(train_rationale_post_processed_path, index=False)
    print("Val dataset")
    rat_pp_val = RationalePostProcess(val_rationale_path)
    df_filtered_val = rat_pp_val.run()
    df_filtered_val.to_csv(val_rationale_post_processed_path, index=False)
    print("Test dataset")
    rat_pp_test = RationalePostProcess(test_rationale_path)
    df_filtered_test = rat_pp_test.run()
    df_filtered_test.to_csv(test_rationale_post_processed_path, index=False)
