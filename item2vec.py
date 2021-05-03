import pandas as pd
import numpy as np
import os
import ast
from tqdm import tqdm
import time
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

class Item2Vec:
    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH
        self.min_count = 5
        self.size = 128
        self.window = 50
        self.sg = 1
        self.c2v_model = WordEmbeddingsKeyedVectors(self.size)

    def load_raw(self,class_name):
        print(f"Loading Raw...{class_name}")
        self.class_name = class_name
        raw_data = pd.read_csv("train.txt",sep="",encoding="cp949")
        raw_data = raw_data[raw_data["KSIC10_BZC_CD"].isnull() == False]
        raw_data["KEDCD"] = raw_data["KEDCD"].astype(int)
        raw_data["대분류"] = raw_data["KSIC10_BZC_CD"].apply(lambda x : x[0])
        raw_data["중분류"] = raw_data["KSIC10_BZC_CD"].apply(lambda x : x[:3])
        raw_data["소분류"] = raw_data["KSIC10_BZC_CD"].apply(lambda x : x[:4])
        raw_data["세분류"] = raw_data["KSIC10_BZC_CD"].apply(lambda x : x[:5])
        raw_data["세세분류"] = raw_data["KSIC10_BZC_CD"].apply(lambda x : x[:6])
        code = raw_data[["KEDCD",class_name]].drop_duplicates(["KEDCD",class_name], keep='first').sort_values("KEDCD").reset_index(drop=True)
        self.load_data()
        pre_train = pd.merge(self.pre_data,code,how="left",on="KEDCD")
        pre_train.apply(lambda x : x["Keyword"].append(x[class_name]),axis=1)


        self.train_data = pre_train


    def load_data(self):
        print("Data Loading...")
        train = pd.read_csv("train_stop_okt.csv")
        val = pd.read_csv("val_stop_okt.csv")
        tqdm.pandas()
        
        train["Keyword"] = train["Keyword"].progress_apply(lambda x : ast.literal_eval(x))
        val["Keyword"] = val["Keyword"].progress_apply(lambda x : ast.literal_eval(x))
        self.pre_data = train
        self.pre_val = val

    def get_dic(self, data):
        item_dic = {}
        total = []
        print("Get Dictionary...")
        for i,q in tqdm(data.iterrows(), total=data.shape[0]):
            item_dic[str(q['KEDCD'])] = q['Keyword']
            total.append(q["Keyword"])
     
        self.item_dic = item_dic
        self.total = total
        
        
    def get_i2v(self, total, min_count, size, window, sg,trained):
        if trained:
            print("Item2Vec is Already Trained")
            i2v_model = KeyedVectors.load_word2vec_format(f"item2vec_{self.size}dim_{self.class_name}")
        else:
            print(f"Item2Vec Training ...{size}dim")
            start = time.time()
            i2v_model = Word2Vec(sentences = total, min_count = min_count, size = size, window = window, sg = sg)
            i2v_model.train(total, total_examples=len(total), epochs=10)
            i2v_model.wv.save_word2vec_format(f"item2vec_{self.size}dim_{self.class_name}")
            end = time.time()
            print("Training Time is {} Seconds ...".format(round(end-start,2)))
        self.i2v_model = i2v_model
 
    def update_c2v(self, data, i2v_model):
        ID = []   
        vec = []
        tqdm.pandas()
        data["ID_Vector"] = data["Keyword"].progress_apply(lambda item : self.id_vector(item,self.size,self.i2v_model))
        for i,q in tqdm(data.iterrows(), total=data.shape[0]):
          self.c2v_model.add(entities=str(q["KEDCD"]), weights=q["ID_Vector"])
    @staticmethod
    def code_filter(x):
        if x in pred_code:
          return x
    @staticmethod
    def id_vector(items,size,model):
        temp = np.zeros(size)
        for item in items:
            try:
              temp += model.wv.get_vector(item)
            except:
              pass
            
        return temp
    


    def get_result(self, c2v_model, item_dic, val_df):
        answers = pd.DataFrame(columns = ["KEDCD","Keyword"])
        for n, q in tqdm(val_df.iterrows(), total = len(val_df),desc="First loop"):
              
              most_id = [x[0] for x in c2v_model.most_similar(q['KEDCD'], topn=50)]
              get_item = []
              for ID in tqdm(most_id,desc="Second loop"):
                  get_item += item_dic[ID]
                  
              get_item = list(pd.value_counts(get_item)[:50].index)
              get_item = filter(lambda x : self.code_filter(x), get_item)
              answers = answers.append(pd.DataFrame({
                  "KEDCD": q["KEDCD"],
                  "Keyword":get_item[:1]
              }))

        display(answers)
        answers.to_csv("Answer.csv", index=False)
    def model_train(self,size,class_name):
        self.load_raw(class_name)
        self.get_dic(self.pre_data)
        self.get_i2v(total=self.total, min_count=self.min_count, size=size, window=self.window, sg=self.sg,trained=False)
    
    def run(self):
        self.load_raw(class_name)
        self.get_dic(self.pre_data)
        self.get_i2v(total=self.total, min_count=self.min_count, size=self.size, window=self.window, sg=self.sg,trained=True)
        self.update_c2v(self.pre_data, self.i2v_model)
        self.get_result(self.c2v_model, self.item_dic, self.pre_val)
        
if __name__ == "__main__":
    I2V = Item2Vec(FILE_PATH)
    I2V.model_train(128,class_name)
    

