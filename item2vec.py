import pandas as pd
import numpy as np
import json
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
        self.negative = 5
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
        pre_train = pre_train[["KEDCD","Keyword"]]
        self.class_name = class_name

        self.train_data = pre_train
        self.code = set(code[class_name])
   

    def load_data(self):
        print("Data Loading...")
        train = pd.read_csv("train_stop_okt.csv")
        val = pd.read_csv("val_stop_okt.csv")
        tqdm.pandas()
        
        train["Keyword"] = train["Keyword"].progress_apply(lambda x : ast.literal_eval(x))
        val["Keyword"] = val["Keyword"].progress_apply(lambda x : ast.literal_eval(x))
        self.pre_data = train
        self.pre_val = val

    def get_train_dic(self, data):
        item_dic = {}
        total = []
        print("Get Train Dictionary...")
        for i,q in tqdm(data.iterrows(), total=data.shape[0]):
            item_dic[str(q['KEDCD'])] = q['Keyword']
            total.append(q["Keyword"])
     
        self.item_dic = item_dic
        self.total = total


    def get_dic(self, train,val):
        item_dic = {}
        total = []
        data = pd.concat([train,val])
        print("Get Dictionary...")
        for i,q in tqdm(data.iterrows(), total=data.shape[0]):
            item_dic[str(q['KEDCD'])] = q['Keyword']
            total.append(q["Keyword"])

        self.item_dic = item_dic
        self.total = total
        
    def get_i2v(self, total, min_count, size, window, sg, negative, class_name, trained):
        if trained:
            print("Item2Vec is Already Trained")
            i2v_model = KeyedVectors.load_word2vec_format(f"item2vec_{size}dim_{class_name}_negative{negative}")
        else:
            print(f"Item2Vec Training ...{size}dim")
            start = time.time()
            i2v_model = Word2Vec(sentences = total, min_count = min_count, size = size, window = window, sg = sg, negative = negative)
            i2v_model.train(total, total_examples=len(total), epochs=10)
            i2v_model.wv.save_word2vec_format(f"item2vec_{size}dim_{class_name}_negative{negative}")
            end = time.time()
            print("Training Time is {} Seconds ...".format(round(end-start,2)))
        self.i2v_model = i2v_model
 
    def update_c2v(self, train,val, i2v_model):
        ID = []   
        vec = []
        data = pd.concat([train,val])
        tqdm.pandas()
        print("Get ID Vector ...")
        data["ID_Vector"] = data["Keyword"].progress_apply(lambda item : self.id_vector(item,self.size,self.i2v_model))
        print("Update ID Vector ...")
        if os.path.isfile(f"vec_{self.size}dim_{self.class_name}_negative{self.negative}") == False:
          for i,q in tqdm(data.iterrows(), total=data.shape[0]):
            self.c2v_model.add(entities=str(q["KEDCD"]), weights=q["ID_Vector"])
          
        else:
            self.c2v_model = WordEmbeddingsKeyedVectors().load_word2vec_format(f"vec_{self.size}dim_{self.class_name}_negative{self.negative}")

    @staticmethod
    def id_vector(items,size,model):
        temp = np.zeros(size)
        for item in items:
            try:
              temp += model.wv.get_vector(item)
            except:
              pass
            
        return temp
    


    def get_result(self, c2v_model,i2v_model, item_dic, val_df, topk, size, class_name):
        for n, q in tqdm(val_df.iterrows(), total = len(val_df),desc="First loop"):
              ans_dic = {}
              most_id = [x[0] for x in c2v_model.most_similar(q['KEDCD'], topn=300) if x[0] in self.item_dic.keys()][:topk]
              get_item = []
              for ID in tqdm(most_id,desc="Second loop"):
                  get_item += item_dic[ID]
              id_v = id_vector(get_item,size,i2v_model)
              result = i2v_model.most_similar(positive=[id_v], topn=50)
              result = [r[0] for r in result if r[0] in self.code]
              ans_dic[q["KEDCD"]] = result[:5]

        print(ans_dic)
        with open(f"{size}dim_{class_name}_val.json", "w") as json_file:
            json.dump(ans_dic, json_file)


    def model_train(self,size,class_name):
        self.load_raw(class_name)
        self.get_train_dic(self.pre_data)
        self.get_i2v(total=self.total, min_count=self.min_count, size=size, window=self.window, sg=self.sg,negative=self.negative,class_name=class_name,trained=False)
    
    def run(self,size,class_name,topk):
        self.load_raw(class_name)
        self.get_dic(self.train_data,self.pre_val)
        self.get_i2v(total=self.total, min_count=self.min_count, size=size,class_name=class_name, window=self.window, sg=self.sg,trained=True)
        self.update_c2v(self.train_data,self.pre_val, self.i2v_model)
        self.get_result(self.c2v_model, self.i2v_model, self.item_dic, self.pre_val, topk, size, class_name)
if __name__ == "__main__":
    I2V = Item2Vec(FILE_PATH)
    I2V.model_train(128,class_name)
    

