import pandas as pd
import numpy as np
import json
import os
import ast
from tqdm import tqdm
import time
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim import utils
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim.models import KeyedVectors
from gensim import utils
import warnings
warnings.filterwarnings("ignore")

class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1


class Item2Vec:
    def __init__(self, size, FILE_PATH):
        self.FILE_PATH = FILE_PATH
        self.min_count = 5
        self.size = size
        self.window = 100000
        self.sg = 1
        self.negative = 15
        
        self.loss_logger = LossLogger()

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
        tst = pd.read_csv("test_stop_okt.csv")
        tqdm.pandas()
        
        train["Keyword"] = train["Keyword"].progress_apply(lambda x : ast.literal_eval(x))
        val["Keyword"] = val["Keyword"].progress_apply(lambda x : ast.literal_eval(x))
        tst["Keyword"] = tst["Keyword"].progress_apply(lambda x : ast.literal_eval(x))
        self.pre_data = train
        self.pre_val = val
        self.pre_tst = tst

    def get_train_dic(self, data):
        item_dic = {}
        total = []
        print("Get Train Dictionary...")
        for i,q in tqdm(data.iterrows(), total=data.shape[0]):
            item_dic[str(q['KEDCD'])] = q['Keyword']
            total.append(q["Keyword"])
     
        self.item_dic = item_dic
        self.total = total


    def get_dic(self, train,val,tst):
        item_dic = {}
        total = []
        data = pd.concat([train,val,tst])
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
            i2v_model = Word2Vec(sentences = total, min_count = min_count, size = size, window = window, sg = sg, negative = negative, callbacks = [self.loss_logger], compute_loss = True)
            i2v_model.train(total, total_examples=len(total), epochs=5)
            i2v_model.wv.save_word2vec_format(f"item2vec_{size}dim_{class_name}_negative{negative}")
            end = time.time()
            print("Training Time is {} Seconds ...".format(round(end-start,2)))
        self.i2v_model = i2v_model
 
    def get_c2v(self, train,val,tst, i2v_model,size,trained):
        ID = []   
        vec = []
        data = pd.concat([train,val,tst])
        self.c2v_model = Word2VecKeyedVectors(vector_size=size)
        
        if trained == False:
            print("Get ID Vector ...")
            data["ID_Vector"] = data["Keyword"].progress_apply(lambda item : self.id_vector(item,self.size,self.i2v_model))
            print("Update ID Vector ...")
            ked_dict = {}
            for i,q in tqdm(data.iterrows(), total=data.shape[0]):
                ked_dict[str(q["KEDCD"])] = q["ID_Vector"]
        
            self.c2v_model.vocab = ked_dict
            self.c2v_model.vectors = np.array(list(ked_dict.values()))
            self.my_save_word2vec_format(binary=True, fname=f"vec_{self.size}dim_{self.class_name}_negative{self.negative}.bin", total_vec=len(ked_dict), vocab=self.c2v_model.vocab, vectors=self.c2v_model.vectors)

        else:
            print("Cor2Vec is Already Trained")
            self.c2v_model = Word2VecKeyedVectors(vector_size=size).load_word2vec_format(f"vec_{self.size}dim_{self.class_name}_negative{self.negative}.bin",binary=True)

    @staticmethod
    def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        vocab : dict
            The vocabulary of words.
        vectors : numpy.array
            The vectors to be stored.
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
            Explicitly specify total number of vectors
            (in case word vectors are appended with document vectors afterwards).

        """
        if not (vocab or vectors):
            raise RuntimeError("no input")
        if total_vec is None:
            total_vec = len(vocab)
        vector_size = vectors.shape[1]
        assert (len(vocab), vector_size) == vectors.shape
        with utils.smart_open(fname, 'wb') as fout:
            print(total_vec, vector_size)
            fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            # store in sorted order: most frequent words at the top
            for word, row in vocab.items():
                if binary:
                    row = row.astype(REAL)
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

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
        print("Get Result...") 
        ans_dic = {}
        for n, q in tqdm(val_df.iterrows(), total = len(val_df),desc="Test loop"):
              similar_id = c2v_model.most_similar(str(q['KEDCD']), topn=topk)
              most_id = [x[0] for x in similar_id]
              
              results = []
              for ID in most_id:
                id_v = self.c2v_model[ID]
                result = i2v_model.most_similar(positive=[id_v], topn=1500)
                result = [r[0] for r in result if r[0] in self.code][:5]
                results.append(result)
              results = pd.DataFrame(results).mode()
              results = results.values.tolist()
              ans_dic[q["KEDCD"]] = results

        
        with open(f"{size}dim_{class_name}_val_{topk}nn1.json", "w") as json_file:
            json.dump(ans_dic, json_file)


    def i2v_model_train(self,class_name):
        self.load_raw(class_name)
        self.get_train_dic(self.pre_data)
        self.get_i2v(total=self.total, min_count=self.min_count, size=self.size, window=self.window, sg=self.sg,negative=self.negative,class_name=class_name,trained=False)
    
    def c2v_model_train(self,class_name,topk):
        self.load_raw(class_name)
        self.get_dic(self.train_data,self.pre_val,self.pre_tst)
        self.get_i2v(total=self.total, min_count=self.min_count, size=self.size,class_name=class_name, negative=self.negative,window=self.window, sg=self.sg,trained=True)
        self.get_c2v(self.train_data,self.pre_val,self.pre_tst, self.i2v_model,size,trained=False)
    
    def run(self,class_name,topk):
        self.load_raw(class_name)
        self.get_dic(self.train_data,self.pre_val,self.pre_tst)
        self.get_i2v(total=self.total, min_count=self.min_count, size=self.size,class_name=class_name, negative=self.negative,window=self.window, sg=self.sg,trained=True)
        self.get_c2v(self.train_data,self.pre_val,self.pre_tst, self.i2v_model,self.size,trained=True)
        self.get_result(c2v_model=self.c2v_model, i2v_model=self.i2v_model, item_dic=self.item_dic, val_df=self.pre_val, topk=topk, size=self.size, class_name=class_name)

