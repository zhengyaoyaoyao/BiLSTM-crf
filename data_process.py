import os.path
import random
from glob import glob
import pandas as pd


def get_annotation(ann_path):
    with open(ann_path,encoding='utf-8') as file:
        anns={}
        for line in file.readlines():#'T1\tDisease 1845 1850\t1型糖尿病
            arr = line.split('\t')[1].split()
            name=arr[0] # 实体名称
            start = int(arr[1])
            end = int(arr[-1])
            # 标注太长可能有问题
            if end-start>50:
                continue
            anns[start]='B-'+name
            for i in range(start+1,end):
                anns[i]='I-'+name
        return anns

def get_text(text_path):
    with open(text_path,encoding='utf-8') as file:
        return file.read()

# 建立文字和标签对应关系
def generate_annotation(config):
    for txt_path in glob(config.ORIGIN_DIR+'*.txt'):
        ann_path = txt_path[:-3]+'ann' # 得到标注的文件
        anns = get_annotation(ann_path)
        text = get_text(txt_path)
        #得到标注和句子，生成对应标注
        # 把一行的一句话拆开，为每个字先贴上O
        df = pd.DataFrame({'word':list(text),'label':['O']*len(text)})
        df.loc[anns.keys(),'label']=list(anns.values())
        # print(list(df.head(100)['label']))
        # print(list(df.head(100)['word']))
        # exit()
        #导出文件
        file_name = os.path.split(txt_path)[1]
        df.to_csv(config.ANNOTATION_DIR+file_name,header=None,index=None)

# 拆分训练集和测试集
def split_sample(config,test_size=0):
    files = glob(config.ANNOTATION_DIR+'*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    train_files = files[:n]
    test_files = files[n:]
    # 合并文件
    merge_file(train_files,config.TRAIN_SAMPLE_PATH)
    merge_file(test_files,config.TEST_SAMPLE_PATH)
# 生成词表
def generate_vocab(config):
    # 用csv读是因为文本有逗号隔开。
    df = pd.read_csv(config.TRAIN_SAMPLE_PATH,usecols=[0],names=['word'])
    vocab_list = [config.WORD_PAD,config.WORD_UNK] +df['word'].value_counts().keys().tolist()
    vocab_dict={v: k for k,v in enumerate(vocab_list)}
    # vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab = pd.DataFrame(vocab_dict.items())
    vocab.to_csv(config.VOCAB_SIZE,header=None,index=None)

def generate_label(config):
    df = pd.read_csv(config.TRAIN_SAMPLE_PATH,usecols=[1],names=['label'])
    label_list=df['label'].value_counts().keys().tolist()
    label_dict = {v:k for k,v in enumerate(label_list)}
    label = pd.DataFrame(label_dict.items())
    label.to_csv(config.LABEL_PATH,header=None,index=None)


def merge_file(files,target_path):
    with open(target_path,'a',encoding='utf-8-sig',errors='ignore') as file:
        for f in files:
            text = open(f,encoding='utf-8').read()
            file.write(text)



if __name__ == '__main__':
    from config import Config
    config = Config()

    # 建立文字和标签的对应关系
    generate_annotation(config)

    # 拆分训练预料和测试预料
    split_sample(config,3)
    # 生成词表
    generate_vocab(config)
    # 生成标签表
    generate_label(config)
