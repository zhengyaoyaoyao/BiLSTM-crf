import torch

from utils import  *
from config import  *


if __name__ == '__main__':
    config = Config()
    text = "每个糖尿病患者,无论是病情轻重,不论是注射胰岛素,还是口服降糖药,都必须合理地控制饮食。"
    _,word2id = get_vacob(config)
    # input = torch.Tensor([[word2id.get(w,config.WORD_UNK_ID) for w in text]])
    input = torch.tensor([[word2id.get(w, config.WORD_UNK_ID) for w in text]])
    mask = torch.tensor([[1]*len(text)]).bool()
    model = torch.load(config.MODEL_DIR+'(cuda)model_0.pth')
    input = input.to(device=config.device)
    mask = mask.to(device = config.device)
    y_pred = model(input,mask)
    id2lable = get_label(config)
    label = [id2lable[0][l] for l in y_pred[0]] # 标签的序列
    print(text)
    print(label)
    def extract(label,text):
        i = 0
        res = []
        while i<len(label):
            if label[i] !='O':
                prefix,name = str(label[i]).split('-')
                start = end = i
                i+=1
                while i<len(label) and label[i] =='I-'+name:
                    end = i
                    i+=1
                res.append([name,text[start:end+1]])
            else:
                i+=1
        return res
    info = extract(label,text)
    print(info)

