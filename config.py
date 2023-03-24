import torch.cuda


class Config:
    def __init__(self):
        self.ORIGIN_DIR='./input/origin/'
        self.ANNOTATION_DIR='./output/annotation/'

        self.TRAIN_SAMPLE_PATH= 'output/originAuthor/train_sample.txt'
        self.TEST_SAMPLE_PATH= 'output/originAuthor/test_sample.txt'
        self.VOCAB_PATH= 'output/originAuthor/vocab.txt'
        self.LABEL_PATH= 'output/originAuthor/label.txt'

        self.WORD_PAD='<PAD>'
        self.WORD_UNK='<UNK>'

        self.WORD_PAD_ID=0
        self.WORD_UNK_ID=1
        self.LABEL_O_ID=0

        self.VOCAB_SIZE=3000
        self.EMBEDDING_DIM=100
        self.HIDDEN_SIZE=256
        self.TARGET_SIZE=31

        self.LR=1e-4 # 学习率
        self.EPOCH=100  # 训练轮次
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.MODEL_DIR='./output/model/'
