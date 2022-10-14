import os
from cv2 import eigenNonSymmetric #下载文件路径
import pandas as pd  #寻找注释文件（annotation file）
import spacy #For tokenizer(分词器)
import torch
from torch.nn.utils.rnn import pad_sequence #补齐空白padding
from torch.utils.data import DataLoader,Dataset 
from PIL import Image   #Load img
import torchvision.transforms as transforms

##We want to convert text →numerical values
#我们希望转换文本→数值
#1.We need a Vocabulary mapping each word to a index
#1.我们需要一个词汇将每个单词映射到一个索引
#2. We need to setup a Pytorch dataset to load the data
#2.我们需要设置一个Python数据集来加载数据
#3. Setup padding of every batch (all examples should be
#of same seq_len and setup dataloader)
#3.每批的设置填充(所有示例都应该是)

#Download with: python -m spacy download en
spacy_eng = spacy.load("en")

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {"<PAD>":0,"<SOS>":1,"<EOS>":2,"<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text): 
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self,sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = 1

            else:
                frequencies[word] += 1

            if frequencies[word] == self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1 

    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK"]
            for token in tokenized_text
        ]

#定义一个读取数据的类
class FlickrDataset(Dataset):
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.roor_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        #Get img, caption colums
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold) #Vocabulary 自定义类
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir,img_id)).convert("RGB") #转化成RGB，不然可能多一个通道（透明色）

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img,torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)

        return imgs,targets
    
def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True
):
    dataset = FlickrDataset(root_folder,annotation_file,transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader

def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]
    )
    dataloader = get_loader("图片地址",annotation_file="",transform=transform)

    for idx,(imgs,captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)

if __name__ == "__main__":
    main()

