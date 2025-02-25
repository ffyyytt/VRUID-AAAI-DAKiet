import gc
import os
import math
import timm
import torch
import pickle
import random
import argparse
import torchvision

import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from transformers import BartModel, BartTokenizer, RobertaTokenizer, RobertaModel
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser("VRUID")
parser.add_argument("-category", help="Category: [table, figure, form, list, form_body]", nargs='?', type=str, default="figure")
args = parser.parse_args()

if not os.path.exists(args.category):
    os.makedirs(args.category)

max_length = 128
if args.category == "table":
    args.child_category = "table_caption" 
    args.parent_category = ["table"]
elif args.category == "figure":
    args.child_category = "figure_caption" 
    args.parent_category = ["figure"]
elif args.category == "form":
    args.child_category = "form"
    args.parent_category = ["summary", "abstract", "section", "subsection", "subsubsection", "subsubsubsection"]
elif args.category == "list":
    args.child_category = "list"
    args.parent_category = ["paragraph", "section", "subsection", "subsubsection", "subsubsubsection"]
    max_length = 48
elif args.category == "form_body":
    args.child_category = "form_body"
    args.parent_category = ["form_title", "summary", "abstract", "section", "subsection", "subsubsection", "subsubsubsection"]



test_data = pickle.load(open("dataset/val_data.pkl", "rb")) | pickle.load(open("dataset/test_data.pkl", "rb"))
train_data = pickle.load(open("dataset/train_data.pkl", "rb"))

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, img_size=224):
        self.data = data
        self.img_size = img_size
        self.filename = list(data.keys())
        self.augmentation = A.Compose([A.LongestMaxSize(max_size=self.img_size, interpolation=1),
                                       A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0, value=(0,0,0))
                                      ])
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.data.keys())

    def get_text(self, obj):
        text = ""
        for key in ["fixed_caption", "discription", "text", "fixed_text"]:
            if key in obj:
                if type(obj[key]) is dict:
                    text = text + " " + " ".join([v.replace(".", "").replace("\n", "") for v in obj[key].values()])
                else:
                    text = text + " " + obj[key].replace(".", "").replace("\n", "")
        return text

    def __getitem__(self, idx):
        name = self.filename[idx]
        child_ids = [ self.data[name]["components"][i]["object_id"] for i in range(len(self.data[name]["components"])) if self.data[name]["components"][i]["category"] == args.child_category ]
        parent_ids = [ self.data[name]["components"][i]["object_id"] for i in range(len(self.data[name]["components"])) if self.data[name]["components"][i]["category"] in args.parent_category ] + [-1]
        child_texts = [""]*len(child_ids)
        parent_texts = [""]*len(parent_ids)
        child_images = [np.zeros([3, self.img_size, self.img_size])]*len(child_ids)
        parent_images = [np.zeros([3, self.img_size, self.img_size])]*len(parent_ids)
        child_metadata = [[-1, -1, -1] for _ in range(len(child_ids))]
        parent_metadata = [[-1, -1, -1] for _ in range(len(parent_ids))]
        labels = []

        for i in range(len(self.data[name]["components"])):
            if self.data[name]["components"][i]["object_id"] in child_ids:
                try:
                    page_image = np.asarray(Image.open(f'''dataset/train/train/{name}_page-{self.data[name]["components"][i]["page"]}.png'''))
                except:
                    page_image = np.zeros([2048, 2048, 3])
                object_img = page_image[max(int(self.data[name]["components"][i]['bbox'][1])-50, 0):int(self.data[name]["components"][i]['bbox'][1]+self.data[name]["components"][i]['bbox'][3])+50, 
                                        max(int(self.data[name]["components"][i]['bbox'][0])-50, 0):int(self.data[name]["components"][i]['bbox'][0]+self.data[name]["components"][i]['bbox'][2])+50]
                child_texts[child_ids.index(self.data[name]["components"][i]["object_id"])] = self.get_text(self.data[name]["components"][i])
                child_metadata[child_ids.index(self.data[name]["components"][i]["object_id"])] = [self.data[name]["ordered_id"].index(self.data[name]["components"][i]["object_id"]),
                                                                                                  self.data[name]["components"][i]["page"],
                                                                                                    self.data[name]["components"][i]["category_id"]
                                                                                                 ]
                child_images[child_ids.index(self.data[name]["components"][i]["object_id"])] = self.transform(self.augmentation(image=object_img)['image'])

                if len(self.data[name]["components"][i]["relations"]["parent"]) > 0:
                    labels.append(parent_ids.index(self.data[name]["components"][i]["relations"]["parent"][0]))
                else:
                    labels.append(parent_ids.index(-1))

            if self.data[name]["components"][i]["object_id"] in parent_ids:
                try:
                    page_image = np.asarray(Image.open(f'''dataset/train/train/{name}_page-{self.data[name]["components"][i]["page"]}.png'''))
                except:
                    page_image = np.zeros([2048, 2048, 3])
                object_img = page_image[max(int(self.data[name]["components"][i]['bbox'][1])-50, 0):int(self.data[name]["components"][i]['bbox'][1]+self.data[name]["components"][i]['bbox'][3])+50, 
                                        max(int(self.data[name]["components"][i]['bbox'][0])-50, 0):int(self.data[name]["components"][i]['bbox'][0]+self.data[name]["components"][i]['bbox'][2])+50]
                parent_texts[parent_ids.index(self.data[name]["components"][i]["object_id"])] = self.get_text(self.data[name]["components"][i])
                parent_metadata[parent_ids.index(self.data[name]["components"][i]["object_id"])] = [self.data[name]["ordered_id"].index(self.data[name]["components"][i]["object_id"]),
                                                                                                    self.data[name]["components"][i]["page"],
                                                                                                    self.data[name]["components"][i]["category_id"]
                                                                                                   ]
                parent_images[parent_ids.index(self.data[name]["components"][i]["object_id"])] = self.transform(self.augmentation(image=object_img)['image'])
        
        if len(labels) == 0:
            return self.__getitem__(random.randrange(len(self.filename)))
        
        return {
            "child_ids": child_ids,
            "parent_ids": parent_ids,
            "child_texts": child_texts,
            "parent_texts": parent_texts,
            "child_images": torch.from_numpy(np.asarray(child_images)).to(torch.float32),
            "parent_images": torch.from_numpy(np.asarray(parent_images)).to(torch.float32),
            "child_metadata": torch.from_numpy(np.asarray(child_metadata)).to(torch.float32),
            "parent_metadata": torch.from_numpy(np.asarray(parent_metadata)).to(torch.float32),
            "labels": torch.from_numpy(np.asarray(labels)),
        }
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, img_size=224):
        self.data = data
        self.img_size = img_size
        self.filename = list(data.keys())
        self.augmentation = A.Compose([A.LongestMaxSize(max_size=self.img_size, interpolation=1),
                                       A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0, value=(0,0,0))
                                      ])
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.data.keys())

    def get_text(self, obj):
        text = ""
        for key in ["fixed_caption", "discription", "text", "fixed_text"]:
            if key in obj:
                if type(obj[key]) is dict:
                    text = text + " " + " ".join([v.replace(".", "").replace("\n", "") for v in obj[key].values()])
                else:
                    text = text + " " + obj[key].replace(".", "").replace("\n", "")
        return text

    def __getitem__(self, idx):
        name = self.filename[idx]
        child_ids = [ self.data[name]["components"][i]["object_id"] for i in range(len(self.data[name]["components"])) if self.data[name]["components"][i]["category"] == args.child_category ]
        parent_ids = [ self.data[name]["components"][i]["object_id"] for i in range(len(self.data[name]["components"])) if self.data[name]["components"][i]["category"] in args.parent_category ] + [-1]
        child_texts = [""]*len(child_ids)
        parent_texts = [""]*len(parent_ids)
        child_images = [np.zeros([3, self.img_size, self.img_size])]*len(child_ids)
        parent_images = [np.zeros([3, self.img_size, self.img_size])]*len(parent_ids)
        child_metadata = [[-1, -1, -1] for _ in range(len(child_ids))]
        parent_metadata = [[-1, -1, -1] for _ in range(len(parent_ids))]

        for i in range(len(self.data[name]["components"])):
            if self.data[name]["components"][i]["object_id"] in child_ids:
                namer = name.replace("'", "_")
                try:
                    page_image = np.asarray(Image.open(f"dataset/val/val/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
                except:
                    try:
                        page_image = np.asarray(Image.open(f"dataset/test/test/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
                    except:
                        page_image = np.zeros([2048, 2048, 3])
                object_img = page_image[max(int(self.data[name]["components"][i]['bbox'][1])-50, 0):int(self.data[name]["components"][i]['bbox'][1]+self.data[name]["components"][i]['bbox'][3])+50, 
                                        max(int(self.data[name]["components"][i]['bbox'][0])-50, 0):int(self.data[name]["components"][i]['bbox'][0]+self.data[name]["components"][i]['bbox'][2])+50]
                child_texts[child_ids.index(self.data[name]["components"][i]["object_id"])] = self.get_text(self.data[name]["components"][i])
                child_metadata[child_ids.index(self.data[name]["components"][i]["object_id"])] = [self.data[name]["ordered_id"].index(self.data[name]["components"][i]["object_id"]),
                                                                                                  self.data[name]["components"][i]["page"],
                                                                                                    self.data[name]["components"][i]["category_id"]
                                                                                                 ]
                child_images[child_ids.index(self.data[name]["components"][i]["object_id"])] = self.transform(self.augmentation(image=object_img)['image'])
                
            if self.data[name]["components"][i]["object_id"] in parent_ids:
                namer = name.replace("'", "_")
                try:
                    page_image = np.asarray(Image.open(f"dataset/val/val/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
                except:
                    try:
                        page_image = np.asarray(Image.open(f"dataset/test/test/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
                    except:
                        page_image = np.zeros([2048, 2048, 3])
                object_img = page_image[max(int(self.data[name]["components"][i]['bbox'][1])-50, 0):int(self.data[name]["components"][i]['bbox'][1]+self.data[name]["components"][i]['bbox'][3])+50, 
                                        max(int(self.data[name]["components"][i]['bbox'][0])-50, 0):int(self.data[name]["components"][i]['bbox'][0]+self.data[name]["components"][i]['bbox'][2])+50]
                parent_texts[parent_ids.index(self.data[name]["components"][i]["object_id"])] = self.get_text(self.data[name]["components"][i])
                parent_metadata[parent_ids.index(self.data[name]["components"][i]["object_id"])] = [self.data[name]["ordered_id"].index(self.data[name]["components"][i]["object_id"]),
                                                                                                    self.data[name]["components"][i]["page"],
                                                                                                    self.data[name]["components"][i]["category_id"]
                                                                                                   ]
                parent_images[parent_ids.index(self.data[name]["components"][i]["object_id"])] = self.transform(self.augmentation(image=object_img)['image'])

        if len(labels) == 0:
            return self.__getitem__(random.randrange(len(self.filename)))
        
        return {
            "child_ids": child_ids,
            "parent_ids": parent_ids,
            "child_texts": child_texts,
            "parent_texts": parent_texts,
            "child_images": torch.from_numpy(np.asarray(child_images)).to(torch.float32),
            "parent_images": torch.from_numpy(np.asarray(parent_images)).to(torch.float32),
            "child_metadata": torch.from_numpy(np.asarray(child_metadata)).to(torch.float32),
            "parent_metadata": torch.from_numpy(np.asarray(parent_metadata)).to(torch.float32),
        }

class Cosine(torch.nn.Module):
    def __init__(self, s=8.0, m=0.3):
        super(Cosine, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features_c, features_p, label):
        cos = torch.nn.functional.linear(torch.nn.functional.normalize(features_c), torch.nn.functional.normalize(features_p))
        if label is None:
            return cos*self.s, cos*self.s
        sin = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
        cos_add = cos * self.cos_m - sin * self.sin_m
        label = torch.nn.functional.one_hot(label, num_classes=len(features_p)).to(cos.dtype)
        output = (label * cos_add) + ((1.0 - label) * cos)
        output *= self.s
        return cos, output
    
class ModelFactory(torch.nn.Module):
    def __init__(self, image_model, text_model, hashlength=512):
        super(ModelFactory, self).__init__()
        self.image_model = image_model
        self.text_model = text_model

        self.linear_header_p = torch.nn.Sequential(torch.nn.Linear(1795, hashlength),
                                                   torch.nn.PReLU(),
                                                   # torch.nn.Linear(hashlength, hashlength),
                                                   # torch.nn.PReLU(),
                                                  )
        self.linear_header_c = torch.nn.Sequential(torch.nn.Linear(1795, hashlength),
                                                   torch.nn.PReLU(),
                                                   # torch.nn.Linear(hashlength, hashlength),
                                                   # torch.nn.PReLU(),
                                                  )
        self.cosine = Cosine()

    def forward(self, child_images, parent_images, child_texts, parent_texts, child_metadata, parent_metadata, labels=None):
        child_image_features = self.image_model(child_images)
        parent_image_features = self.image_model(parent_images)
        
        child_text_features = self.text_model(**child_texts).last_hidden_state[:, 0, :]
        parent_text_features = self.text_model(**parent_texts).last_hidden_state[:, 0, :]
        
        child_features = torch.cat([child_image_features, child_text_features, child_metadata], dim=1)
        parent_features = torch.cat([parent_image_features, parent_text_features, parent_metadata], dim=1)

        features_c = self.linear_header_c(child_features)
        features_p = self.linear_header_p(parent_features)
        
        logits = self.cosine(features_c, features_p, labels)
        return logits
    
class Accuracy:
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

    def update(self, predictions, targets):
        predicted_labels = torch.argmax(predictions, dim=1)
        self.correct_count += (predicted_labels == targets).sum().item()
        self.total_count += targets.size(0)

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def compute(self):
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count
    
test_dataset = torch.utils.data.DataLoader(TestDataset(test_data), batch_size=1, shuffle=False, num_workers=16)
train_dataset = torch.utils.data.DataLoader(TrainDataset(train_data), batch_size=1, shuffle=True, num_workers=16) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = timm.create_model("timm/swinv2_cr_tiny_ns_224.sw_in1k", pretrained=True, num_classes=0).to(device)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
text_model = BartModel.from_pretrained("facebook/bart-large").to(device)

# for param in image_model.parameters():
#     param.requires_grad = False
# for param in text_model.parameters():
#     param.requires_grad = False

# if torch.cuda.device_count() > 1:
#     text_model = torch.nn.DataParallel(text_model)

model = ModelFactory(image_model, text_model).to(device)
model.load_state_dict(torch.load(f"{args.category}.pth").state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

loss_function = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler(enabled=True)
metrics = [Accuracy()]


for epoch in range(100):
    gc.collect()
    model.train()
    for metric in metrics:
        metric.reset()
    pbar = tqdm(train_dataset)
    for i, batch in enumerate(pbar):
        if len(batch["labels"][0]) == 0:
            continue
        child_texts = tokenizer([x[0] for x in batch["child_texts"]], return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
        parent_texts = tokenizer([x[0] for x in batch["parent_texts"]], return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
        for k in child_texts :
            child_texts[k] = child_texts[k].to(device)
            parent_texts[k] = parent_texts[k].to(device)
        child_images = batch["child_images"][0].to(device)
        parent_images = batch["parent_images"][0].to(device)
        child_metadata = batch["child_metadata"][0].to(device)
        parent_metadata = batch["parent_metadata"][0].to(device)
        labels = batch["labels"][0].to(device)
        with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=True):
            real, output = model(child_images, parent_images, child_texts, parent_texts, child_metadata, parent_metadata, labels)
            loss = loss_function(output, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        for metric in metrics:
            metric.update(real, labels)
        pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Accuracy: {metrics[0].compute():.5f}")

    model.eval()
    pbar = tqdm(test_dataset)
    predictions = []
    child_ids = []
    parent_ids = []
    for batch in pbar:
        if len(batch["child_ids"]) == 0:
            child_ids.append( batch["child_ids"] )
            parent_ids.append( batch["parent_ids"] )
            predictions.append( [] )
            continue
        child_texts = tokenizer([x[0] for x in batch["child_texts"]], return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
        parent_texts = tokenizer([x[0] for x in batch["parent_texts"]], return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
        for k in child_texts :
            child_texts[k] = child_texts[k].to(device)
            parent_texts[k] = parent_texts[k].to(device)
        child_images = batch["child_images"][0].to(device)
        parent_images = batch["parent_images"][0].to(device)
        child_metadata = batch["child_metadata"][0].to(device)
        parent_metadata = batch["parent_metadata"][0].to(device)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16, enabled=True):
            output, _ = model(child_images, parent_images, child_texts, parent_texts, child_metadata, parent_metadata)
            child_ids.append( batch["child_ids"] )
            parent_ids.append( batch["parent_ids"] )
            predictions.append(output.detach().cpu().numpy())

    torch.save(model, f"{args.category}.pth")
    with open(f'{args.category}/output_{500+epoch}.pickle', 'wb') as handle:
        pickle.dump([child_ids, parent_ids, predictions], handle, protocol=pickle.HIGHEST_PROTOCOL)
