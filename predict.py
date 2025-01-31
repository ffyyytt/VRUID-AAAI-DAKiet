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
parser.add_argument("-category", help="YAML file", nargs='?', type=str, default="figure")
parser.add_argument("-datapath", help="Path to dataset", nargs='?', type=str, default="/kaggle/input/aaai-25-visually-rich-document-vrd-iu-leaderboard")
parser.add_argument("-modelpath", help="Path to dataset", nargs='?', type=str, default="/kaggle/input/vruid-aaai-dakiet/pytorch/figure/2")
parser.add_argument("-device", help="Path to dataset", nargs='?', type=str, default="cuda")
args = parser.parse_args()


if args.category == "table":
    max_length = 16
    args.child_category = "table_caption" 
    args.parent_category = ["table"]
elif args.category == "figure":
    max_length = 8
    args.child_category = "figure_caption" 
    args.parent_category = ["figure"]
elif args.category == "form":
    max_length = 16
    args.child_category = "form"
    args.parent_category = ["summary", "abstract", "section", "subsection", "subsubsection", "subsubsubsection"]
elif args.category == "list":
    args.child_category = "list"
    args.parent_category = ["paragraph", "section", "subsection", "subsubsection", "subsubsubsection"]
    max_length = 8
elif args.category == "form_body":
    max_length = 16
    args.child_category = "form_body"
    args.parent_category = ["form_title", "summary", "abstract", "section", "subsection", "subsubsection", "subsubsubsection"]



test_data = pickle.load(open(f"{args.datapath}/val_data.pkl", "rb")) | pickle.load(open(f"{args.datapath}/test_data.pkl", "rb"))

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
        r = 0
        for idx in range(len(self.data.keys())):
            r += len([ self.data[self.filename[idx]]["components"][i]["object_id"] for i in range(len(self.data[self.filename[idx]]["components"])) if self.data[self.filename[idx]]["components"][i]["category"] == args.child_category ])
            
        return r

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
        true_idx = 0
        c = len([ self.data[self.filename[true_idx]]["components"][i]["object_id"] for i in range(len(self.data[self.filename[true_idx]]["components"])) if self.data[self.filename[true_idx]]["components"][i]["category"] == args.child_category ])
        while idx >= c:
            true_idx += 1
            c += len([ self.data[self.filename[true_idx]]["components"][i]["object_id"] for i in range(len(self.data[self.filename[true_idx]]["components"])) if self.data[self.filename[true_idx]]["components"][i]["category"] == args.child_category ])

        child_ids = [ [ self.data[self.filename[true_idx]]["components"][i]["object_id"] for i in range(len(self.data[self.filename[true_idx]]["components"])) if self.data[self.filename[true_idx]]["components"][i]["category"] == args.child_category ][-c + idx] ]
        return self.get(true_idx, child_ids)
    
    def get(self, idx, child_ids):
        name = self.filename[idx]
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
                    page_image = np.asarray(Image.open(f"{args.datapath}//val/val/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
                except:
                    try:
                        page_image = np.asarray(Image.open(f"{args.datapath}/test/test/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
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
                    page_image = np.asarray(Image.open(f"{args.datapath}/val/val/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
                except:
                    try:
                        page_image = np.asarray(Image.open(f"{args.datapath}/test/test/{namer}_page-{self.data[name]['components'][i]['page']}.png"))
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
        
        # logits = self.cosine(features_c, features_p, labels)
        return features_c, features_p
    
test_dataset = torch.utils.data.DataLoader(TestDataset(test_data), batch_size=1, shuffle=False, num_workers=16)
device = torch.device(args.device)

image_model = timm.create_model("timm/swinv2_cr_tiny_ns_224.sw_in1k", pretrained=True, num_classes=0).to(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
text_model = BartModel.from_pretrained("facebook/bart-large").to(device)

model = ModelFactory(image_model, text_model).to(device)
model.load_state_dict(torch.load(f"{args.modelpath}/{args.category}.pth", map_location=device).state_dict())
if torch.cuda.device_count() > 1:
    text_model = torch.nn.DataParallel(text_model)
    image_model = torch.nn.DataParallel(image_model)
    # model = torch.nn.DataParallel(model)
scaler = torch.amp.GradScaler(enabled=True)

model.eval()
pbar = tqdm(test_dataset)
predictions = []
child_ids = []
parent_ids = []
for batch in pbar:
    gc.collect()
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
        features_c, features_p = model(child_images, parent_images, child_texts, parent_texts, child_metadata, parent_metadata)
        output = torch.nn.functional.linear(torch.nn.functional.normalize(features_c).detach().cpu(), torch.nn.functional.normalize(features_p).detach().cpu())
        child_ids.append( batch["child_ids"] )
        parent_ids.append( batch["parent_ids"] )
        predictions.append(output.detach().cpu().numpy())

with open(f'output_{args.category}.pickle', 'wb') as handle:
    pickle.dump([child_ids, parent_ids, predictions], handle, protocol=pickle.HIGHEST_PROTOCOL)
