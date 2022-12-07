#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:32:10 2022

@author: mikewang
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import hshap
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from itertools import combinations 
import math
# import cv2
# select device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
M = 4 # total number of players
#%%

model_directory = 'shap_cal/prediction_model/'
filename = model_directory + 'model.pt'
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(filename, map_location=device))
model = model.to(device)
model.eval()
torch.cuda.empty_cache()
model = nn.Sequential(
    model,
    nn.Softmax(dim=1),
)


#%%
image_path ='/Users/mikewang/Library/CloudStorage/OneDrive-JohnsHopkins/Study/Master/Semaster_1/EN.580.697/Interpretable-ML/output_shap/1f8f08ea-b5b3-4f68-94d4-3cc071b7dce8.png'
resized_image_path ='/Users/mikewang/Library/CloudStorage/OneDrive-JohnsHopkins/Study/Master/Semaster_1/EN.580.697/Interpretable-ML/output_shap/original.png'
image = Image.open(image_path)
newsize = (256, 256)
image = image.resize(newsize)
image.save(resized_image_path)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)
image_t = transform(image).to(device)
prep_img = torch.unsqueeze(image_t, 0)

#%%
prob_scores = model(prep_img).detach().numpy()[0]
cl, baseline_v = np.argmax(prob_scores), prob_scores[np.argmax(prob_scores)]
(cl,baseline_v)


#%%
directory = 'output_shap'
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)
exclude_player2_list = []
exclude_player2_team_index_list = []
player_list = []
probList = []
contains_player2_list = []
contains_player2_team_index_list = []
team_index = 0 
all_players = list(range(1,M+1))

for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and f[-4:] == '.png' and "output" in filename:
            ext = filename[6:-4]
            players_excluded = list(ext)
            players_excluded = [int(i) for i in players_excluded]
            players = all_players.copy()
            for excludedPalyerId in players_excluded: 
                players.remove(excludedPalyerId)
            if 2 in players:
                contains_player2_list.append(players)
                contains_player2_team_index_list.append(team_index)
            else: 
                exclude_player2_list.append(players)
                exclude_player2_team_index_list.append(team_index)
            print(filename)
            print(players)
            player_list.append(players).open(f)
            image_t = transform(image).to(device)
            prep_img = torch.unsqueeze(image_t, 0)
            prob_scores = model(prep_img).detach().numpy()[0][cl]
            p = prob_scores
            probList.append(p)
            print(p)
            team_index += 1


#%% calcualte shaply
shap_2 = 0
for i, teams_with2 in enumerate(contains_player2_list):
    
    teams_without2 = teams_with2.copy()
    teams_without2.remove(2)
    subset_size = len(teams_without2)
    prob_excluded = probList[exclude_player2_team_index_list[exclude_player2_list.index(teams_without2)]]
    prob_with = probList[contains_player2_team_index_list[i]]
    shap_2 = shap_2 + (math.factorial(subset_size)*math.factorial(M-subset_size-1))/math.factorial(M)*(prob_with-prob_excluded)
    
print("the shap value of 2 is: " + str(shap_2))

            




#%% load image

#%% generate combination index


#%% mask generation method



