import pandas as pd
import os

# 生成数据列表
tdir = "A. Segmentation/1. Original Images/a. Training Set/"
files = os.listdir(tdir)
files = list(set([ fname[:-4].split('_')[0] for fname in files ]))
files = { "filename" : files }
df = pd.DataFrame(files).sample(frac=1)
files_train = { "filename" : [] }

for d in files['filename']:
    files_train['filename'].append(d + ".png")
    if os.path.exists(tdir + d + "_m1.png"):
        files_train['filename'].append(d + "_m1.png")
    if os.path.exists(tdir + d + "_m2.png"):
        files_train['filename'].append(d + "_m2.png")
    if os.path.exists(tdir + d + "_m3.png"):
        files_train['filename'].append(d + "_m3.png")

df1 = pd.DataFrame(files_train).sample(frac=1)
df1[:int(len(files['filename'])* 0.8)].to_csv(f'segmentation_split.csv')
#df[:int(len(files['filename'])* 0.8)].to_csv(f'segmentation_split.csv')

for l in range(len(df)):
    df.iloc[l] += ".png"
df[int(len(files['filename'])* 0.8):].to_csv(f'segmentation_split_test.csv')