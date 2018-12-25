import sys,os
sys.path.append(os.getcwd())
from config import config

#wider face original images path
path_to_image = '%s/data/HollywoodHeads/JPEGImages'%config.root

target_anno_file = '%s/prepare_data/annotations/anno.txt'%config.root

target_train_list_file = '%s/data/mtcnn/imglists/train.txt'%config.root

with open(target_anno_file, 'r') as f:
    lines = f.readlines()
num = len(lines)
f = open(target_train_list_file,'w')
for i in range(num):
    cur_line = path_to_image+'/'+lines[i].split()[0]+' \n'
    f.write(cur_line)
f.close()



