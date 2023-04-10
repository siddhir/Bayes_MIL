import os
import shutil
from tqdm import tqdm

src_path = '/data1/liuxiangyu/projects/CLAM-BMIL-v1/CLAM-BMIL4/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/normal'

dst_path = 'cp_files1'

if not os.path.exists(dst_path):
    os.makedirs(dst_path)


for folder in tqdm(os.listdir(src_path)):
    for file in os.listdir(os.path.join(src_path, folder)):
        if 'blockmap.h5' in file:
            shutil.copy(os.path.join(src_path, folder, file), dst_path)