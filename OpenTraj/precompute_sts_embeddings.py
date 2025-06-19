import numpy as np
import torch
from model import LET
from downstream import predictor as DownPredictor
import data
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据和配置
ALL_TRIPS = np.load('use_case/new_trip_2.npz', allow_pickle=True)['arr_0']
OD_POIS = np.load('use_case/odpois-3_2.npz', allow_pickle=True)
stat = pd.read_hdf('use_case/stat.h5', key='expanded_stat')

# 初始化数据处理器
trip_denormalizer = data.Denormalizer(stat, feat_cols=[0, 3, 4, 8, 9, 10], norm_type='minmax')

# 模型配置
class TaskConfig:
    def __init__(self):
        self.d_model = 768
        self.output_size = 128
        self.add_feats = [1, 11, 12, 13]
        self.add_embeds = [4315, 8, 5, 2]
        self.dis_feats = [1, 5, 7]
        self.num_embeds = [4315, 7, 24]
        self.con_feats = [0, 3, 4, 8, 9, 10]
        self.model_class = 'gpt2'
        self.lora = True
        self.lora_alpha = 16
        self.lora_dim = 8
        self.kernel_size = 5
        self.num_virtual_anchors = 15

# 加载模型
config = TaskConfig()
model = LET(
    d_model=config.d_model,
    output_size=config.output_size,
    add_feats=config.add_feats,
    add_embeds=config.add_embeds,
    dis_feats=config.dis_feats,
    num_embeds=config.num_embeds,
    con_feats=config.con_feats,
    model_class=config.model_class,
    lora=config.lora,
    lora_alpha=config.lora_alpha,
    lora_dim=config.lora_dim,
    kernel_size=config.kernel_size,
    num_virtual_anchors=config.num_virtual_anchors
)
model.load_state_dict(torch.load(
    'model_save/generative_b16-lr0.0001/LOSS_trip_causual-LOSS_trip-dis-0.5-1-con-0.5-3,4,8,9,10-LOSS_poi/LET-d768-o128-1,5,7,0,3,4,8,9,10-gpt2-m30-a15-psp-lora8,16-conv-k5-poi.model',
    map_location=device
))
model.eval().to(device)


def process_input(trip_data, o_pois=None, d_pois=None):
    processed_trip = trip_data.copy()
    time_array = pd.to_datetime(trip_data[:, 7], unit='s')
    #print(time_array)
    weekday = int(time_array.weekday[0])
    #print(weekday)
    hour = int(time_array.hour[0])
    #trip_data[:, 7] = time_array.hour.to_numpy()
    processed_trip[:, 7] = time_array.hour.to_numpy()
    #print(hour)
    return {
        'x': torch.FloatTensor(processed_trip).unsqueeze(0).to(device),
        'valid_len': torch.tensor([len(processed_trip)], dtype=torch.long).to(device),  # 确保为LongTensor
        'o_pois': [o_pois[0]] if o_pois else [""],
        'd_pois': [d_pois[0]] if d_pois else [""],
        'start_weekday': torch.tensor([weekday], dtype=torch.long).to(device),    # 转换为张量并指定类型
        'start_hour': torch.tensor([hour], dtype=torch.long).to(device)
    }


# 预计算所有嵌入
embeddings = []
for idx in tqdm(range(len(ALL_TRIPS))):
    trip = ALL_TRIPS[idx]
    o_pois = OD_POIS['arr_0'][idx]
    d_pois = OD_POIS['arr_1'][idx]
    with torch.no_grad():
        inputs = process_input(trip, o_pois, d_pois)
        embed = model.forward(
            x=inputs['x'],
            valid_len=inputs['valid_len'],
            o_pois=inputs['o_pois'],
            d_pois=inputs['d_pois'],
            start_weekday=inputs['start_weekday'],
            start_hour=inputs['start_hour']
        ).cpu().numpy().squeeze(0)  # 压缩batch维度
    print(embed.shape)
    embeddings.append(embed)

np.save('sts_embeddings.npy', np.array(embeddings))
print(f"Embeddings saved with shape: {np.array(embeddings).shape}")      #(7024,128)