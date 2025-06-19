# app.py
import numpy as np
import torch
from torch import nn
from flask import Flask, request, render_template, jsonify
from model import LET
from downstream import task, predictor as DownPredictor
import pandas as pd
import data
from typing import Dict, Any

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The global model stores the dictionary
models: Dict[str, Dict[str, Any]] = {
    'dp': {'model': None, 'predictor': None, 'config': None},
    'tte': {'model': None, 'predictor': None, 'config': None},
    'sts': {'model': None, 'predictor': None, 'config': None}
}

stat = pd.read_hdf('use_case/stat.h5', key='expanded_stat')
road_info = pd.read_csv('use_case/road_info.csv', encoding='gbk')

road_info_dict = {}
for _, row in road_info.iterrows():
    road_info_dict[row['road']] = {
        'lng': row['road_lng'],
        'lat': row['road_lat'],
        'length': row['length']
    }

def calculate_circle(lng, lat, length_meters):
    return {
        'center': (lng, lat),
        'radius': length_meters * 0.7 
    }

TRIP_COLS = ['tod', 'road', 'road_prop', 'lng', 'lat', 'weekday', 'seq_i', 'seconds',
             'speed', 'acceleration', 'heading_angle', 'road_type', 'lanes', 'direction',]
trip_denormalizer =data.Denormalizer(stat, feat_cols=[0, 3, 4, 8, 9, 10], norm_type='minmax')

class TaskConfig:
    """Dynamic task configuration class"""
    def __init__(self, task_type: str):
        # Base configuration
        self.d_model = 768
        self.output_size = 128
        self.model_class = 'gpt2'
        self.lora = True
        self.lora_alpha = 16
        self.lora_dim = 8
        self.kernel_size = 5
        self.num_virtual_anchors = 15

        # Task-specific configuration
        if task_type == 'dp':
            self.add_feats = [1, 11, 12, 13]
            self.add_embeds = [4315, 8, 5, 2]
            self.dis_feats = [1, 5, 7]
            self.num_embeds = [4315, 7, 24]
            self.con_feats = [0, 3, 4, 8, 9, 10]
        elif task_type == 'tte':
            self.add_feats = [1, 11, 12, 13]
            self.add_embeds = [4315, 8, 5, 2]
            self.dis_feats = [1, 5, 7]
            self.num_embeds = [4315, 7, 24]
            self.con_feats = [3, 4] 
        elif task_type == 'sts':
            self.add_feats = [1, 11, 12, 13]
            self.add_embeds = [4315, 8, 5, 2]
            self.dis_feats = [1, 5, 7]
            self.num_embeds = [4315, 7, 24]
            self.con_feats = [0, 3, 4, 8, 9, 10]

def load_models():
    """Load all task models"""
    try:
        dp_config = TaskConfig('dp')
        models['dp']['model'] = LET(
            d_model=dp_config.d_model,
            output_size=dp_config.output_size,
            add_feats=dp_config.add_feats,
            add_embeds=dp_config.add_embeds,
            dis_feats=dp_config.dis_feats,
            num_embeds=dp_config.num_embeds,
            con_feats=dp_config.con_feats,
            model_class=dp_config.model_class,
            lora=dp_config.lora,
            lora_alpha=dp_config.lora_alpha,
            lora_dim=dp_config.lora_dim,
            kernel_size=dp_config.kernel_size,
            num_virtual_anchors=dp_config.num_virtual_anchors
        )
        models['dp']['model'].load_state_dict(
            torch.load('model_save/destination/LET-d768-o128-1,5,7,0,3,4,8,9,10-gpt2-m30-a15-psp-lora8,16-conv-k5-poi.model', 
                      map_location=device)
        )
        models['dp']['predictor'] = DownPredictor.FCPredictor(input_size=128, output_size=4315, hidden_size=256)
        models['dp']['predictor'].load_state_dict(torch.load('model_save/destination/fc-h256.model', map_location=device))


        tte_config = TaskConfig('tte')
        models['tte']['model'] = LET(
            d_model=tte_config.d_model,
            output_size=tte_config.output_size,
            add_feats=tte_config.add_feats,
            add_embeds=tte_config.add_embeds,
            dis_feats=tte_config.dis_feats,
            num_embeds=tte_config.num_embeds,
            con_feats=tte_config.con_feats,
            model_class=tte_config.model_class,
            lora=tte_config.lora,
            lora_alpha=tte_config.lora_alpha,
            lora_dim=tte_config.lora_dim,
            kernel_size=tte_config.kernel_size,
            num_virtual_anchors=tte_config.num_virtual_anchors
        )
        models['tte']['model'].load_state_dict(
            torch.load('model_save/tte/LET-d768-o128-1,5,7,3,4-gpt2-m30-a15-psp-lora8,16-conv-k5-poi.model', 
                      map_location=device)
        )
        models['tte']['predictor'] = DownPredictor.FCPredictor(input_size=128, output_size=1, hidden_size=256)
        models['tte']['predictor'].load_state_dict(torch.load('model_save/tte/fc-h256.model', map_location=device))


        sts_config = TaskConfig('sts')
        models['sts']['model'] = LET(
            d_model=sts_config.d_model,
            output_size=sts_config.output_size,
            add_feats=sts_config.add_feats,
            add_embeds=sts_config.add_embeds,
            dis_feats=sts_config.dis_feats,
            num_embeds=sts_config.num_embeds,
            con_feats=sts_config.con_feats,
            model_class=sts_config.model_class,
            lora=sts_config.lora,
            lora_alpha=sts_config.lora_alpha,
            lora_dim=sts_config.lora_dim,
            kernel_size=sts_config.kernel_size,
            num_virtual_anchors=sts_config.num_virtual_anchors
        )
        models['sts']['model'].load_state_dict(
            torch.load('model_save/generative_b16-lr0.0001/LOSS_trip_causual-LOSS_trip-dis-0.5-1-con-0.5-3,4,8,9,10-LOSS_poi/LET-d768-o128-1,5,7,0,3,4,8,9,10-gpt2-m30-a15-psp-lora8,16-conv-k5-poi.model',
                      map_location=device)
        )
        models['sts']['predictor'] = DownPredictor.NonePredictor()

        for task in models.values():
            task['model'].eval().to(device)
            task['predictor'].eval().to(device)
        
        print("✅ All models loaded successfully")

    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise

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
    #print(o_pois)
    return {
        'x': torch.FloatTensor(processed_trip).unsqueeze(0).to(device),
        'valid_len': torch.tensor([len(processed_trip)], dtype=torch.long).to(device), 
        'o_pois': [o_pois[0]] if o_pois else [""],
        'd_pois': [d_pois[0]] if d_pois else [""],
        'start_weekday': torch.tensor([weekday], dtype=torch.long).to(device),
        'start_hour': torch.tensor([hour], dtype=torch.long).to(device)
    }


ALL_TRIPS = np.load('use_case/new_trip_2.npz', allow_pickle=True)['arr_0']
OD_POIS = np.load('use_case/odpois-3_2.npz', allow_pickle=True)




@app.route('/')
def index():
    #indices = list(range(30))
    indices = [0,4,11,16,19,23,24,25,28,29]
    trajectories = []
    
    for idx in indices:
        trip = ALL_TRIPS[idx]
        denorm_trip = trip_denormalizer([0,3,4,8,9,10], trip[:])
        time_str = pd.to_datetime(trip[0][7], unit='s').strftime('%Y-%m-%d %H:%M')
        preview_list = denorm_trip[:, [3,4,8,10,7]].tolist()
        trajectories.append({
            'id': int(idx),
            'start_lng': float(denorm_trip[0][3]),
            'start_lat': float(denorm_trip[0][4]),
            'start_time': time_str,
            'preview': preview_list[:-5],
            'preview_full': preview_list[:]
        })
    
    return render_template('index.html', trajectories=trajectories)

@app.route('/api/predict/<int:trip_id>', methods=['POST'])
def predict(trip_id):
    try:
        trip_data = ALL_TRIPS[trip_id]
        o_pois = OD_POIS['arr_0'][trip_id]
        #print(o_pois)
        #print(trip_data)
        show_data = trip_denormalizer([0, 3, 4, 8, 9, 10], trip_data)
        inputs = process_input(trip_data, o_pois)
        #print(inputs)
        suffix_prompt = "目的地所在路段为"
        downstream_token = nn.Parameter(torch.zeros(models['dp']['model'].emb_size).float(), requires_grad=True)

        with torch.no_grad():
            embeddings = models['dp']['model'].forward(
                x=inputs['x'],
                valid_len=inputs['valid_len']-5,
                o_pois=inputs['o_pois'],
                d_pois=inputs['d_pois'],
                start_weekday=inputs['start_weekday'],
                start_hour=inputs['start_hour'],
                suffix_prompt=suffix_prompt,
                token=downstream_token,
                d_mask=True
            )
            o_pois=inputs['o_pois']
            #print(embeddings)
            pred = models['dp']['predictor'](embeddings).argmax(-1).item()
            print(pred)

        road_info = road_info_dict.get(pred)

        if road_info:
            circle = calculate_circle(
                road_info['lng'], 
                road_info['lat'], 
                road_info['length']
            )
            destination_data = {
                'type': 'circle',
                'center': {'lng': circle['center'][0], 'lat': circle['center'][1]},
                'radius': circle['radius']
            }
        else:
            destination_data = None

        time_points = [
            pd.to_datetime(ts, unit='s').strftime('%H:%M:%S')
            for ts in show_data[:-5, 7]
        ]

        return jsonify({
            'prediction': pred,
            'coordinates': show_data[:, [3,4]].tolist(),
            'timestamps': time_points,
            'pois': inputs['o_pois'],
            'destination': destination_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/predict_tte/<int:trip_id>', methods=['POST'])
def predict_tte(trip_id):
    try:
        trip_data = ALL_TRIPS[trip_id]
        o_pois = OD_POIS['arr_0'][trip_id]
        d_pois = OD_POIS['arr_1'][trip_id]
        show_data = trip_denormalizer([0, 3, 4, 8, 9, 10], trip_data)
        inputs = process_input(trip_data, o_pois, d_pois)
        
        suffix_prompt = "旅行时间为"
        downstream_token = nn.Parameter(torch.zeros(models['tte']['model'].emb_size).float(), requires_grad=True)

        with torch.no_grad():
            embeddings = models['tte']['model'].forward(
                x=inputs['x'],
                valid_len=inputs['valid_len'],
                o_pois=inputs['o_pois'],
                d_pois=inputs['d_pois'],
                start_weekday=inputs['start_weekday'],
                start_hour=inputs['start_hour'],
                suffix_prompt=suffix_prompt,
                token=downstream_token
            )
            pred_seconds = models['tte']['predictor'](embeddings).item()
            print(pred_seconds)

        time_points = [
            pd.to_datetime(ts, unit='s').strftime('%H:%M:%S')
            for ts in show_data[:-5, 7]
        ]

        return jsonify({
            'prediction': pred_seconds,
            'coordinates': show_data[:, [3,4]].tolist(),
            'timestamps': time_points
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



HOP_QRY_TGT = np.load('use_case/hopqrytgt-1000_2.npz', allow_pickle=True)['arr_0']
HOP_POIS = np.load('use_case/hopqrytgtpois-1000-3_2.npz', allow_pickle=True)
NEG_INDICES = np.load('use_case/hopnegindex-1000-5000_2.npz', allow_pickle=True)['arr_0']
STS_EMBEDDINGS = np.load('sts_embeddings.npy')

@app.route('/api/search/<int:query_id>', methods=['POST'])
def search_similar(query_id):
    try:
        if query_id >= len(HOP_QRY_TGT) // 2:
            return jsonify({"error": "Invalid query ID"}), 400

        query_trip = HOP_QRY_TGT[query_id]
        target_trip = HOP_QRY_TGT[query_id + len(HOP_QRY_TGT) // 2]

        query_o_pois = HOP_POIS['arr_0'][query_id]
        target_O_pois = HOP_POIS['arr_0'][query_id + len(HOP_POIS['arr_1']) // 2]

        query_d_pois = HOP_POIS['arr_1'][query_id]
        target_d_pois = HOP_POIS['arr_1'][query_id + len(HOP_POIS['arr_1']) // 2]

        query_input = process_input(query_trip, query_o_pois, query_d_pois)
        target_input = process_input(target_trip, target_O_pois, target_d_pois)

        neg_indices = NEG_INDICES[query_id]

        with torch.no_grad():
            # Get the query trajectory embedding
            query_embed = models['sts']['model'].forward(
                x=query_input['x'],
                valid_len=query_input['valid_len'],
                o_pois=query_input['o_pois'],
                d_pois=query_input['o_pois'],
                start_weekday=query_input['start_weekday'],
                start_hour=query_input['start_hour']
            )

            print(query_embed.shape)
            target_embed = models['sts']['model'].forward(
                x=target_input['x'],
                valid_len=target_input['valid_len'],
                o_pois=target_input['o_pois'],
                d_pois=query_input['o_pois'],
                start_weekday=target_input['start_weekday'],
                start_hour=target_input['start_hour']
            )

            print(target_embed.shape)

        # Precomputed negative sample embeddings are used directly
        print(STS_EMBEDDINGS.shape)
        neg_embeds = STS_EMBEDDINGS[neg_indices]
        print(neg_embeds.shape)

        # Calculating similarity
        query_embed_np = query_embed.squeeze()  # (128,)
        target_embed_np = target_embed.squeeze()  # (128,)
        neg_embeds_np = neg_embeds # (5000, 128)

        pos_sim = -np.linalg.norm(query_embed_np - target_embed_np)
        neg_sims = -np.linalg.norm(query_embed_np - neg_embeds_np, axis=1)

        similarities = np.concatenate([[pos_sim], neg_sims])
        sorted_indices = np.argsort(similarities)[::-1]
        results = {
            "query": {
                "id": query_id,
                "coordinates": trip_denormalizer([0,3,4,8,9,10], query_trip)[:, [3,4]].tolist()
            },
            "target": {
                "id": query_id + len(HOP_QRY_TGT)//2,
                "similarity": float(similarities[0]),
                "coordinates": trip_denormalizer([0,3,4,8,9,10], target_trip)[:, [3,4]].tolist()
            },
            "top_matches": []
        }

        for idx in sorted_indices[1:6]:
            if idx == 0:
                continue
            neg_id = neg_indices[idx-1]
    
            results["top_matches"].append({
                "id": int(neg_id),
                "similarity": float(similarities[idx]),
                "coordinates": trip_denormalizer([0,3,4,8,9,10], ALL_TRIPS[neg_id])[:, [3,4]].tolist()
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000)


