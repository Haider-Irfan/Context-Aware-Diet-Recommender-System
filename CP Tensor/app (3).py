"""
Diet Recommender — Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── 1. CP Decomposition Model (same architecture as training) ─────────────────
class CPDietModel(nn.Module):
    def __init__(self, n_users, n_foods, n_contexts, n_nutrients, rank=64):
        super().__init__()
        self.U = nn.Embedding(n_users    + 1, rank)
        self.F = nn.Embedding(n_foods    + 1, rank)
        self.C = nn.Embedding(n_contexts + 1, rank)
        for emb in [self.U, self.F, self.C]:
            nn.init.xavier_uniform_(emb.weight)
        self.emb_drop  = nn.Dropout(0.2)
        self.nut_proj  = nn.Linear(n_nutrients, rank)
        self.mlp = nn.Sequential(
             nn.Linear(rank * 3 + 3, 128),
             nn.BatchNorm1d(128),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(128, 64),
             nn.BatchNorm1d(64),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(64, 32),
             nn.ReLU(),
             nn.Linear(32, 1),
             nn.Sigmoid()
             )         
         
    def forward(self, u, f, ctx, nutrients):
        u_e   = self.emb_drop(self.U(u))
        f_e   = self.emb_drop(self.F(f))
        ctx_e = self.emb_drop(self.C(ctx))
        triple   = u_e * f_e * ctx_e
        n_e      = self.nut_proj(nutrients)
        nut_ctx  = n_e * ctx_e
        u_f      = u_e * f_e
        cp_score = triple.sum(dim=1,  keepdim=True)
        nc_score = nut_ctx.sum(dim=1, keepdim=True)
        uf_score = u_f.sum(dim=1,     keepdim=True)
        combined = torch.cat([triple, nut_ctx, u_f, cp_score, nc_score, uf_score], dim=1)
        return self.mlp(combined)

# ── 2. Load all saved artifacts ───────────────────────────────────────────────
print("Loading model and artifacts...")

df_cat        = pd.read_csv("D:/FYP Material/Context Aware Diet Dataset/diet_data_categorical (3).csv")
df_num        = pd.read_csv("D:/FYP Material/Context Aware Diet Dataset/diet_data_numeric (3).csv").dropna(axis=1, how='all')
scaler        = joblib.load("D:/FYP Material/Context Aware Diet Dataset/CP Tensor/nutrient_scaler.save")
best_thresh   = joblib.load("D:/FYP Material/Context Aware Diet Dataset/CP Tensor/best_threshold.save")
goal_to_id    = joblib.load("D:/FYP Material/Context Aware Diet Dataset/CP Tensor/goal_to_id.save")
country_to_id = joblib.load("D:/FYP Material/Context Aware Diet Dataset/CP Tensor/country_to_id.save")
food_to_id    = joblib.load("D:/FYP Material/Context Aware Diet Dataset/CP Tensor/food_to_id.save")

# Rebuild reverse maps
id_to_goal    = {v: k for k, v in goal_to_id.items()}
time_to_id    = {"Breakfast": 1, "Lunch": 2, "Snack": 3, "Dinner": 4}
goal_display  = {"Weight Gain": "Muscle Gain", "Muscle Gain": "Muscle Gain",
                 "Weight Loss": "Weight Loss", "Maintenance": "Maintenance"}

N_GOALS = int(df_num['goal_id'].max())
N_TIMES = int(df_num['time_id'].max())

def encode_context(country_id, goal_id, time_id):
    return (int(country_id)-1)*(N_GOALS*N_TIMES) + (int(goal_id)-1)*N_TIMES + (int(time_id)-1)

df_num['context_id'] = df_num.apply(
    lambda r: encode_context(r['country_id'], r['goal_id'], r['time_id']), axis=1)

N_USERS    = int(df_num['User_id'].max())
N_FOODS    = int(df_num['food_id'].max())
N_CONTEXTS = int(df_num['context_id'].max())
N_NUT      = 7

model = CPDietModel(N_USERS, N_FOODS, N_CONTEXTS, N_NUT, rank=64)
model.load_state_dict(torch.load("cp_diet_model.pth", map_location="cpu"))
model.eval()
print("Model loaded ✅")

# ── 3. Scoring helpers ────────────────────────────────────────────────────────
NUT_COLS         = ['calories','protein_g','fat_g','carbs_g','protein_ratio','carb_ratio','fat_ratio']
MEAL_CAL_RATIOS  = {"Breakfast":0.25,"Lunch":0.35,"Snack":0.10,"Dinner":0.30}
MEAL_PROT_RATIOS = {"Breakfast":0.25,"Lunch":0.35,"Snack":0.05,"Dinner":0.35}
CAL_WEIGHTS      = [0.38, 0.35, 0.27]

SAMPLE_UIDS = df_num['User_id'].drop_duplicates().sample(
    min(10, df_num['User_id'].nunique()), random_state=42).tolist()

def score_food(food_row, country_id, goal_id, time_id):
    f_id  = food_to_id[food_row['food_name']]
    ctx   = encode_context(country_id, goal_id, time_id)
    p_rat = food_row['protein_g'] / (food_row['calories'] + 1)
    c_rat = food_row['carbs_g']   / (food_row['calories'] + 1)
    f_rat = food_row['fat_g']     / (food_row['calories'] + 1)
    nut   = torch.FloatTensor(scaler.transform(
        [[food_row['calories'], food_row['protein_g'], food_row['fat_g'],
          food_row['carbs_g'],  p_rat, c_rat, f_rat]]))
    scores = []
    with torch.no_grad():
        for uid in SAMPLE_UIDS:
            s = model(torch.LongTensor([uid]), torch.LongTensor([f_id]),
                      torch.LongTensor([ctx]), nut).item()
            scores.append(s)
    return sum(scores) / len(scores)

def calc_option(items, cal_target, prot_target):
    if len(items) < 3: return [], 0, 0
    result, tp, tc = [], 0, 0
    for idx, item in enumerate(items[:3]):
        gcal  = (cal_target  * CAL_WEIGHTS[idx] / item['cal_100'])  * 100
        gprot = (prot_target * CAL_WEIGHTS[idx] / item['prot_100']) * 100 \
                if item['prot_100'] > 0 else gcal
        g  = max(min(round((gcal + gprot) / 2), 400), 30)
        pr = round((item['prot_100'] * g) / 100)
        cl = round((item['cal_100']  * g) / 100)
        tp += pr; tc += cl
        result.append({"food": item['name'], "grams": g,
                        "protein": pr, "calories": cl, "score": round(item['score'], 3)})
    return result, tp, tc

# ── 4. Main diet plan function ────────────────────────────────────────────────
def generate_plan(weight, height, age, gender, goal, country):
    if gender.lower() == 'male':
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161

    dataset_goal = goal_display.get(goal, goal)
    if goal in ("Weight Gain", "Muscle Gain"):
        daily_cal, prot_factor = bmr + 500, 2.0
    elif goal == "Weight Loss":
        daily_cal, prot_factor = bmr - 500, 1.5
    else:
        daily_cal, prot_factor = bmr, 1.0

    total_protein = weight * prot_factor
    country_id    = country_to_id[country]
    goal_id       = goal_to_id[dataset_goal]
    candidates    = df_cat[df_cat['country'] == country].drop_duplicates('food_name')

    plan = {}
    for meal in ["Breakfast", "Lunch", "Snack", "Dinner"]:
        t_id   = time_to_id[meal]
        calt   = daily_cal    * MEAL_CAL_RATIOS[meal]
        prot_t = total_protein * MEAL_PROT_RATIOS[meal]
        foods  = candidates[candidates['time_of_day'] == meal]

        scored = []
        for _, row in foods.iterrows():
            scored.append({'name': row['food_name'], 'score': score_food(row, country_id, goal_id, t_id),
                           'cal_100': row['calories'], 'prot_100': row['protein_g']})

        high = sorted([x for x in scored if x['prot_100'] >= 15], key=lambda x: x['score'], reverse=True)
        med  = sorted([x for x in scored if  5 <= x['prot_100'] < 15], key=lambda x: x['score'], reverse=True)
        low  = sorted([x for x in scored if x['prot_100'] <  5], key=lambda x: x['score'], reverse=True)
        all_ = sorted(scored, key=lambda x: x['score'], reverse=True)

        def build_pool(exclude):
            pool, used = [], set(exclude)
            for bucket in [high, med, low, all_]:
                for item in bucket:
                    if item['name'] not in used:
                        pool.append(item); used.add(item['name']); break
                if len(pool) == 3: break
            for item in all_:
                if len(pool) >= 3: break
                if item['name'] not in {i['name'] for i in pool}: pool.append(item)
            return pool

        pool1 = build_pool(set())
        pool2 = build_pool({i['name'] for i in pool1})

        o1, o1p, o1c = calc_option(pool1[:3], calt, prot_t)
        o2, o2p, o2c = calc_option(pool2[:3], calt, prot_t)

        plan[meal] = {
            "target_cal": round(calt), "target_prot": round(prot_t),
            "option1": {"items": o1, "total_protein": o1p, "total_calories": o1c},
            "option2": {"items": o2, "total_protein": o2p, "total_calories": o2c},
        }

    return {
        "plan": plan,
        "daily_calories": round(daily_cal),
        "protein_target": round(total_protein),
        "bmr": round(bmr),
        "goal": goal,
        "country": country,
    }

# ── 5. Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route("/api/countries")
def get_countries():
    return jsonify(sorted(df_cat['country'].unique().tolist()))

@app.route("/api/recommend", methods=["POST"])
def recommend():
    try:
        data    = request.json
        weight  = float(data['weight'])
        height  = float(data['height'])
        age     = int(data['age'])
        gender  = data['gender']
        goal    = data['goal']
        country = data['country']

        if country not in country_to_id:
            return jsonify({"error": f"Country '{country}' not in dataset."}), 400
        if goal not in goal_display:
            return jsonify({"error": f"Invalid goal '{goal}'."}), 400

        result = generate_plan(weight, height, age, gender, goal, country)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Diet Recommender running at:")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
