import streamlit as st
import pandas as pd
import pickle

st.header("Insurance Claim Prediction", text_alignment="center")


# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_data
def load_model():
    with open("decision_tree_model_1.pkl", "rb") as f:
        return pickle.load(f)
    
model = load_model()


# --------------------------------
# Load reference CLEAN dataset
# (to maintain exact column structure)
# --------------------------------

clean_cols = pd.read_csv("clean_dataset.csv").drop(['is_claim', 'Unnamed: 0', 'turning_radius'], axis=1).columns
required_cols = model.feature_names_in_


# --------------------------------
# USER INPUT (RAW like dataset.csv)
# --------------------------------

st.markdown("#### Enter Vehicle / Policy Details :")
left_col, right_col = st.columns(2)


# --------------------------------------------------
# NUMERIC INPUT COLUMN (LEFT)
# --------------------------------------------------

with left_col:
    df_temp = pd.read_csv("dataset.csv", nrows=5)  # small read to get columns
    policy_tenure = st.number_input("policy_tenure")
    age_of_car = st.number_input("age_of_car")
    age_of_policyholder = st.number_input("age_of_policyholder")

    height = st.selectbox("height", df_temp['height'].dropna().unique().tolist())
    displacement = st.selectbox("displacement", df_temp['displacement'].dropna().unique().tolist())
    cylinder = st.number_input("cylinder", df_temp['cylinder'].dropna().unique().tolist()[0])
    gear_box = st.number_input("gear_box", df_temp['gear_box'].dropna().unique().tolist()[0])
    population_density = st.selectbox("population_density (raw e.g 1_234)", df_temp['population_density'].dropna().unique().tolist())
    area_cluster = st.selectbox("area_cluster", df_temp['area_cluster'].dropna().unique().tolist())

    Length = st.selectbox("Length (e.g 1745(mm))", df_temp['Length'].dropna().unique().tolist())
    width = st.selectbox("width (raw format)", df_temp['width'].dropna().unique().tolist())
    Gross_weight = st.selectbox("Gross_weight (e.g 1400Kg)", df_temp['Gross_weight'].dropna().unique().tolist())

    max_torque = st.selectbox("max_torque (e.g 113Nm@4200rpm)", df_temp['max_torque'].dropna().unique().tolist())
    max_power = st.selectbox("max_power (e.g 81bhp@6000rpm)", df_temp['max_power'].dropna().unique().tolist())


# --------------------------------------------------
# CATEGORICAL / TEXT INPUT COLUMN (RIGHT)
# (includes torque, power, length etc. because they
# are raw text in dataset.csv)
# --------------------------------------------------

with right_col:

    # ---------- Binary Yes/No ----------
    binary_cols = [
        'is_esc','is_adjustable_steering','is_tpms','is_parking_sensors',
        'is_parking_camera','is_front_fog_lights','is_rear_window_wiper',
        'is_rear_window_washer','is_rear_window_defogger','is_brake_assist',
        'is_power_door_locks','is_central_locking','is_power_steering',
        'is_driver_seat_height_adjustable','is_day_night_rear_view_mirror',
        'is_ecw','is_speed_alert'
    ]

    binary_values = {}
    for c in binary_cols:
        binary_values[c] = st.selectbox(c, ["Yes", "No"])
    
    segment = st.selectbox("segment", ["A","B1","B2","C1","C2","Utility"])
    model_name = st.selectbox("model", [f"M{i}" for i in range(1,12)])
    fuel_type = st.selectbox("fuel_type", ["Petrol","Diesel","CNG"])
    rear_brakes_type = st.selectbox("rear_brakes_type", ["Drum","Disc"])
    transmission_type = st.selectbox("transmission_type", ["Manual","Automatic"])
    steering_type = st.selectbox("steering_type", ["Power","Manual","Electric"])
    engine_type = st.selectbox("engine_type", [
        "1.0 SCe","1.2 L K Series Engine","1.2 L K12N Dualjet","1.5 L U2 CRDi",
        "1.5 Turbocharged Revotorq","1.5 Turbocharged Revotron",
        "F8D Petrol Engine","G12B","K Series Dual jet","K10C","i-DTEC"
    ])



# -------------------------------
# BUILD RAW DF ROW
# -------------------------------

raw = {
    "policy_tenure": policy_tenure,
    "age_of_car": age_of_car,
    "age_of_policyholder": age_of_policyholder,
    "population_density": population_density,
    "area_cluster": area_cluster,
    "Length": Length,
    "width": width,
    "height": height,
    "Gross_weight": Gross_weight,
    "max_torque": max_torque,
    "max_power": max_power,
    "displacement": displacement,
    "cylinder": cylinder,
    "gear_box": gear_box,
    "segment": segment,
    "model": model_name,
    "fuel_type": fuel_type,
    "rear_brakes_type": rear_brakes_type,
    "transmission_type": transmission_type,
    "steering_type": steering_type,
    "engine_type": engine_type
}
raw.update(binary_values)
input_df = pd.DataFrame([raw])


# --------------------------------
# APPLY SAME CLEANING AS TRAINING
# --------------------------------

def safe(func, x):
    try:
        return func(x)
    except:
        return 0
    
def length(l):
    return int(l[l.index(')')+1:]) / 1000

def width_f(w):
    return int(w[: w.index('(')]) / 1000

def weight(w):
    return int(w[:w.index('K')]) / 1000

def torque_ratio(s):
    torque = float(s[:s.index('N')])
    rpm = float(s[s.index('@')+1:s.index('r')])
    return torque/rpm

def power_ratio(s):
    power = float(s[:s.index('b')])
    rpm = float(s[s.index('@')+1:s.index('r')])
    return power/rpm

input_df["Length"] = input_df["Length"].apply(lambda x: safe(length,x))
input_df["width"] = input_df["width"].apply(lambda x: safe(width_f,x))
input_df["height"] = input_df["height"]/1000
input_df["Gross_weight"] = input_df["Gross_weight"].apply(lambda x: safe(weight,x))
input_df["max_torque"] = input_df["max_torque"].apply(lambda x: safe(torque_ratio,x))
input_df["max_power"] = input_df["max_power"].apply(lambda x: safe(power_ratio,x))
input_df["population_density"] = (
    input_df["population_density"]
    .astype(str)
    .str.replace('"','')
    .str.replace('_','')
    .str.strip()
    .replace("",None)
    .astype(float)
)
from sklearn.preprocessing import LabelEncoder
input_df["area_cluster"] = LabelEncoder().fit_transform(input_df["area_cluster"])


# --------------------------------
# DUMMIES (must exactly match training)
# --------------------------------

# Convert Yes/No to 0/1
binary_cols = [
    'is_esc','is_adjustable_steering','is_tpms','is_parking_sensors',
    'is_parking_camera','is_front_fog_lights','is_rear_window_wiper',
    'is_rear_window_washer','is_rear_window_defogger','is_brake_assist',
    'is_power_door_locks','is_central_locking','is_power_steering',
    'is_driver_seat_height_adjustable','is_day_night_rear_view_mirror',
    'is_ecw','is_speed_alert'
]
input_df[binary_cols] = input_df[binary_cols].replace({"Yes":1, "No":0})
# One-hot encode categorical like training
cat_cols = [
    'segment','model','fuel_type','rear_brakes_type',
    'transmission_type','steering_type','engine_type'
]
input_encoded = pd.get_dummies(input_df, columns=cat_cols, dtype=int)
# Align to model's expected schema
input_encoded = input_encoded.reindex(columns=required_cols, fill_value=0)


# --------------------------------
# PREDICT
# --------------------------------

if st.button("Predict Claim"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.error(f" Claim Likely | Probability = {prob:.2f}")
    else:
        st.success(f" No Claim | Probability = {prob:.2f}")
    
# st.dataframe(input_encoded)
# st.write(input_encoded.shape)
# st.write(input_encoded.columns.tolist())