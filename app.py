
import base64
import streamlit as st
import pandas as pd
import pickle

st.markdown(
    """
    <style>
    /* Remove Streamlit default padding */
    .block-container {
        padding-top: 0.8rem;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        padding: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(     
    page_title="Insurance Claim Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


with st.sidebar:
    st.image(
        "assets/insurance_logo0.png",
        width=150
    )
    st.markdown("## Insurance Analytics")
    st.markdown("---")

    menu = st.radio(
        "Navigation",
        ["About", "Dashboard", "Model Info", "Claim Prediction" ]
    )

    st.markdown("---")
    st.caption("© 2026 Insurance")




if menu == "About":
    st.markdown('<a id="about"></a>', unsafe_allow_html=True)
    st.header("About")

    st.write("CarInsure : AI-Driven Insurance Claim Prediction and Risk Analytics Platform")

    st.markdown("""
    ## Domain Knowledge: Insurance Claim Prediction & Risk Analytics
### 1. Overview of the Insurance Domain

The insurance industry operates on the principle of risk pooling and risk assessment, where insurers evaluate the likelihood of loss events (claims) and price insurance products accordingly. In motor insurance, claims arise from factors such as vehicle characteristics, driver behavior, policy conditions, environmental exposure, and safety features.

Accurate risk assessment is critical for:

- Pricing premiums correctly

- Preventing adverse selection

- Reducing claim losses

- Maintaining profitability and regulatory compliance

### 2. Insurance Claims and Their Impact

An insurance claim represents a financial liability for the insurer. High claim frequency or severity directly impacts:

- Loss ratio

- Combined ratio

- Portfolio profitability

Traditional rule-based underwriting systems struggle to capture complex, nonlinear relationships among risk factors. This has driven the adoption of data-driven and machine learning–based approaches for claim prediction.

### 3. Role of Data in Claim Risk Assessment

Motor insurance risk is influenced by multiple data categories:

#### a. Policyholder Attributes

- Age of the policyholder

- Policy tenure

- Historical behavior (where available)

Younger or inexperienced drivers and short-tenure policyholders generally exhibit higher claim frequencies.

#### b. Vehicle Characteristics

- Age of vehicle

- Vehicle segment (entry-level, mid-range, premium)

- Engine type and fuel type

- Transmission type

Older vehicles and higher-performance vehicles tend to have higher claim probabilities due to wear, maintenance costs, and driving patterns.

#### c. Safety Features

- Electronic Stability Control (ESC)

- Anti-lock braking systems

- Parking sensors and cameras

- Brake assist systems

Vehicles equipped with advanced safety features generally demonstrate lower claim frequency and severity, making safety features a critical risk-mitigating factor.

#### d. Geographic and Environmental Factors

- Population density

- Area clusters (urban, semi-urban, rural)

Urban areas typically show higher claim frequency due to traffic congestion, while rural areas may show lower frequency but potentially higher severity.

### 4. Machine Learning in Insurance Risk Modeling

Machine learning enables insurers to:

- Identify complex patterns across multiple risk variables

- Capture nonlinear interactions between features

- Imp- rove prediction accuracy over traditional actuarial models

In this project:

- A Decision Tree classifier is used for its interpretability

- Feature preprocessing ensures alignment between training and inference

- Model outputs include both binary claim prediction and probability scores


### 5. Risk Analytics and Business Intelligence

Beyond prediction, insurers require explainable insights to understand why claims occur.

Risk analytics helps in:

- Identifying high-risk customer segments

- Monitoring portfolio-level claim trends

- Supporting underwriting and pricing decisions

- Evaluating effectiveness of safety features

Tableau dashboards serve as a decision-support layer, enabling stakeholders to explore claim patterns interactively without requiring technical expertise.

### 6. Integration of Predictive and Descriptive Analytics

This project demonstrates a modern insurance analytics architecture:
""")
    table_data = pd.DataFrame({
        "Layer": [
            "Machine Learning",
            "Streamlit Application",
            "Tableau Dashboards"
        ],
        "Purpose": [
            "Predict claim likelihood",
            "Real-time user interaction",
            "Portfolio monitoring & insights"
        ]
    })

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("""
This layered approach ensures:

- Operational efficiency (instant predictions)

- Strategic insight (trend analysis)

- Scalability for future enhancements

### 7. Business Relevance of the Project

The platform supports multiple insurance business functions:

- Underwriting: Risk-based decision-making

- Pricing: Premium differentiation by risk

- Risk Management: Portfolio monitoring

- Customer Strategy: Targeted interventions

By leveraging AI and analytics, insurers can transition from reactive claim handling to proactive risk management.

### 8. Conclusion

The domain knowledge underlying this project reflects real-world insurance practices, combining actuarial principles with modern machine learning and analytics techniques. The project aligns with industry trends toward data-driven underwriting, explainable AI, and interactive business intelligence, making it both technically sound and business-relevant.
    """)

if menu == "Dashboard":
    st.markdown('<a id="dashboard"></a>', unsafe_allow_html=True)
    st.header("📊 Dashboard Overview")

    st.write("""
    The dashboard provides insights into insurance claim patterns,
    risk factors, and model performance metrics.
    """)
    st.components.v1.iframe(
        src="https://public.tableau.com/views/Car_claim_17684158841960/Car_claim?:embed=yes&:showVizHome=no&:tabs=no&:toolbar=yes",
        width=1200,
        height=800,
        scrolling=True
    )


if menu == "Model Info":
    st.markdown('<a id="model"></a>', unsafe_allow_html=True)
    
    # def show_pdf(pdf_path, height=700):
    #     with open(pdf_path, "rb") as f:
    #         pdf_bytes = f.read()

    #     base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    #     pdf_html = f"""
    #         <iframe
    #             src="data:application/pdf;base64,{base64_pdf}"
    #             width="100%"
    #             height="{height}px"
    #             style="border: none;"
    #         ></iframe>
    #     """

    #     st.markdown(pdf_html, unsafe_allow_html=True)


    # st.header("Problem Statement and Information")
    # show_pdf("Project Problem Statement.pdf")

    st.header("📊 Model Overview")

    st.write("""
    - **Algorithm:** Decision Tree Classifier  
    - **Objective:** Predict likelihood of insurance claim  
    - **Input:** Vehicle, policy, and safety features  
    - **Output:** Binary decision with probability score
    """)

    # c1, c2 = st.columns(2)
    # with c1:
    #     st.subheader("Feature Importance")
    #     st.image("assets/graph3.png", width=600)
    # with c2:
    #     st.subheader("Model Performance")
    #     st.image("assets/graph.png", width=600)
    #     st.image("assets/graph2.png", width=600)




if menu == "Claim Prediction":
    # st.header("Insurance Claim Prediction", text_alignment="center")
    st.markdown(
        """
        <div style="background-color:#0f2c44;padding:20px;border-radius:10px">
            <h1 style="color:white;text-align:center;">
                🚗 CarSecure
            </h1>
            <p style="color:#d3d3d3;text-align:center;font-size:16px;">
                AI-Driven Risk Assessment for Vehicle Insurance Policies
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<a id="prediction"></a>', unsafe_allow_html=True)

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

    st.markdown("#### Enter Vehicle / Policy Details")
    st.markdown("<hr>", unsafe_allow_html=True)

    df_temp = pd.read_csv("dataset.csv", nrows=5)

    # ================================
    # NUMERICAL / POLICY FEATURES
    # ================================
    st.markdown("##### Numerical & Policy Features")
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        policy_tenure = st.number_input("policy_tenure")
    with c2:
        age_of_car = st.number_input("age_of_car")
    with c3:
        age_of_policyholder = st.number_input("age_of_policyholder")

    c1, c2, c3 = st.columns(3)
    with c1:
        population_density = st.selectbox(
            "population_density",
            df_temp['population_density'].dropna().unique().tolist()
        )
    with c2:
        area_cluster = st.selectbox(
            "area_cluster",
            df_temp['area_cluster'].dropna().unique().tolist()
        )
    with c3:
        height = st.selectbox(
            "height",
            df_temp['height'].dropna().unique().tolist()
        )
    st.markdown("<br>", unsafe_allow_html=True)
    # ================================
    # VEHICLE SPECIFICATIONS
    # ================================
    st.markdown("##### Vehicle Specifications")
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        Length = st.selectbox(
            "Length",
            df_temp['Length'].dropna().unique().tolist()
        )
    with c2:
        width = st.selectbox(
            "width",
            df_temp['width'].dropna().unique().tolist()
        )
    with c3:
        Gross_weight = st.selectbox(
            "Gross_weight",
            df_temp['Gross_weight'].dropna().unique().tolist()
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        displacement = st.number_input(
            "displacement",
            value=796
        )
    with c2:
        cylinder = st.number_input(
            "cylinder",
            df_temp['cylinder'].dropna().unique().tolist()[0]
        )
    with c3:
        gear_box = st.number_input(
            "gear_box",
            df_temp['gear_box'].dropna().unique().tolist()[0]
        )
    st.markdown("<br>", unsafe_allow_html=True)
    # ================================
    # PERFORMANCE METRICS
    # ================================
    st.markdown("##### Performance Metrics")
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        max_torque = st.selectbox(
            "max_torque",
            df_temp['max_torque'].dropna().unique().tolist()
        )
    with c2:
        max_power = st.selectbox(
            "max_power",
            df_temp['max_power'].dropna().unique().tolist()
        )
    with c3:
        pass

    st.markdown("<br>", unsafe_allow_html=True)
    # ================================
    # CATEGORICAL FEATURES
    # ================================
    st.markdown("##### Configuration Details")
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        segment = st.selectbox("segment", ["A","B1","B2","C1","C2","Utility"])
    with c2:
        model_name = st.selectbox("model", [f"M{i}" for i in range(1,12)])
    with c3:
        fuel_type = st.selectbox("fuel_type", ["Petrol","Diesel","CNG"])

    c1, c2, c3 = st.columns(3)
    with c1:
        rear_brakes_type = st.selectbox("rear_brakes_type", ["Drum","Disc"])
    with c2:
        transmission_type = st.selectbox("transmission_type", ["Manual","Automatic"])
    with c3:
        steering_type = st.selectbox("steering_type", ["Power","Manual","Electric"])

    c1, c2, c3 = st.columns(3)
    with c1:
        engine_type = st.selectbox(
            "engine_type",
            [
                "1.0 SCe","1.2 L K Series Engine","1.2 L K12N Dualjet",
                "1.5 L U2 CRDi","1.5 Turbocharged Revotorq",
                "1.5 Turbocharged Revotron","F8D Petrol Engine",
                "G12B","K Series Dual jet","K10C","i-DTEC"
            ]
        )
    with c2:
        pass
    with c3:
        pass
    
    st.markdown("<br>", unsafe_allow_html=True)
    # ================================
    # BINARY SAFETY FEATURES
    # ================================
    st.markdown("##### Safety & Assist Features")
    st.markdown("<hr>", unsafe_allow_html=True)

    binary_cols = [
        'is_esc','is_adjustable_steering','is_tpms','is_parking_sensors',
        'is_parking_camera','is_front_fog_lights','is_rear_window_wiper',
        'is_rear_window_washer','is_rear_window_defogger','is_brake_assist',
        'is_power_door_locks','is_central_locking','is_power_steering',
        'is_driver_seat_height_adjustable','is_day_night_rear_view_mirror',
        'is_ecw','is_speed_alert'
    ]

    binary_values = {}
    for i in range(0, len(binary_cols), 3):
        c1, c2, c3 = st.columns(3)
        for col, feature in zip([c1, c2, c3], binary_cols[i:i+3]):
            with col:
                binary_values[feature] = st.selectbox(
                    feature,
                    ["Yes", "No"]
                )


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
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        predict_btn = st.button("🔍 Predict Claim Risk", use_container_width=True)

    if predict_btn:
        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)

        if pred == 1:
            st.markdown(
                f"""
                <div style="background-color:#ffe6e6;padding:20px;border-radius:10px">
                    <h3 style="color:#b30000;">⚠ Claim Likely</h3>
                    <p><b>Risk Probability:</b> {prob:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#e6ffed;padding:20px;border-radius:10px">
                    <h3 style="color:#006622;">✅ No Claim Expected</h3>
                    <p><b>Risk Probability:</b> {prob:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # st.dataframe(input_encoded)
    # st.write(input_encoded.shape)
    # st.write(input_encoded.columns.tolist())



st.markdown(
    """
    <div class="footer">
        <hr>
        © 2026 CarSecure Insurance • Machine Learning Risk Platform
        <br>
        <a href="https://www.linkedin.com/in/pratham-raikar-7b3921234/">LinkedIn</a> • 
        <a href="https://github.com/Prathamr17">GitHub</a> • 
        <a href="mailto:xelortop1@gmail.com?subject=Insurance%20Claim%20Query&body=Hello%2C%20I%20would%20like%20to%20know%20more.">
            Email
        </a>

    </div>
    """,
    unsafe_allow_html=True
)


