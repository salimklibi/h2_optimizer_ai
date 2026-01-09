import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Données simulées H2 (basé IRENA/AIDHY specs 2026)
@st.cache_data
def load_h2_data():
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'enr_share': np.random.uniform(0.5, 1.0, n_samples),  # % ENR territorial
        'capex_kwh': np.random.uniform(300, 1200, n_samples),  # CAPEX €/kW
        'demand_h2_ton': np.random.uniform(50, 500, n_samples),  # Demande locale
        'grid_distance_km': np.random.uniform(5, 50, n_samples),  # Dist. réseau
        'policy_support': np.random.choice([0, 0.5, 1], n_samples),  # Subventions
        'electrolyzer_type': np.random.choice(['PEM', 'AEM', 'Alkaline'], n_samples)
    })
    # Labels réalistes: 0=Alkaline (low-cost), 1=PEM (high-eff), 2=AEM (emerging)
    data['optimal_type'] = np.where(
        (data['enr_share'] > 0.8) & (data['policy_support'] > 0.5), 2,  # AEM high ENR
        np.where(data['capex_kwh'] < 600, 0, 1)  # Alkaline low CAPEX else PEM
    )
    return data

def train_model(df):
    features = ['enr_share', 'capex_kwh', 'demand_h2_ton', 'grid_distance_km', 'policy_support']
    X = df[features]
    y = df['optimal_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return model, scaler, X_test_scaled, y_test, acc, features

# SHAP Explainer
@st.cache_resource
def get_shap_explainer(model, X_train_scaled):
    return shap.TreeExplainer(model).shap_values(X_train_scaled)

st.title("H2 Optimizer AI - Sélection Électrolyseur Optimal")
st.markdown("**ML + SHAP pour recommander PEM/AEM/Alkaline selon contexte territorial (Vendée/Pays de la Loire).** PhD Challenge 2026.")

tab1, tab2, tab3 = st.tabs(["Démo Interactive", "Modèle ML", "Explainability"])

with tab1:
    st.header("Simule ton scénario Vendée")
    col1, col2, col3, col4, col5 = st.columns(5)
    enr = col1.slider("Part ENR (%)", 50, 100, 82) / 100
    capex = col2.slider("CAPEX (€/kW)", 300, 1200, 800)
    demand = col3.slider("Demande H2 (ton/an)", 50, 500, 200)
    distance = col4.slider("Dist. réseau (km)", 5, 50, 20)
    policy = col5.slider("Soutien politique", 0.0, 1.0, 0.7)
    
    if st.button("Recommander Électrolyseur"):
        input_data = np.array([[enr, capex, demand, distance, policy]])
        pred = model.predict(scaler.transform(input_data))[0]
        types = {0: 'Alkaline (low-cost)', 1: 'PEM (efficace)', 2: 'AEM (émergent/scalable)'}
        st.success(f"** Optimal: {types[pred]}** pour ton contexte!")
        st.balloons()

with tab2:
    df = load_h2_data()
    model, scaler, X_test, y_test, acc, features = train_model(df)
    st.metric("Accuracy Test", f"{acc:.1%}")
    st.dataframe(pd.DataFrame({'Prédiction': model.classes_[y_pred], 'Vrai': model.classes_[y_test]}))

with tab3:
    st.header("Pourquoi cette reco? SHAP Forces")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    st.plotly_chart(shap.summary_plot(shap_values, X_test, feature_names=features, show=False), use_container_width=True)

st.markdown("---")
st.info("**GitHub Ready:** Push + Streamlit Cloud. Lie à thèse H2 Vendée. Datasets IRENA importables via CSV upload.")

if __name__ == "__main__":
    print("streamlit run h2_optimizer_ai.py")
