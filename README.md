# ⚡ H2 Optimizer AI - ML Explainable pour Électrolyseurs

**Optimiseur IA pour infrastructures Hydrogène territoriales (PEM vs AEM vs Alkaline).**  
XGBoost-like RandomForest + SHAP pour recommandations optimales selon ENR, CAPEX, demande locale. **Dashboard interactif Streamlit.**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF1493?logo=streamlit)](https://h2-optimizer.streamlit.app)
[![SHAP](https://img.shields.io/badge/SHAP-FF0000?logo=shap)](https://shap.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit-learn-F7931E?logo=scikit-learn)](https://scikit-learn.org)

## 🎯 Problème PhD Résolu
❓ **PEM cher/efficace vs AEM émergent/scalable pour Vendée (ENR 82%, infra moyenne)?**  
🤖 **ML Classifier** prédit optimal + **SHAP** explique (ENR >80% → AEM prioritaire).

## 🚀 Quickstart
```bash
git clone https://github.com/salimklibi/h2_optimizer_ai
cd h2_optimizer_ai
pip install -r requirements.txt  # streamlit pandas scikit-learn shap plotly

streamlit run h2_optimizer_ai.py
