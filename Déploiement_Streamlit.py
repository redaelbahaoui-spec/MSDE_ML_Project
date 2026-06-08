import streamlit as st
import pandas as pd
import joblib
import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

col1, col2 = st.columns([1, 5])

with col1:
    st.image(
        "pngegg.png",
        width=150
    )

with col2:
    st.title("Projet Machine Learning")
    st.subheader("Prédiction du prix des billets d'avion")

st.caption("Bienvenue dans ce projet de prédiction du prix des billets d'avion basé sur un modèle de machine learning entraîné sur " \
"un dataset de vols. Vous pouvez soit entrer les paramètres d'un vol directement, soit charger un fichier CSV contenant plusieurs vols pour obtenir leurs prédictions de prix.")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    choice = st.radio(
        'Comment souhaitez-vous utiliser le modèle de prédiction ? ',
        [ 'Saisir les paramètres directement', 'Charger un fichier de données']
    ,    horizontal=True,index=None)

if choice == 'Saisir les paramètres directement':
    st.sidebar.title("Paramètres du vol")

    airline          = st.sidebar.selectbox('Compagnie aérienne', 
                           ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India'])
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
    source_city      = st.sidebar.selectbox('Ville de départ', cities)
    destinations = [city for city in cities if city != source_city]                
    destination_city = st.sidebar.selectbox('Ville d\'arrivée', destinations)
    departure_time   = st.sidebar.selectbox('Heure de départ', 
                           ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
    arrival_time     = st.sidebar.selectbox('Heure d\'arrivée', 
                           ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
    stops            = st.sidebar.selectbox('Nombre d\'escales', ['zero', 'one', 'two_or_more'])
    flight_class     = st.sidebar.selectbox('Classe', ['Economy', 'Business'])
    duration         = st.sidebar.number_input('Durée du vol (heures)', 
                           min_value=1.0, max_value=50.0, value=2.0, step=1.0)
    days_left        = st.sidebar.slider('Jours avant départ', 1, 49, 15)

    df_input = pd.DataFrame([{
        "airline"          : airline,
        "source_city"      : source_city,
        "departure_time"   : departure_time,
        "stops"            : stops,
        "arrival_time"     : arrival_time,
        "destination_city" : destination_city,
        "class"            : flight_class,
        "duration"         : duration,
        "days_left"        : days_left
    }])

    st.subheader("Paramètres saisis :")
    st.dataframe(df_input.T.rename(columns={0: 'Valeur'}))

    model = joblib.load('Billet_avion_pipeline.pkl')

    if st.button("🔍 Prédire le prix"):
        pred = np.expm1(model.predict(df_input))
        st.success(f"💰 Prix estimé : {pred[0]:,.0f} ₹")
elif choice == 'Charger un fichier de données':
    st.subheader("Charger un fichier CSV")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        model = joblib.load('Billet_avion_pipeline.pkl')
        if st.button("🔍 Prédire les prix pour tous les vols du fichier"):
            preds = np.expm1(model.predict(df))
            df['predicted_price'] = preds
            st.subheader("Résultats avec prédictions :")
            st.dataframe(df)   
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    





