import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

ventes_2425 = "Ventes_2024_2025.xlsx"
df_2425 = pd.read_excel(ventes_2425)

df_2425= df_2425.fillna(0)
code_ticket = [3500,4100,2005]
df_2425_filtré = df_2425[df_2425["Code Titre"].isin(code_ticket)]
#classification des tickets
Ticket_24 = [4100,2005]
Ticket_multiple = [3500]
def classify_tickets(code) :
    if code in Ticket_24 : 
        return "24H"
    else : 
        return "multi"

df_2425_filtré["type_ticket"] = df_2425_filtré["Code Titre"].apply(classify_tickets)

#précision de la période
ramadan_dates = [(pd.Timestamp('2024-03-11'), pd.Timestamp('2024-04-09')),
                 (pd.Timestamp('2025-02-28'), pd.Timestamp('2025-03-28'))]

aid_el_fitr_dates = [(pd.Timestamp('2024-04-10'), pd.Timestamp('2024-04-12')),
                     (pd.Timestamp('2025-03-29'), pd.Timestamp('2025-03-31'))]

aid_adha_dates = [(pd.Timestamp('2024-06-16'), pd.Timestamp('2024-06-18')),
                  (pd.Timestamp('2025-06-06'), pd.Timestamp('2025-06-08'))]

def get_periode(date):
    for start, end in ramadan_dates:
        if start <= date <= end:
            return 'ramadan'
    for start, end in aid_el_fitr_dates:
        if start <= date <= end:
            return 'aid_el_fitr'
    for start, end in aid_adha_dates:
        if start <= date <= end:
            return 'aid_adha'
    return 'standard'
df_2425_filtré["Période"]= df_2425_filtré['Date'].apply(get_periode)

#précision du jour
Jour_férié = [
    # --- 2024 --- (Dates réelles)
    "2024-01-01", "2024-01-12",
    "2024-04-10", "2024-04-11", "2024-04-12", # Aïd el-Fitr (3 jours)
    "2024-05-01",
    "2024-06-16", "2024-06-17", "2024-06-18", # Aïd el-Adha (3 jours)
    "2024-07-05", 
    "2024-07-07", # Awal Moharrem (Nouvel an Hégire)
    "2024-07-16", # Achoura
    "2024-09-15", # Mawlid en-Nabaoui
    "2024-11-01",

    # --- 2025 --- (Dates prévisionnelles / estimations lunaires)
    "2025-01-01", "2025-01-12",
    "2025-03-31", "2025-04-01", "2025-04-02", # Aïd el-Fitr (3 jours est.)
    "2025-05-01",
    "2025-06-07", "2025-06-08", "2025-06-09", # Aïd el-Adha (3 jours est.)
    "2025-06-26", # Awal Moharrem (est.)
    "2025-07-05", # Fête de l'Indépendance
    "2025-07-06", # Achoura (est.)
    "2025-09-04", # Mawlid en-Nabaoui (est.)
    "2025-11-01"  # Révolution
]

Jour_férié = pd.to_datetime(Jour_férié)

#fonction pour classifier les jours 
def classify_days (row) : 
    date = row["Date"]
    nom_jour = date.strftime("%A")  # Nom du jour en anglais

    if date in Jour_férié : 
        return "JF"
    elif nom_jour == "Friday" : 
        return "JV"
    else : 
        return "JO"
    
df_2425_filtré["type_jour"] = df_2425_filtré.apply(classify_days,axis=1)

gbr = joblib.load("Modéle_ML_Validations_GBR.pkl")
colonnes_modèle = joblib.load("colonnes_modèle_GBR.pkl")
profils = joblib.load("profils_historiques.pkl")

df_2425_filtré = df_2425_filtré.merge(profils[['Station','type_ticket','validations_moyenne','ratio_hist']],
    on=['Station','type_ticket'],
    how='left'
)
df_model = df_2425_filtré.copy()

cat_col = ['type_ticket',
            'Station', 
            'Période', 
            'type_jour'
            ]

df_model = pd.get_dummies(df_model, columns=cat_col,drop_first=True)

#Aligner les colonnes du modéle
X=df_model.reindex(columns=colonnes_modèle,fill_value=0)

df_2425_filtré['Nbr validation reconstruite'] = (
    np.clip(gbr.predict(X),0,None)
    .round()
    .astype(int)
)

df_2425_filtré.to_excel("df_2425_filtré_prédictions.xlsx", index=False)
