import pandas as pd 
import numpy as np
import matplotlib as plt 
import seaborn as sns
import missingno as msno
from  sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

ventes = "vente.xlsx"
validations = "validations.xlsx"
df_ventes = pd.read_excel(ventes)
df_validations = pd.read_excel(validations)

ventes_agg = df_ventes.groupby(['Date', 'Station','Code Titre'], as_index=False)['Quantité'].sum()
#print(ventes_agg.head())
validation_agg = df_validations.groupby(['Date','Station','Code Titre'], as_index=False)['Nbr validations'].sum()
#print(validation_agg.head())

df_final = pd.merge(ventes_agg,validation_agg, how='outer',on=['Date','Station','Code Titre'])
df_final = df_final.fillna(0)
#print(df_final.head())

code_ticket = [3500,4100,2005]
df = df_final[df_final['Code Titre'].isin(code_ticket)]

#créer une copie du data frame
df = df.copy()
#print(df.head())

#Classification des tickets
Ticket_24 = [4100,2005]
Ticket_multiple = [3500]
def classify_tickets(code) :
    if code in Ticket_24 : 
        return "24H"
    else : 
        return "multi"

df["type_ticket"] = df["Code Titre"].apply(classify_tickets)

#précision de la période
ramadan_dates = [(pd.Timestamp('2022-04-02'), pd.Timestamp('2022-05-02')),
                 (pd.Timestamp('2023-03-23'), pd.Timestamp('2023-04-21')),
                 (pd.Timestamp('2024-03-11'), pd.Timestamp('2024-04-09')),
                 (pd.Timestamp('2025-02-28'), pd.Timestamp('2025-03-28'))]

aid_el_fitr_dates = [(pd.Timestamp('2022-05-02'), pd.Timestamp('2022-05-03')),
                     (pd.Timestamp('2023-04-21'), pd.Timestamp('2023-04-22')),
                     (pd.Timestamp('2024-04-10'), pd.Timestamp('2024-04-12')),
                     (pd.Timestamp('2025-03-29'), pd.Timestamp('2025-03-31'))]

aid_adha_dates = [(pd.Timestamp('2022-07-09'), pd.Timestamp('2022-07-10')),
                  (pd.Timestamp('2023-06-28'), pd.Timestamp('2023-06-29')),
                  (pd.Timestamp('2024-06-16'), pd.Timestamp('2024-06-18')),
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

df['Période'] = df['Date'].apply(get_periode)


#précision du jour
Jour_férié = [
    # --- 2022 ---
    "2022-01-01", "2022-01-12", "2022-05-01",
    "2022-05-02", "2022-05-03",
    "2022-07-05", "2022-07-09",
    "2022-07-30", "2022-08-07",
    "2022-10-09", "2022-11-01",

    # --- 2023 ---
    "2023-01-01", "2023-01-12",
    "2023-04-22", 
    "2023-05-01", 
    "2023-06-29", 
    "2023-07-05", "2023-07-19",
    "2023-07-28", "2023-09-27",
    "2023-11-01",

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
    
#Application de la fonction 
df["type_jour"] = df.apply(classify_days, axis=1)


# --- Colonne de stratification et split train/test stratifié ---
# Créer une clé de stratification: type_ticket + Période + type_jour
df['stratify_col'] = df['type_ticket'].astype(str) + "_" + df['Période'].astype(str) + "_" + df['type_jour'].astype(str)+"_"+df['Station'].astype(str)

# Split stratifié (20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['stratify_col'])

#print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

#calculer les features sur le train set 
# --- Calculs comportementaux historiques ---
# Moyenne de validations par (Station, type_ticket)
train_df['validations_moyenne_station_code'] = train_df.groupby(['Station', 'type_ticket'])['Nbr validations'].transform('mean')

# Moyenne de validations par type_ticket
train_df['validations_par_ticket_historiques'] = train_df.groupby('type_ticket')['Nbr validations'].transform('mean')

# Quantité moyenne historique par (Station, type_ticket)
train_df['quantite_moyenne_station_code'] = train_df.groupby(['Station', 'type_ticket'])['Quantité'].transform('mean')

# Ratio historique: (moyenne validations) / (moyenne quantité) — éviter division par zéro
train_df['ratio_moyenne_historique'] = train_df['validations_moyenne_station_code'] / train_df['quantite_moyenne_station_code'].replace(0, np.nan)

# Ratio observé pour chaque ligne: Nbr validations / Quantité
train_df['ratio_observe'] = train_df.apply(lambda r: (r['Nbr validations'] / r['Quantité']) if r['Quantité'] > 0 else np.nan, axis=1)

# Aperçu rapide pour vérification
"""print(train_df[['Date','Station','Code Titre','Quantité','Nbr validations',
          'validations_moyenne_station_code','validations_par_ticket_historiques',
          'quantite_moyenne_station_code','ratio_moyenne_historique','ratio_observe']].head())"""


# --- Variables temporelles: lags et moyenne mobile ---
# S'assurer que la date est en datetime et trier
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df = train_df.sort_values(['Station', 'type_ticket', 'Date'])

# Groupby pour opérations temporelles
grp = train_df.groupby(['Station', 'type_ticket'])
# Valeur de validations à j-1 , j-7 et j-30 (décalage en lignes pour la même série)
train_df['validations_j-1'] = grp['Nbr validations'].shift(1)
train_df['validations_j-7'] = grp['Nbr validations'].shift(7)
train_df['validations_j-30'] = grp['Nbr validations'].shift(30)

# Moyenne mobile 30 jours (exclut la valeur courante pour éviter fuite d'information)
train_df['moyenne_mobile_30j'] = grp['Nbr validations'].transform(lambda s: s.shift(1).rolling(window=30, min_periods=1).mean())

# Afficher un extrait pour contrôle
#print(train_df[['Date','Station','type_ticket','Nbr validations','validations_j-1','validations_j-7','validations_j-30','moyenne_mobile_30j']].head(10))

#Nettoyage du nouveau dataset 
lag_cols = [
    'validations_j-1',
    'validations_j-7',
    'validations_j-30',
    'moyenne_mobile_30j'
]
train_df[lag_cols]= train_df[lag_cols].fillna(0)
ratio_cols = ['ratio_moyenne_historique','ratio_observe']
train_df[ratio_cols]= train_df[ratio_cols].fillna ( train_df[ratio_cols].median())

#Définition des variables expliquée et explicatives 
#train set
Y_train = train_df['Nbr validations']
X_train = train_df.drop(columns=[
    'Nbr validations',
    'Date',
    'stratify_col'
])

#encodage des variables catégorielles
cat_col = ['type_ticket',
            'Station', 
            'Période', 
            'type_jour'
            ]
encoder = OneHotEncoder (sparse_output= False, handle_unknown= "ignore")
#fit and tranform
X_train_cat = encoder.fit_transform(X_train[cat_col])
#convertir en dataframe 
X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(cat_col),index=X_train.index)
X_train_final = pd.concat([X_train_cat_df,X_train['Quantité']],axis=1)

#test set
Y_test= test_df['Nbr validations']
X_test = test_df.drop(columns=[
    'Nbr validations',
    'Date',
    'stratify_col'
])
X_test_cat = encoder.transform(X_test[cat_col])
X_test_cat_df = pd.DataFrame(X_test_cat,columns=encoder.get_feature_names_out(cat_col),index=X_test.index)
X_test_final = pd.concat([X_test_cat_df,X_test['Quantité']],axis=1)

#Entrainement du modèle LRM
LRM = LinearRegression()
LRM.fit(X_train_final,Y_train)
#predictions
Y_pred_lrm = LRM.predict(X_test_final)
#Evaluation du modèle
MAE_LRM =  mean_absolute_error(Y_test,Y_pred_lrm)
RMSE_LRM = np.sqrt(mean_squared_error(Y_test,Y_pred_lrm))
R2_LRM = r2_score(Y_test,Y_pred_lrm)

print("MAE_lrm = ", MAE_LRM)
print("RMSE_lrm = ", RMSE_LRM )
print("R2_lrm = ", R2_LRM)

#Entrainement du modèle RF
rf = RandomForestRegressor(
    n_estimators=400, #defines the number of decision trees 
    max_depth=25,
    min_samples_leaf=5,
    random_state=42, #ensures the randomness in model training 
    n_jobs=-1
)
rf.fit(X_train_final,Y_train)
#Predictions
Y_pred_RF = rf.predict(X_test_final)

#evaluation du modèle
MAE_RF =  mean_absolute_error(Y_test,Y_pred_RF)
RMSE_RF = np.sqrt(mean_squared_error(Y_test,Y_pred_RF))
R2_RF = r2_score(Y_test,Y_pred_RF)

print("MAE_rf = ", MAE_RF)
print("RMSE_rf = ", RMSE_RF )
print("R2_rf = ", R2_RF)

importance_rf = pd.Series(
    rf.feature_importances_,
    index= X_train_final.columns
).sort_values(ascending=False)

print(importance_rf.head())

#Entrainment du modèle Gradient boosting
gbr= GradientBoostingRegressor(
    n_estimators=400, #nombre d'arbres
    learning_rate=0.05, #taux d'apprentissage (plus petit = plus stable)
    max_depth= 5,       #profondeur des arbres
    min_samples_leaf=5, #éviter le surapprentissage
    subsample=0.8,      #stochastic gradient boosting
    random_state=42
)

gbr.fit(X_train_final, Y_train)

Y_pred_gbr = gbr.predict(X_test_final)

MAE_GBR =  mean_absolute_error(Y_test,Y_pred_gbr)
RMSE_GBR = np.sqrt(mean_squared_error(Y_test,Y_pred_gbr))
R2_GBR = r2_score(Y_test,Y_pred_gbr)

print("MAE_gbr = ", MAE_GBR)
print("RMSE_gbr = ", RMSE_GBR)
print("R2_gbr = ", R2_GBR)

importance_gb = pd.Series(
    gbr.feature_importances_,
    index=X_train_final.columns
).sort_values(ascending=False)

print(importance_gb.head())

#resultats et comparaison
résultats = pd.DataFrame({
    "Modéle" : ['Regressions', 'random forest', 'gradient boosting'],
    "MAE" : [MAE_LRM,MAE_RF,MAE_GBR],
    "RMSE" : [RMSE_LRM,RMSE_RF,RMSE_GBR],
    "R2" : [R2_LRM,R2_RF,R2_GBR]
})
print (résultats)
print("R2 train :", gbr.score(X_train_final, Y_train))
print("R2 test  :", gbr.score(X_test_final, Y_test))

#les résultats démontrent que le modéle gradient boosting est plus adapté
#prévision
#réentrainer sur 100% des données de 2022-2023
#calculet les profils historique (2022 et 2023)
profils = df.groupby(['Station','type_ticket']).agg(
    validations_moyenne = ('Nbr validations','mean'),
    quantite_moyenne = ('Quantité','mean')
    ).reset_index()
profils['ratio_hist'] = profils['validations_moyenne']/profils['quantite_moyenne']
#les ajouter au data set 2022 et 2023
df = df.merge(
    profils[['Station','type_ticket','validations_moyenne','ratio_hist']],
    on=['Station','type_ticket'],
    how='left'
)
df= pd.get_dummies(df,columns=cat_col,drop_first=True)

X = df.drop(columns=[
            "Nbr validations",
            "Date",
            "stratify_col"
            ])
Y = df["Nbr validations"]

gbr_final= GradientBoostingRegressor(
    n_estimators=400, #nombre d'arbres
    learning_rate=0.05, #taux d'apprentissage (plus petit = plus stable)
    max_depth= 5,       #profondeur des arbres
    min_samples_leaf=5, #éviter le surapprentissage
    subsample=0.8,      #stochastic gradient boosting
    random_state=42
)

gbr_final.fit(X,Y)

#refaire un split pour contrôler 
X_tr, X_te, Y_tr, Y_te = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

gbr_test = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
gbr_test.fit(X_tr, Y_tr)
print("R2 test:", gbr_test.score(X_te, Y_te))

#sauvegarder le modéle 
joblib.dump(gbr_final, "Modéle_ML_Validations_GBR.pkl")
#sauvegarder la structure des colonnes 
joblib.dump(X.columns.tolist(), "colonnes_modèle_GBR.pkl")
#sauvgarder les profils historique
joblib.dump(profils, "Profils_historiques.pkl")