from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
app_train = pd.read_csv("dataset/application_train.csv")
app_test = pd.read_csv("dataset/application_test.csv")
bureau = pd.read_csv("dataset/bureau.csv")
bureau_balance = pd.read_csv("dataset/bureau_balance.csv")
credit_card_balance = pd.read_csv("dataset/credit_card_balance.csv")
installments_payments = pd.read_csv("dataset/installments_payments.csv")
pos_cash_balance = pd.read_csv("dataset/POS_CASH_balance.csv")
previous_application = pd.read_csv("dataset/previous_application.csv")
sample_submission = pd.read_csv("dataset/sample_submission.csv")

# %% [markdown]
# ### Merging dataframe

# %% [markdown]
# Maintenant que nous aovns analyser les différents qui compose nos données nous allons aggréger ces données dans un seul et même dataframe

# %%
# Create a simple dataset with the train / test merge app
data = pd.concat([app_test, app_train], ignore_index=True)


# %%
# Check data
print('Train:' + str(app_test.shape))
print('Test:' + str(app_train.shape))
print('>>> Data:' + str(data.shape))

# %%
sum(data.SK_ID_CURR[data.TARGET.isna()] ==
    app_test.SK_ID_CURR)  # all is good

# %% [markdown]
#
# A partir du fichier bureau.csv, il est possible d'extraire un historique sur les précédents crédits enregistrés par les clients. Il peut donc être intéressant d'enrichir l'échantillon avec ce type de données.


# %%
# Total number of previous credits taken by each customer
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
    columns={'SK_ID_BUREAU': 'PREVIOUS_LOANS_COUNT'})
previous_loan_counts.head()

# %%
# Merge this new column in our data sample
data = data.merge(previous_loan_counts, on='SK_ID_CURR', how='left')
data.shape

# %% [markdown]
# bureau_balance : bureau_balance.csv


# %%
# Monthly average balances of previous credits in Credit Bureau.
bureau_bal_mean = bureau_balance.groupby('SK_ID_BUREAU', as_index=False)[
    'MONTHS_BALANCE'].mean().rename(columns={'MONTHS_BALANCE': 'MONTHS_BALANCE_MEAN'})
bureau_bal_mean.head()

# %%
bureau_full = bureau.merge(bureau_bal_mean, on='SK_ID_BUREAU', how='left')
bureau_full.drop('SK_ID_BUREAU', axis=1, inplace=True)

# %%
bureau_mean = bureau_full.select_dtypes(include=[np.number]).groupby(
    'SK_ID_CURR', as_index=False).mean().add_prefix('PREV_BUR_MEAN_')
bureau_mean = bureau_mean.rename(
    columns={'PREV_BUR_MEAN_SK_ID_CURR': 'SK_ID_CURR'})

# %%
# Merge all this features with our data sample
data = data.merge(bureau_mean, on='SK_ID_CURR', how='left')
data.shape

# %% [markdown]
# previous_application
#
# Vérification des valeurs de 'SK_ID_CURR' entre data et previous_application…


# %%
# Check
len(previous_application.SK_ID_CURR.isin(
    data.SK_ID_CURR)) == len(previous_application)

# %%
# Number of previous applications of the clients to Home Credit
previous_application_counts = previous_application.groupby('SK_ID_CURR',
                                                           as_index=False)['SK_ID_PREV'].count().rename(
    columns={'SK_ID_PREV': 'PREVIOUS_APPLICATION_COUNT'})
previous_application_counts.head()

# %%
# Merge this new column in our data sample
data = data.merge(previous_application_counts, on='SK_ID_CURR', how='left')
data.shape

# %% [markdown]
# credit_card_balance


# %%
credit_card_balance.drop('SK_ID_CURR', axis=1, inplace=True)

# %%
credit_card_balance_mean = credit_card_balance.groupby(
    'SK_ID_PREV', as_index=False).mean(numeric_only=True).add_prefix('CARD_MEAN_')
credit_card_balance_mean.rename(
    columns={'CARD_MEAN_SK_ID_PREV': 'SK_ID_PREV'}, inplace=True)

# %%
# Merge with previous_application
previous_application = previous_application.merge(
    credit_card_balance_mean, on='SK_ID_PREV', how='left')
previous_application.shape

# %% [markdown]
# installments_payments


# %%
installments_payments.drop('SK_ID_CURR', axis=1, inplace=True)

# %%
install_pay_mean = installments_payments.groupby(
    'SK_ID_PREV', as_index=False).mean().add_prefix('INSTALL_MEAN_')
install_pay_mean.rename(
    columns={'INSTALL_MEAN_SK_ID_PREV': 'SK_ID_PREV'}, inplace=True)
install_pay_mean.shape

# %%
# Merge with previous_application
previous_application = previous_application.merge(
    install_pay_mean, on='SK_ID_PREV', how='left')
previous_application.shape

# %% [markdown]
# POS_CASH_balance


# %%
pos_cash_balance.drop('SK_ID_CURR', axis=1, inplace=True)

# %%
POS_mean = installments_payments.groupby(
    'SK_ID_PREV', as_index=False).mean().add_prefix('POS_MEAN_')
POS_mean.rename(columns={'POS_MEAN_SK_ID_PREV': 'SK_ID_PREV'}, inplace=True)
POS_mean.shape

# %%
# Merge with previous_application
previous_application = previous_application.merge(
    POS_mean, on='SK_ID_PREV', how='left')
previous_application.shape

# %% [markdown]
# previous_application

# %% [markdown]
# Retour sur previous_application pour assembles les lignes d'observation selon 'SK_ID_CURR'.


# %%
prev_appl_mean = previous_application.groupby('SK_ID_CURR', as_index=False).mean(
    numeric_only=True).add_prefix('PREV_APPL_MEAN_')
prev_appl_mean.rename(
    columns={'PREV_APPL_MEAN_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)
prev_appl_mean = prev_appl_mean.drop('PREV_APPL_MEAN_SK_ID_PREV', axis=1)


# %%
# Reminder…
print('data shape', data.shape)

# %%
# Last merge with our data sample
data = data.merge(prev_appl_mean, on='SK_ID_CURR', how='left')
# data.set_index('SK_ID_CURR', inplace=True)


# %% [markdown]
# ## Feature enginering

# %% [markdown]
# Le "feature engineering" est le processus de transformation des données brutes en caractéristiques (features) utiles qui aident à améliorer la performance d'un modèle de machine learning. Les différents types incluent la création de variables (comme combiner des colonnes), la sélection de caractéristiques (choisir les variables les plus pertinentes), l'extraction de caractéristiques (comme l'analyse de composantes principales), et la transformation de variables (normalisation, standardisation, encodage de variables catégorielles).

# %% [markdown]
#
# **3 features extraites des précédentes étapes**
#
# Pour rappel, les étapes précédentes consistaient uniquement à établir des liens entre nos fichiers, des fusions de table dans le but d'enrichir l'échantillon de travail. Ceci étant, avant de procéder au merging des éléments, on a pu facilement extraire 3 variables de moyenne et de comptage.
# - PREVIOUS_LOANS_COUNT from bureau.csv: Nombre total des précédents crédits pris par chaque client
# - MONTHS_BALANCE_MEAN from bureau_balance.csv: Solde moyen mensuel des précédents crédits
# - PREVIOUS_APPLICATION_COUNT from previous_application.csv: Nombre de demandes antérieures des clients au crédit immobilier<br>
#
# **Création de 4 nouvelles variables métiers**
#
# Sans être expert en crédit bancaire, on peut assez facilement apporter quelques ratios explicatifs. D'autant plus qu'une veille parallèle permet de mieux comprendre les enjeux attendus. Voyons ci-dessous quelles features est-il pertinent d'intégrer.
#
# - CREDIT_INCOME_PERCENT: Pourcentage du montant du crédit par rapport au revenu d'un client
# - ANNUITY_INCOME_PERCENT: Pourcentage de la rente de prêt par rapport au revenu d'un client
# - CREDIT_TERM: Durée du paiement en mois
# - DAYS_EMPLOYED_PERCENT: Pourcentage des jours employés par rapport à l'âge du client
#

# %% [markdown]
# ### Features domaines métier

# %% [markdown]
# Sans être expert en crédit bancaire, on peut assez facilement apporter quelques ratios explicatifs. D'autant plus qu'une veille parallèle permet de mieux comprendre les enjeux attendus. Voyons ci-dessous quelles features est-il pertinent d'intégrer.
#
# - CREDIT_INCOME_PERCENT: Pourcentage du montant du crédit par rapport au revenu d'un client
# - ANNUITY_INCOME_PERCENT: Pourcentage de la rente de prêt par rapport au revenu d'un client
# - CREDIT_TERM: Durée du paiement en mois
# - DAYS_EMPLOYED_PERCENT: Pourcentage des jours employés par rapport à l'âge du client

# %%
# Before…
data.shape

# %%
data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

# %%
# After…
data.shape

# %%
# New Variables from features engineering
features_engin = ['PREVIOUS_LOANS_COUNT', 'MONTHS_BALANCE_MEAN', 'PREVIOUS_APPLICATION_COUNT',
                  'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']


# %% [markdown]
# ### Preprocessing des données

# %% [markdown]
# Il est nécessaire de commencer par la mise en place des données d'entrainement / test. On peut procéder en rappel avec les jeux de données application_train/test.

# %%
data_train = data[data['SK_ID_CURR'].isin(app_train.SK_ID_CURR)]
data_test = data[data['SK_ID_CURR'].isin(app_test.SK_ID_CURR)]

data_test = data_test.drop('TARGET', axis=1)

# %%
data_train.set_index('SK_ID_CURR', inplace=True)
data_test.set_index('SK_ID_CURR', inplace=True)

# %%
print('Training Features shape with categorical columns: ', data_train.shape)
print('Testing Features shape with categorical columns: ', data_test.shape)

# %% [markdown]
# ### Encodage des variables catégorielle

# %% [markdown]
# Une modèle de machine learning n'est pas capable d'interpréter les variables catégorielle. Pour pouvoir les utiliser dans notre modèle on doit encoder ces variables en nombre.Il existe deux méthode pour transformer ces valeur, soit le _Label Encoding_ ou le _one-hot-encoding_.Pour les colonnes ayant seulement deux valeur unique nous utiliseront la méthode du Label Encoding et pour les autres le one-hot-encoding.

# %% [markdown]
# ### Label Encoding et One-Hot Encoding

# %%
# Create a label encoder object
le = LabelEncoder()
list_col_name = []
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            list_col_name.append(col)
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])

            # Keep track of how many columns were label encoded
            le_count += 1

print(f'{le_count} colonnes ont été encodées:{list_col_name}')

# %%
# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# %% [markdown]
# ### Alignement des datasets train et test

# %% [markdown]
# Le one-hot-encoding a créer plus de colonne dans le training data car certaines categories ne sont pas representer dans les données de test. Pour supprimer ces colonnes nous allons 'aligné' nos dataframes, tout en gardant la colonne 'TARGET' dans notre app_train.

# %%
train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join='inner', axis=1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# %%
# Copy before imputation of missing values
train = data_train.copy()
test = data_test.copy()
train.shape, test.shape

data_sample = app_train.sample(frac=0.1, random_state=42)

# Conversion des types pour économiser la mémoire
for col in data_sample.columns:
    if data_sample[col].dtype == np.float64:
        data_sample[col] = data_sample[col].astype(np.float32)
    if data_sample[col].dtype == np.int64:
        data_sample[col] = data_sample[col].astype(np.int32)

X = data_sample.drop('TARGET', axis=1)
y = data_sample['TARGET']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Supposons que 'app_train' est votre DataFrame initial
# Pour l'exemple, on continue avec data_sample déjà défini

# Création d'un pipeline pour la baseline avec SimpleImputer utilisant la médiane
baseline_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Imputation par la médiane
    ('model', LinearRegression())  # Modèle de régression linéaire
])

# Définition de la métrique de scoring comme MSE négatif
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Exécution de la validation croisée et conversion des scores MSE en positifs
cv_scores = cross_val_score(baseline_pipeline, X, y, cv=5 ,scoring=mse_scorer)
cv_scores_positive = -cv_scores

# Calcul de la moyenne des scores MSE positifs
mean_cv_score = np.mean(cv_scores_positive)

# Impression des résultats
print("MSE de chaque validation croisée :", cv_scores_positive)
print("Moyenne des scores MSE sur les validations croisées :", mean_cv_score)

# Création d'un DataFrame pour stocker les résultats
results_df = pd.DataFrame(cv_scores_positive, columns=['MSE'])
results_df.loc['Moyenne'] = mean_cv_score

# Enregistrement des résultats dans un fichier CSV
results_df.to_csv('cv_mse_results_median.csv', index_label='Validation Fold')
