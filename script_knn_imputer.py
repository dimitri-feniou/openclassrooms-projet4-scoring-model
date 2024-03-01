# %% [markdown]
# **Table of contents**<a id='toc0_'></a>
# - [Projet 4 Openclassrooms : Construisez un modèle de scoring](#toc1_)
#   - [Présentation du projet](#toc1_1_)
#
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %% [markdown]
# # <a id='toc1_'></a>[Projet 4 Openclassrooms : Construisez un modèle de scoring](#toc0_)

# %% [markdown]
# ![Description de l'image](https://user.oc-static.com/upload/2023/03/24/16796540347308_Data%20Scientist-P7-01-banner.png)

# %% [markdown]
# ## <a id='toc1_1_'></a>[Présentation du projet](#toc0_)

# %% [markdown]
# Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser",  qui propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt.<br>
# Elle souhaite donc développer un **algorithme de classification** pour aider à décider si un prêt peut être accordé à un client.

# %% [markdown]
# ## Installation des librairies

# %%
# Import library
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas as pd
import os
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import missingno as msno
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# %%
# Create Dataframes with CSV from dataset

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

# %% [markdown]
# ### Imputation des valeurs manquantes

# %%
# Reduce the size of the data and convert types to save memory
data_sample = app_train.sample(frac=0.1, random_state=42)
for col in data_sample.columns:
    if data_sample[col].dtype == np.float64:
        data_sample[col] = data_sample[col].astype(np.float32)
    if data_sample[col].dtype == np.int64:
        data_sample[col] = data_sample[col].astype(np.int32)

# Display the size of the sample
print(f"Sample size: {data_sample.shape}")

X = data_sample.drop('TARGET', axis=1)
y = data_sample['TARGET']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Pipeline with KNNImputer and LinearRegression
pipeline = Pipeline([
    # Initial value for n_neighbors; will be adjusted by GridSearch
    ('imputer', KNNImputer(n_neighbors=5)),
    ('model', LinearRegression())
])

# Parameters for GridSearchCV
param_grid = {
    # Extending the search range up to 15
    'imputer__n_neighbors': list(range(3, 16))
}

# Setting up GridSearchCV and executing the Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='neg_mean_squared_error', verbose=3)
grid_search.fit(X_train, y_train)

# Best parameters found
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Evaluation on the test set with the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
test_score = mean_squared_error(y_test, predictions)
print(f"MSE on the test set: {test_score}")

# Accessing cross-validation results for each parameter configuration
cv_results = grid_search.cv_results_

# Preparing the DataFrame to store the results
results_df = pd.DataFrame(cv_results["params"])
for i in range(5):  # Assuming cv=5 as specified in GridSearchCV
    results_df[f'MSE Fold {i+1}'] = -cv_results[f'split{i}_test_score']
results_df['MSE Mean'] = -cv_results['mean_test_score']

# Adding the best parameter and the MSE on the test set to the DataFrame
results_df['Best Parameter'] = results_df['imputer__n_neighbors'] == best_params['imputer__n_neighbors']
results_df.loc['Mean / Total'] = results_df.mean(numeric_only=True)
results_df.at['Mean / Total', 'Best Parameter'] = test_score

# Saving the results to a CSV file
results_df.to_csv('grid_search_results.csv', index=False)

print("Results saved in 'grid_search_results.csv'.")
