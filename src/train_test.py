import ast
import pickle
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from src.utils import plot_roc_curve_graph


def convert_to_numeric_list(value):
    try:
        # If the value is a string that represents a list, safely parse it
        if isinstance(value, str):
            return ast.literal_eval(value)
        elif isinstance(value, list):
            # If the value is already a list, return as is
            return value
        else:
            # Handle missing or invalid values
            return []
    except (ValueError, SyntaxError):
        # Return an empty list if there's an error during conversion
        return []


def extract_features_from_sequences(df):
    # Converter os valores de y_values (centroids) em listas numéricas
    df['y_values'] = df['y_values'].apply(convert_to_numeric_list)

    # Extrair estatísticas dos y_values (centroids)
    df['y_mean'] = df['y_values'].apply(np.mean)
    df['y_std'] = df['y_values'].apply(np.std)
    df['y_min'] = df['y_values'].apply(np.min)
    df['y_max'] = df['y_values'].apply(np.max)

    # Calcular estatísticas para as velocidades
    df['speeds'] = df['speeds'].apply(
        convert_to_numeric_list)  # Converter para lista
    df['speed_mean'] = df['speeds'].apply(np.mean)
    df['speed_max'] = df['speeds'].apply(np.max)
    df['speed_final'] = df['speeds'].apply(
        lambda x: x[-1] if len(x) > 0 else 0)  # Velocidade final

    # Converter direction_vector para listas numéricas e calcular estatísticas (opcional, como magnitude)
    # df['direction_vector'] = df['direction_vector'].apply(
    #     convert_to_numeric_list)
    # df['direction_magnitude'] = df['direction_vector'].apply(
    #     lambda directions: np.mean([np.linalg.norm(d) for d in directions]) if len(directions) > 0 else 0)

    # Selecionar features relevantes
    feature_columns = ['y_mean', 'y_std', 'y_min', 'y_max',
                       'speed_mean', 'speed_max', 'speed_final']

    return df[feature_columns], df['winner']


def train_and_test(train_file_name, test_file_name) -> None:
    df = pd.read_csv(train_file_name)
    new_video_df = pd.read_csv(test_file_name)

    # Extrair features e rótulos do conjunto de treinamento
    X, y = extract_features_from_sequences(df)

    # Extrair features do novo vídeo para prever os vencedores
    X_new, _ = extract_features_from_sequences(new_video_df)

    # Separar o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Carregar ou treinar o modelo Random Forest
    model = None
    filename = "models/model.pkl"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
    else:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    # Avaliar o modelo no conjunto de teste
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Accuracy Default: {accuracy:.2f}")
    print(f"AUC Default: {auc_score:.2f}")

    # Plotar a curva ROC
    plot_roc_curve_graph(fpr, tpr, auc_score)

    new_predictions = model.predict(X_new)
    # Salvar as previsões no dataset do novo vídeo
    new_video_df['predicted_winner'] = new_predictions
    new_video_df.to_csv(
        'data/test_dataset_predictions_regular.csv', index=False)

    # Treinamento com Random Search para hiperparâmetros
    rs_model = None
    filename = "models/model_rfc.pkl"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            rs_model = pickle.load(f)
    else:
        rfc_search_space = {
            'n_estimators': range(10, 101),
            'criterion': ['gini', 'entropy'],
            'max_depth': range(2, 51),
            'min_samples_split': range(2, 11),
            'min_samples_leaf': range(1, 11),
            'max_features': ['sqrt', 'log2', None]
        }

        rfc = RandomForestClassifier()
        rs_model = RandomizedSearchCV(
            estimator=rfc, param_distributions=rfc_search_space, n_iter=100, cv=5)
        rs_model.fit(X_train, y_train)
        with open(filename, 'wb') as f:
            pickle.dump(rs_model, f)

    # Obter os melhores hiperparâmetros e treinar o modelo final
    best_params = rs_model.best_params_
    rfc = RandomForestClassifier(**best_params)
    rfc.fit(X_train, y_train)

    # Avaliar o modelo com os melhores hiperparâmetros
    y_pred = rfc.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print("Accuracy HTRS:", accuracy)
    print(f"AUC HTRS: {auc_score:.2f}")

    new_predictions = rfc.predict(X_new)

    # Salvar as previsões no dataset do novo vídeo
    new_video_df['predicted_winner'] = new_predictions
    new_video_df.to_csv('data/test_dataset_predictions_htrs.csv', index=False)
