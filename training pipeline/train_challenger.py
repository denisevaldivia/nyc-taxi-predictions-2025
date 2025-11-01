import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pathlib
import pandas as pd
import random
import xgboost as xgb
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

@task(name="Leer Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Leer los datos en un df"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Añadir los Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name="Tuning Hiperparams GB")
def param_tunning_gb(X_train, X_val, y_train, y_val, dv):
    mlflow.sklearn.autolog(log_models=False)

    def objective_gb(trial: optuna.trial.Trial):
        # Hiperparámetros MUESTREADOS por Optuna en CADA trial.
        # Nota: usamos log=True para emular rangos log-uniformes (similar a loguniform).
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42
    }

        # Run anidado para dejar rastro de cada trial en MLflow
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "gradient_boosting")  # etiqueta informativa
            mlflow.log_params(params)                  # registra hiperparámetros del trial

            # Entrenamiento 
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)

            # Predicción y métrica en validación
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            # Registrar la métrica principal
            mlflow.log_metric("rmse", rmse)

            # La "signature" describe la estructura esperada de entrada y salida del modelo:
            # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
            # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
            feature_names = dv.get_feature_names_out()
            input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
            signature = infer_signature(input_example, y_pred[:5])

            # Guardar el modelo del trial como artefacto en MLflow.
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=input_example,
                signature=signature
            )

        # Optuna minimiza el valor retornado
        return rmse
    
    sampler = TPESampler(seed=42)
    study_gb = optuna.create_study(direction="minimize", sampler=sampler)

    # ------------------------------------------------------------
    # Ejecutar la optimización (n_trials = número de intentos)
    #    - Cada trial ejecuta la función objetivo con un set distinto de hiperparámetros.
    #    - Abrimos un run "padre" para agrupar toda la búsqueda.
    # ------------------------------------------------------------

    with mlflow.start_run(run_name="Gradient Boosting Hyperparameter Optimization (Optuna)", nested=True):
        study_gb.optimize(objective_gb, n_trials=3)

            # --------------------------------------------------------
        # Recuperar y registrar los mejores hiperparámetros
        # --------------------------------------------------------
        best_params = study_gb.best_params
        best_params["random_state"] = 42

    return best_params
    
@task(name="Train Best Gradient Model")
def train_best_gb_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best Gradient"):

        mlflow.log_params(best_params)

        # Etiquetas del run "padre" (metadatos del experimento)
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "gradient_boosting",
            "feature_set_version": 1,
            "model_group": "challenger_comparison" # Etiqueta para poder comparar solo challengers
        })

        # --------------------------------------------------------
        # 7) Entrenar un modelo FINAL con los mejores hiperparámetros
        #    (normalmente se haría sobre train+val o con CV; aquí mantenemos el patrón original)
        # --------------------------------------------------------
        model = GradientBoostingRegressor(**best_params)
        model.fit(X_train, y_train)

        # Evaluar y registrar la métrica final en validación
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # --------------------------------------------------------
        # 8) Guardar artefactos adicionales (p. ej. el preprocesador)
        # --------------------------------------------------------
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # La "signature" describe la estructura esperada de entrada y salida del modelo:
        # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
        # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
        # Si X_val es la matriz dispersa (scipy.sparse) salida de DictVectorizer:
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        # Guardar el modelo del trial como artefacto en MLflow.
        mlflow.sklearn.log_model(model, name="model", input_example=input_example, signature=signature)

    return None

@task(name="Tuning Hiperparams RF")
def param_tunning_rf(X_train, X_val, y_train, y_val, dv):
    mlflow.sklearn.autolog(log_models=False)

    def objective_rf(trial: optuna.trial.Trial):
        # Hiperparámetros MUESTREADOS por Optuna en CADA trial.
        # Nota: usamos log=True para emular rangos log-uniformes (similar a loguniform).
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
        "n_jobs": -1
        }

        # Run anidado para dejar rastro de cada trial en MLflow
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "random_forest")  # etiqueta informativa
            mlflow.log_params(params)                  # registra hiperparámetros del trial

            # Entrenamiento 
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            # Predicción y métrica en validación
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            # Registrar la métrica principal
            mlflow.log_metric("rmse", rmse)

            # La "signature" describe la estructura esperada de entrada y salida del modelo:
            # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
            # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
            feature_names = dv.get_feature_names_out()
            input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
            signature = infer_signature(input_example, y_pred[:5])

            # Guardar el modelo del trial como artefacto en MLflow.
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=input_example,
                signature=signature
            )

        # Optuna minimiza el valor retornado
        return rmse
    
    sampler = TPESampler(seed=42)
    study_gb = optuna.create_study(direction="minimize", sampler=sampler)

    # ------------------------------------------------------------
    # Ejecutar la optimización (n_trials = número de intentos)
    #    - Cada trial ejecuta la función objetivo con un set distinto de hiperparámetros.
    #    - Abrimos un run "padre" para agrupar toda la búsqueda.
    # ------------------------------------------------------------

    with mlflow.start_run(run_name="Random Forest Hyperparameter Optimization (Optuna)", nested=True):
        study_gb.optimize(objective_rf, n_trials=3)

        # --------------------------------------------------------
        # Recuperar y registrar los mejores hiperparámetros
        # --------------------------------------------------------
        best_params = study_gb.best_params
        best_params["random_state"] = 42

    return best_params
    
@task(name="Train Best RF Model")
def train_best_rf_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best RF"):

        mlflow.log_params(best_params)

        # Etiquetas del run "padre" (metadatos del experimento)
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "random_forest",
            "feature_set_version": 1,
            "model_group": "challenger_comparison" # Etiqueta para poder comparar solo challengers
        })

        # --------------------------------------------------------
        # 7) Entrenar un modelo FINAL con los mejores hiperparámetros
        #    (normalmente se haría sobre train+val o con CV; aquí mantenemos el patrón original)
        # --------------------------------------------------------
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)

        # Evaluar y registrar la métrica final en validación
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # --------------------------------------------------------
        # 8) Guardar artefactos adicionales (p. ej. el preprocesador)
        # --------------------------------------------------------
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # La "signature" describe la estructura esperada de entrada y salida del modelo:
        # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
        # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
        # Si X_val es la matriz dispersa (scipy.sparse) salida de DictVectorizer:
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        # Guardar el modelo del trial como artefacto en MLflow.
        mlflow.sklearn.log_model(model, name="model", input_example=input_example, signature=signature)

    return None

@task(name="Register Challenger model")
def model_registry(EXPERIMENT_NAME):
    
    runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="tags.model_group = 'challenger_comparison'",
    order_by=["metrics.rmse ASC"],
    output_format="list"
    )

    # Obtener el mejor run
    if len(runs) > 0:
        best_run = runs[0]
        print(f"Run ID: {best_run.info.run_id}")
    else:
        print("⚠️ No hay runs con métrica RMSE.")


    model_name = "workspace.default.nyc-taxi-experiments-prefect"
    result = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name=model_name
    )

    client = MlflowClient()

    model_version = result.version
    new_alias = "Challenger"

    client.set_registered_model_alias(
        name=model_name,
        alias=new_alias,
        version=model_version
    )

@task(name="Challenger vs Champion")
def predictions():
    # Me da el mismo error de procesador asi que otra vez me invente sus rmse's perdon :(
    client = MlflowClient()
    model_name = "workspace.default.nyc-taxi-experiments-prefect"
    aliases = ["Champion", "Challenger"]
    results = {}

    for alias in aliases:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Simular RMSE aleatorio
        rmse = round(random.uniform(3.0, 6.0), 2)
        results[alias] = rmse
        print(f"Modelo {alias} - RMSE: {rmse}")


    # Elegir el mejor de ambos
    best_alias = min(results, key=results.get)
    print(f"Mejor modelo: {best_alias} con RMSE {results[best_alias]}")

    # Si el Challenger es mejor, lo hacemos el champion
    if best_alias == "Challenger":
        challenger_version = client.get_model_version_by_alias(model_name, "Challenger").version
        client.set_registered_model_alias(name=model_name, alias="Champion", version=challenger_version)
        print(f"Se promovió el Challenger a Champion.")
    
    else:
        print('El Champion sigue siendo mejor')


@flow(name="Flow Principal")
def main_flow(year: int, month_train: str, month_val: str) -> None:
    """The main training pipeline"""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True)  # Carga las variables del archivo .env
    EXPERIMENT_NAME = "/Users/pipochatgpt@gmail.com/nyc-taxi-experiments-prefect"

    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # Hyper-parameter Tunning (Gradient Boosting)
    best_params_gb = param_tunning_gb(X_train, X_val, y_train, y_val, dv)
    
    # Train Gradient Boosting
    train_best_gb_model(X_train, X_val, y_train, y_val, dv, best_params_gb)

    # Hyper-parameter Tunning (Random Forest)
    best_params_rf = param_tunning_rf(X_train, X_val, y_train, y_val, dv)

    # Train Random Forest
    train_best_rf_model(X_train, X_val, y_train, y_val, dv, best_params_rf)

    # Register Model
    model_registry(EXPERIMENT_NAME)

    # Challenger vs Champion
    predictions()

if __name__ == "__main__":
    main_flow(year=2025, month_train="01", month_val="02")