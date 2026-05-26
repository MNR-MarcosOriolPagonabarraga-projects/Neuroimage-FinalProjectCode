import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.fmri_dataset import build_window_records, discover_feature_paths, FmriRestChunkDataset
from src.utils import _set_seeds, _subject_stratified_split

PROCESSED_DATA_PATH = "data/processed"
RNG_SEED = 42
VAL_FRACTION = 0.2

def extract_features_and_labels(dataset: FmriRestChunkDataset):
    """Extrae el triángulo superior de la matriz de correlación a un array 1D."""
    X, y = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        # corr es (1, ROIs, ROIs). Extraemos solo la matriz 2D
        corr_matrix = sample["corr"].squeeze(0).numpy() 
        
        # Extraemos índices del triángulo superior (sin la diagonal)
        row_idx, col_idx = np.triu_indices_from(corr_matrix, k=1)
        upper_tri = corr_matrix[row_idx, col_idx]
        
        X.append(upper_tri)
        y.append(sample["label"])
        
    return np.array(X), np.array(y)

def main():
    _set_seeds(RNG_SEED)
    print("Cargando datos para Random Forest...")
    
    paths = discover_feature_paths(PROCESSED_DATA_PATH)
    records, meta = build_window_records(paths)
    
    train_idx, val_idx = _subject_stratified_split(records, VAL_FRACTION, RNG_SEED)
    
    # Solo necesitamos 'corr' para el Random Forest
    train_ds = FmriRestChunkDataset([records[i] for i in train_idx], "corr")
    val_ds = FmriRestChunkDataset([records[i] for i in val_idx], "corr")
    
    X_train, y_train = extract_features_and_labels(train_ds)
    X_val, y_val = extract_features_and_labels(val_ds)
    
    print(f"Vector de entrenamiento: {X_train.shape} (Muestras x Enlaces)")
    
    # Entrenar Random Forest
    # n_estimators=500 y max_depth controlado ayudan a combatir el overfitting típico de fMRI
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=16, 
        min_samples_split=5,
        random_state=RNG_SEED, 
        n_jobs=-1,
        class_weight="balanced"
    )
    
    print("Entrenando modelo...")
    clf.fit(X_train, y_train)
    
    # Evaluación
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print("\n--- Resultados Random Forest ---")
    print(f"Accuracy en Validación: {acc:.4f}\n")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_val, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_val, y_pred))
    
    # Guardar modelo
    joblib.dump(clf, "outputs/random_forest_fmri.pkl")
    print("Modelo guardado en outputs/random_forest_fmri.pkl")

if __name__ == "__main__":
    main()