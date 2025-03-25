import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import joblib
import time

# Configurar reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# 1. Cargar el dataset
print("Cargando y procesando el dataset...")
df = pd.read_csv('smart_mobility_dataset.csv')
df = df.dropna()  # Eliminar valores nulos

# Separar características y variable objetivo
X = df.drop(['Traffic_Condition', 'Timestamp'], axis=1)
y = df['Traffic_Condition']

# Codificar la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Crear transformadores para características numéricas y categóricas
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Crear un preprocesador columnar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 2. División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Aplicar preprocesamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convertir a formato denso si es necesario
if not isinstance(X_train_processed, np.ndarray):
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()

# 3. Definición de diferentes arquitecturas de redes neuronales
def create_basic_model(input_dim, num_classes):
    """Modelo básico con pocas capas."""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_deeper_model(input_dim, num_classes):
    """Modelo más profundo con más capas."""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_wide_model(input_dim, num_classes):
    """Modelo ancho con más neuronas por capa."""
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_leaky_relu_model(input_dim, num_classes):
    """Modelo con LeakyReLU en lugar de ReLU."""
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 4. Función para evaluar modelo con validación cruzada
def evaluate_model_cv(create_model_func, X, y, n_splits=5, epochs=50, batch_size=32, verbose=0):
    """Evaluar modelo con validación cruzada."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    
    # Métricas por fold
    accuracy_per_fold = []
    loss_per_fold = []
    f1_per_fold = []
    precision_per_fold = []
    recall_per_fold = []
    
    # Tiempo de entrenamiento por fold
    time_per_fold = []
    
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    
    # Validación cruzada
    for train_idx, val_idx in kfold.split(X):
        # Dividir datos para este fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Crear modelo para este fold
        model = create_model_func(input_dim, num_classes)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        
        # Entrenar modelo
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        # Calcular tiempo de entrenamiento
        train_time = time.time() - start_time
        time_per_fold.append(train_time)
        
        # Evaluar modelo
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        loss_per_fold.append(scores[0])
        accuracy_per_fold.append(scores[1])
        
        # Predecir clases para métricas adicionales
        y_pred = np.argmax(model.predict(X_val_fold), axis=1)
        
        # Calcular métricas adicionales
        f1 = f1_score(y_val_fold, y_pred, average='weighted')
        precision = precision_score(y_val_fold, y_pred, average='weighted')
        recall = recall_score(y_val_fold, y_pred, average='weighted')
        
        f1_per_fold.append(f1)
        precision_per_fold.append(precision)
        recall_per_fold.append(recall)
        
        print(f'Fold {fold_no}: Accuracy = {scores[1]:.4f}, F1 = {f1:.4f}, Time = {train_time:.2f}s')
        fold_no += 1
    
    # Calcular métricas promedio
    avg_loss = np.mean(loss_per_fold)
    avg_accuracy = np.mean(accuracy_per_fold)
    avg_f1 = np.mean(f1_per_fold)
    avg_precision = np.mean(precision_per_fold)
    avg_recall = np.mean(recall_per_fold)
    avg_time = np.mean(time_per_fold)
    
    # Métricas de variabilidad
    std_accuracy = np.std(accuracy_per_fold)
    std_f1 = np.std(f1_per_fold)
    
    # Resumen
    print('-' * 50)
    print(f'Métricas de validación cruzada (promedio de {n_splits} folds):')
    print(f'Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})')
    print(f'F1 Score: {avg_f1:.4f} (±{std_f1:.4f})')
    print(f'Precision: {avg_precision:.4f}')
    print(f'Recall: {avg_recall:.4f}')
    print(f'Tiempo de entrenamiento: {avg_time:.2f}s')
    
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'accuracy_std': std_accuracy,
        'f1': avg_f1,
        'f1_std': std_f1,
        'precision': avg_precision,
        'recall': avg_recall,
        'time': avg_time
    }

# 5. Definir función para la evaluación final en conjunto de prueba
def final_evaluation(create_model_func, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Entrenar modelo con todos los datos de entrenamiento y evaluar en conjunto de prueba."""
    print('\nRealizando evaluación final en conjunto de prueba...')
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Crear y entrenar modelo
    model = create_model_func(input_dim, num_classes)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluar en conjunto de prueba
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Predecir clases
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calcular métricas
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Calcular ROC AUC (para problemas multiclase)
    y_pred_proba = model.predict(X_test)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión (Modelo Final)')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png')
    plt.close()
    
    # Visualizar curva de aprendizaje
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del Modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del Modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    plt.tight_layout()
    plt.savefig('final_learning_curves.png')
    plt.close()
    
    # Guardar el modelo final
    model.save(f'final_traffic_model.h5')
    
    return {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 6. Comparar diferentes arquitecturas de modelos
print("\nIniciando comparación de arquitecturas de redes neuronales...\n")

# Definir arquitecturas a comparar
architectures = {
    'Básica': create_basic_model,
    'Profunda': create_deeper_model,
    'Ancha': create_wide_model,
    'LeakyReLU': create_leaky_relu_model
}

# Almacenar resultados
results = {}

# Evaluar cada arquitectura con validación cruzada
for name, create_func in architectures.items():
    print(f"\nEvaluando arquitectura: {name}")
    print("-" * 30)
    
    results[name] = evaluate_model_cv(
        create_func,
        X_train_processed,
        y_train,
        n_splits=5,
        epochs=50,
        batch_size=32,
        verbose=0
    )

# 7. Comparar resultados
print("\n" + "=" * 50)
print("COMPARACIÓN DE ARQUITECTURAS DE REDES NEURONALES")
print("=" * 50)

# Crear dataframe con resultados
results_df = pd.DataFrame({
    'Arquitectura': list(results.keys()),
    'Accuracy': [results[arch]['accuracy'] for arch in results],
    'F1 Score': [results[arch]['f1'] for arch in results],
    'Precision': [results[arch]['precision'] for arch in results],
    'Recall': [results[arch]['recall'] for arch in results],
    'Tiempo (s)': [results[arch]['time'] for arch in results]
})

# Mostrar resultados
print(results_df.to_string(index=False))

# Visualizar resultados
plt.figure(figsize=(12, 10))

# Accuracy
plt.subplot(2, 2, 1)
bars = plt.bar(results_df['Arquitectura'], results_df['Accuracy'])
plt.title('Accuracy por Arquitectura')
plt.ylim(0.7, 1.0)  # Ajustar según resultados
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.4f}', ha='center', va='bottom')

# F1 Score
plt.subplot(2, 2, 2)
bars = plt.bar(results_df['Arquitectura'], results_df['F1 Score'])
plt.title('F1 Score por Arquitectura')
plt.ylim(0.7, 1.0)  # Ajustar según resultados
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.4f}', ha='center', va='bottom')

# Precision y Recall
plt.subplot(2, 2, 3)
width = 0.35
x = np.arange(len(results_df['Arquitectura']))
bars1 = plt.bar(x - width/2, results_df['Precision'], width, label='Precision')
bars2 = plt.bar(x + width/2, results_df['Recall'], width, label='Recall')
plt.xticks(x, results_df['Arquitectura'])
plt.title('Precision y Recall por Arquitectura')
plt.ylim(0.7, 1.0)  # Ajustar según resultados
plt.legend()

# Tiempo de entrenamiento
plt.subplot(2, 2, 4)
bars = plt.bar(results_df['Arquitectura'], results_df['Tiempo (s)'])
plt.title('Tiempo de Entrenamiento por Arquitectura (s)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('architecture_comparison.png')
plt.close()

# 8. Seleccionar mejor arquitectura y realizar evaluación final
best_architecture = results_df.iloc[results_df['F1 Score'].argmax()]['Arquitectura']
print(f"\nMejor arquitectura según F1 Score: {best_architecture}")

# Obtener función para crear el mejor modelo
best_model_func = architectures[best_architecture]

# Evaluación final con la mejor arquitectura
final_results = final_evaluation(
    best_model_func,
    X_train_processed,
    y_train,
    X_test_processed,
    y_test,
    epochs=100,
    batch_size=32
)

# Mostrar resultados finales
print("\nRESULTADOS FINALES (MEJOR MODELO)")
print("-" * 50)
print(f"Arquitectura seleccionada: {best_architecture}")
print(f"Accuracy en test: {final_results['test_accuracy']:.4f}")
print(f"F1 Score en test: {final_results['f1']:.4f}")
print(f"Precision en test: {final_results['precision']:.4f}")
print(f"Recall en test: {final_results['recall']:.4f}")

# 9. Guardar resultados y modelos
# Guardar resultados en CSV
results_df.to_csv('architecture_comparison_results.csv', index=False)

# Guardar preprocesador y codificador para uso futuro
joblib.dump(preprocessor, 'final_preprocessor.joblib')
joblib.dump(label_encoder, 'final_label_encoder.joblib')

print("\nComparación completada. Archivos generados:")
print("- architecture_comparison.png: Gráficos comparativos")
print("- architecture_comparison_results.csv: Resultados en formato CSV")
print("- final_confusion_matrix.png: Matriz de confusión del mejor modelo")
print("- final_learning_curves.png: Curvas de aprendizaje del mejor modelo")
print("- final_traffic_model.h5: Mejor modelo guardado")
print("- final_preprocessor.joblib: Preprocesador final")
print("- final_label_encoder.joblib: Codificador de etiquetas final")

# 10. Código de ejemplo para usar el modelo en producción
print("\nEjemplo de código para usar el modelo en producción:")
print("""
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo, preprocesador y codificador
model = load_model('final_traffic_model.h5')
preprocessor = joblib.load('final_preprocessor.joblib')
label_encoder = joblib.load('final_label_encoder.joblib')

# Función para predecir
def predict_traffic_condition(data):
    # Preprocesar datos
    processed_data = preprocessor.transform(data)
    
    # Convertir a array denso si es necesario
    if not isinstance(processed_data, np.ndarray):
        processed_data = processed_data.toarray()
    
    # Realizar predicción
    prediction_proba = model.predict(processed_data)
    prediction_class_idx = np.argmax(prediction_proba, axis=1)
    
    # Convertir índice a etiqueta
    prediction_class = label_encoder.inverse_transform(prediction_class_idx)
    
    return prediction_class, prediction_proba

# Ejemplo: predecir para nuevos datos
import pandas as pd

# Crear o cargar nuevos datos (mismo formato que en entrenamiento, sin 'Traffic_Condition')
new_data = pd.DataFrame({
    'Vehicle_Count': [150],
    'Traffic_Speed_kmh': [35.5],
    'Road_Occupancy_%': [65.2],
    'Weather_Condition': ['Cloudy'],
    # ... incluir todas las características requeridas excepto 'Traffic_Condition' y 'Timestamp'
})

# Realizar predicción
predicted_class, probabilities = predict_traffic_condition(new_data)
print(f'Clase predicha: {predicted_class[0]}')
print(f'Probabilidades: {probabilities[0]}')
""")