import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Configurar reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# 1. Cargar el dataset
df = pd.read_csv('smart_mobility_dataset.csv')

# 2. Exploración inicial
print("Información del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Verificar valores únicos en la variable objetivo
print("\nDistribución de la variable objetivo (Traffic_Condition):")
print(df['Traffic_Condition'].value_counts())

# 3. Preprocesamiento de datos
# Verificar y manejar valores nulos
print("\nValores nulos en el dataset:")
print(df.isnull().sum())

# Manejar valores nulos si existen
df = df.dropna()  # O usar estrategias de imputación según sea necesario

# Separar características y variable objetivo
X = df.drop(['Traffic_Condition', 'Timestamp'], axis=1)  # También eliminamos Timestamp para el modelado
y = df['Traffic_Condition']

# Codificar la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nClases codificadas:")
for i, clase in enumerate(label_encoder.classes_):
    print(f"{clase}: {i}")

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

# 4. División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Aplicar preprocesamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Verificar la forma de los datos procesados
print(f"\nForma de los datos de entrenamiento procesados: {X_train_processed.shape}")

# Convertir a formato denso si es necesario (si hay características categóricas)
if isinstance(X_train_processed, np.ndarray):
    X_train_processed_dense = X_train_processed
    X_test_processed_dense = X_test_processed
else:
    X_train_processed_dense = X_train_processed.toarray()
    X_test_processed_dense = X_test_processed.toarray()

# 5. Definición del modelo de red neuronal
def create_model(input_dim, num_classes):
    model = Sequential([
        # Capa de entrada
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capas ocultas
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Capa de salida (3 clases: bajo, medio, alto)
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Número de clases (bajo, medio, alto)
num_classes = len(np.unique(y_encoded))

# Crear el modelo
model = create_model(X_train_processed_dense.shape[1], num_classes)

# Resumen del modelo
model.summary()

# 6. Entrenamiento del modelo
# Configurar early stopping para evitar el sobreajuste
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Entrenar el modelo
history = model.fit(
    X_train_processed_dense, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 7. Evaluación del modelo
# Evaluar en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test_processed_dense, y_test)
print(f"\nPrecisión en el conjunto de prueba: {test_accuracy:.4f}")

# Predecir clases para el conjunto de prueba
y_pred = model.predict(X_test_processed_dense)
y_pred_classes = np.argmax(y_pred, axis=1)

# Mostrar reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_classes, 
                           target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=label_encoder.classes_,
           yticklabels=label_encoder.classes_)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 8. Visualización de la curva de aprendizaje
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
plt.savefig('learning_curves.png')
plt.close()

# 9. Análisis de importancia de características (mediante un modelo auxiliar)
from sklearn.inspection import permutation_importance
import joblib
from sklearn.ensemble import RandomForestClassifier

# Entrenar un modelo auxiliar para el análisis de importancia
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed_dense, y_train)

# Calcular importancia de características
result = permutation_importance(rf_model, X_test_processed_dense, y_test, n_repeats=10, random_state=42)
importance = result.importances_mean

# Crear indices para las características después del preprocesamiento
feature_names = []
for name, trans, cols in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(cols)
    elif name == 'cat':
        for col in cols:
            feature_names.extend([f"{col}_{val}" for val in df[col].unique()])

# Limitar a la cantidad real de características
feature_names = feature_names[:X_train_processed_dense.shape[1]]

# Visualizar importancia de características
plt.figure(figsize=(12, 8))
sorted_idx = np.argsort(importance)
plt.barh(range(len(sorted_idx)), importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in sorted_idx])
plt.title('Importancia de Características')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 10. Guardar el modelo y los preprocesadores para su uso posterior
model.save('traffic_congestion_model.h5')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("\nModelo entrenado y guardado correctamente.")
print("Archivos generados:")
print("- traffic_congestion_model.h5: Modelo de red neuronal")
print("- preprocessor.joblib: Preprocesador de características")
print("- label_encoder.joblib: Codificador de etiquetas")
print("- confusion_matrix.png: Matriz de confusión")
print("- learning_curves.png: Curvas de aprendizaje")
print("- feature_importance.png: Importancia de características")

# 11. Implementar función para nuevas predicciones
def predict_traffic_condition(data, model, preprocessor, label_encoder):
    """
    Predice el nivel de congestión del tráfico para nuevos datos.
    
    Parámetros:
    data (pandas.DataFrame): Datos de entrada para la predicción
    model: Modelo entrenado
    preprocessor: Preprocesador entrenado
    label_encoder: Codificador de etiquetas entrenado
    
    Retorna:
    tuple: (Etiqueta predicha, Probabilidades)
    """
    # Preprocesar los datos
    processed_data = preprocessor.transform(data)
    
    # Asegurar que los datos estén en formato denso
    if not isinstance(processed_data, np.ndarray):
        processed_data = processed_data.toarray()
    
    # Realizar predicción
    probabilities = model.predict(processed_data)
    predicted_class_index = np.argmax(probabilities, axis=1)
    
    # Convertir índice a etiqueta
    predicted_class = label_encoder.inverse_transform(predicted_class_index)
    
    return predicted_class, probabilities

# Ejemplo de uso:
print("\nEjemplo de predicción con datos de prueba:")
sample_data = X_test.iloc[0:1]  # Tomar una muestra
predicted_class, probabilities = predict_traffic_condition(
    sample_data, model, preprocessor, label_encoder)

print(f"Datos de entrada:\n{sample_data.to_dict('records')}")
print(f"Clase predicha: {predicted_class[0]}")
print(f"Probabilidades: {probabilities[0]}")

# 12. Realizar un análisis de los resultados
print("\nAnálisis de resultados:")
print(f"- Precisión general: {accuracy_score(y_test, y_pred_classes):.4f}")

# Calcular precisiones por clase
class_accuracy = {}
for i, class_name in enumerate(label_encoder.classes_):
    class_mask = (y_test == i)
    class_pred = y_pred_classes[class_mask]
    class_true = y_test[class_mask]
    class_accuracy[class_name] = accuracy_score(class_true, class_pred)
    print(f"- Precisión para '{class_name}': {class_accuracy[class_name]:.4f}")

# Identificar desafíos y áreas de mejora
print("\nDesafíos y áreas de mejora:")
lowest_class = min(class_accuracy, key=class_accuracy.get)
print(f"- La clase con menor precisión es '{lowest_class}' con {class_accuracy[lowest_class]:.4f}")
print("  Se podría mejorar mediante técnicas de balanceo de datos o ajuste de hiperparámetros.")
print("- Considerar la incorporación de features adicionales o ingenierías de características.")
print("- Explorar arquitecturas de red neuronal más complejas o modelos ensemble.")