import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from collections import Counter

# Configurar reproducibilidad
np.random.seed(42)

# 1. Cargar el dataset
df = pd.read_csv('smart_mobility_dataset.csv')

# 2. Preprocesamiento básico
# Eliminar valores nulos
df = df.dropna()

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

# 3. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 4. Aplicar preprocesamiento a los datos de entrenamiento
X_train_processed = preprocessor.fit_transform(X_train)

# Convertir a array denso si es necesario
if not isinstance(X_train_processed, np.ndarray):
    X_train_processed = X_train_processed.toarray()

# 5. Visualizar distribución original de clases
def plot_class_distribution(y, title):
    counter = Counter(y)
    plt.figure(figsize=(10, 6))
    plt.bar(label_encoder.classes_, [counter[i] for i in range(len(label_encoder.classes_))])
    plt.title(title)
    plt.xlabel('Nivel de Congestión del Tráfico')
    plt.ylabel('Número de Muestras')
    for i, count in enumerate([counter[i] for i in range(len(label_encoder.classes_))]):
        plt.text(i, count + 5, str(count), ha='center')
    plt.tight_layout()
    return plt

# Visualizar distribución original
plt_original = plot_class_distribution(y_train, 'Distribución Original de Clases')
plt_original.savefig('original_distribution.png')
plt_original.close()

print("Distribución original de clases:")
print(Counter(y_train))

# 6. Aplicar diferentes técnicas de balanceo
# Función para aplicar y visualizar técnica de balanceo
def apply_balancing_technique(X, y, technique_name, technique):
    print(f"\nAplicando {technique_name}...")
    X_resampled, y_resampled = technique.fit_resample(X, y)
    
    # Visualizar distribución balanceada
    plt_balanced = plot_class_distribution(y_resampled, f'Distribución de Clases después de {technique_name}')
    plt_balanced.savefig(f'{technique_name.lower().replace(" ", "_")}_distribution.png')
    plt_balanced.close()
    
    print(f"Distribución después de {technique_name}:")
    print(Counter(y_resampled))
    
    return X_resampled, y_resampled

# 6.1 SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_smote, y_smote = apply_balancing_technique(X_train_processed, y_train, "SMOTE", smote)

# 6.2 ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = apply_balancing_technique(X_train_processed, y_train, "ADASYN", adasyn)

# 6.3 Random Oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = apply_balancing_technique(X_train_processed, y_train, "Random Oversampling", ros)

# 7. Comparación de técnicas
print("\nResumen de balanceo de datos:")
print(f"Datos originales: {Counter(y_train)}")
print(f"Después de SMOTE: {Counter(y_smote)}")
print(f"Después de ADASYN: {Counter(y_adasyn)}")
print(f"Después de Random Oversampling: {Counter(y_ros)}")

# 8. Guardar datasets balanceados
import joblib

# Guardar los datos balanceados
joblib.dump((X_smote, y_smote), 'smote_balanced_data.joblib')
joblib.dump((X_adasyn, y_adasyn), 'adasyn_balanced_data.joblib')
joblib.dump((X_ros, y_ros), 'ros_balanced_data.joblib')

# Guardar el preprocesador
joblib.dump(preprocessor, 'preprocessor_for_balancing.joblib')
joblib.dump(label_encoder, 'label_encoder_for_balancing.joblib')

# 9. Función para cargar los datos balanceados
def load_balanced_data(technique_name):
    """
    Carga los datos balanceados según la técnica especificada
    
    Parámetros:
    technique_name (str): Nombre de la técnica ('smote', 'adasyn', 'ros', o 'original')
    
    Retorna:
    tuple: (X_balanceado, y_balanceado)
    """
    if technique_name.lower() == 'smote':
        return joblib.load('smote_balanced_data.joblib')
    elif technique_name.lower() == 'adasyn':
        return joblib.load('adasyn_balanced_data.joblib')
    elif technique_name.lower() == 'ros':
        return joblib.load('ros_balanced_data.joblib')
    elif technique_name.lower() == 'original':
        return X_train_processed, y_train
    else:
        raise ValueError("Técnica no reconocida. Usar 'smote', 'adasyn', 'ros', o 'original'")

print("\nDatos balanceados y guardados correctamente.")
print("Archivos generados:")
print("- smote_balanced_data.joblib: Datos balanceados con SMOTE")
print("- adasyn_balanced_data.joblib: Datos balanceados con ADASYN")
print("- ros_balanced_data.joblib: Datos balanceados con Random Oversampling")
print("- preprocessor_for_balancing.joblib: Preprocesador para nuevos datos")
print("- label_encoder_for_balancing.joblib: Codificador de etiquetas")
print("- original_distribution.png: Gráfico de distribución original")
print("- smote_distribution.png: Gráfico de distribución después de SMOTE")
print("- adasyn_distribution.png: Gráfico de distribución después de ADASYN")
print("- random_oversampling_distribution.png: Gráfico de distribución después de Random Oversampling")

# 10. Ejemplo de uso para entrenar un modelo con datos balanceados
print("\nEjemplo de cómo cargar y usar los datos balanceados:")
print("""
# Cargar datos balanceados
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import joblib

# Cargar datos balanceados (elegir una técnica)
X_balanced, y_balanced = load_balanced_data('smote')  # O 'adasyn', 'ros', 'original'

# Crear y entrenar el modelo
model = Sequential([
    Dense(128, activation='relu', input_dim=X_balanced.shape[1]),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(len(np.unique(y_balanced)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo con datos balanceados
model.fit(
    X_balanced, y_balanced,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
""")