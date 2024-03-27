import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Carregando os dados
data_path = 'phpPrh7lv.csv'
data = pd.read_csv(data_path)

# Seleção de atributos baseada em análise de correlação
selected_features = data[['V3', 'V5', 'V6', 'V7']]

# Normalização dos atributos
scaler = StandardScaler()
normalized_features = scaler.fit_transform(selected_features)
normalized_features_df = pd.DataFrame(normalized_features, columns=selected_features.columns)

# Definição de função para calcular coesão e separação
def calculate_cohesion_separation(data, labels):
    cluster_centers = []
    total_mean = np.mean(data, axis=0)
    within_cluster_sum = 0
    between_cluster_sum = 0
    
    for cluster in np.unique(labels):
        cluster_data = data[labels == cluster]
        cluster_mean = np.mean(cluster_data, axis=0)
        cluster_centers.append(cluster_mean)
        
        within_cluster_sum += np.sum((cluster_data - cluster_mean) ** 2)
        between_cluster_sum += len(cluster_data) * np.sum((cluster_mean - total_mean) ** 2)
    
    total_sum = np.sum((data - total_mean) ** 2)
    
    cohesion = within_cluster_sum / total_sum
    separation = between_cluster_sum / total_sum
    
    return cohesion, separation

# Agrupamento k-Means para 3 e 4 grupos
kmeans_3 = KMeans(n_clusters=3, random_state=42).fit(normalized_features_df)
kmeans_4 = KMeans(n_clusters=4, random_state=42).fit(normalized_features_df)

# Cálculo de coesão e separação
cohesion_3, separation_3 = calculate_cohesion_separation(normalized_features_df.to_numpy(), kmeans_3.labels_)
cohesion_4, separation_4 = calculate_cohesion_separation(normalized_features_df.to_numpy(), kmeans_4.labels_)

print(f"Coesão (3 grupos): {cohesion_3}, Separação (3 grupos): {separation_3}")
print(f"Coesão (4 grupos): {cohesion_4}, Separação (4 grupos): {separation_4}")

# Agrupamento hierárquico
linked = linkage(normalized_features_df, 'ward')

# Criação do dendrograma
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrograma do Agrupamento Hierárquico')
plt.xlabel('Índice da Instância')
plt.ylabel('Distância')
plt.show()
