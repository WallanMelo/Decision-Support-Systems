# IMPORTAÇÕES DAS BIBLIOTECAS NECESSÁRIAS
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

#Carregamento e pre-processamento da base de dados íris
#Foi carregada a partir da própria biblioteca do sklearn
iris = load_iris()
x = iris.data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

## Ponto gerado para teste
novoPonto = [[5.9, 3, 5.1, 1.8]]
novoPontoScaled = scaler.transform(novoPonto)

#Criação do Kmeans para comparação
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(x_scaled)

#========== Aplicação do Gaussian Mixture Model
#O objeto "novoPonto" foi criado para exemplificar uma diferença marcante entre esse algoritmo e o kmeans. 
#O fato do Gaussian Mixture Model conseguir dizer a chance de um novo ponto pertencer a um cluster específico.

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, n_init=10)
clusters = gmm.fit_predict(x_scaled)
probPonto = gmm.predict_proba(novoPontoScaled)

cores = ['purple', 'yellow', 'green']

plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
for i in range(3):
  plt.scatter(x_scaled[clusters == i, 2], x_scaled[clusters == i, 3], c=cores[i], label=f"cluster {i}")

plt.scatter(novoPontoScaled[0, 2], novoPontoScaled[0, 3], s=50, c="red", label=f'0 - {probPonto[0][0]*100:.2f}%, 1 - {probPonto[0][1]*100:.2f}%, 2 - {probPonto[0][2]*100:.2f}%')
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend()
plt.title("Clusters do Gaussian Mixture Model")

plt.subplot(1, 2, 2)
for i in range(3):
  plt.scatter(x_scaled[clusters_kmeans == i, 2], x_scaled[clusters_kmeans == i, 3], c=cores[i], label=f"cluster {i}")

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend()
plt.title("Clusters do Kmeans")

plt.show()

### COMPARAÇÃO DOS RESULTADOS DOS DOIS ALGORITMOS ---> KMEANS VS GAUSSIAN MIXTURE MODEL
# Métricas de validação

ari_score = adjusted_rand_score(clusters_kmeans, clusters)
nmi_score = normalized_mutual_info_score(clusters_kmeans, clusters)
print(f"ARI Score: {ari_score:.3f} | NMI Score: {nmi_score:.3f}\n")

plt.figure(figsize=(10, 8))

for i in range(3):
    mask_both = (clusters_kmeans == i) & (clusters == i)
    plt.scatter(x_scaled[mask_both, 2], x_scaled[mask_both, 3],
                c='white', edgecolors=cores[i], s=120, linewidth=2.5,
                label=f'Acordo cluster {i}', alpha=0.9, zorder=3)


mask_discord = clusters_kmeans != clusters
plt.scatter(x_scaled[mask_discord, 2], x_scaled[mask_discord, 3],
            c='red', alpha=0.7, s=60, label='Discordância', zorder=2)

plt.scatter(novoPontoScaled[0, 2], novoPontoScaled[0, 3],
            s=200, c='black', marker='X', linewidth=3,
            label=f'0 - {probPonto[0][0]*100:.1f}%| 1 - {probPonto[0][1]*100:.1f}%| 2 - {probPonto[0][2]*100:.1f}%', zorder=5)




plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Comparação KMeans vs GMM\n(Branco = Acordo | Vermelho = Discordância)", fontsize=14, pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
