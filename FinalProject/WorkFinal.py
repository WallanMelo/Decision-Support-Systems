# ======== IMPORTAÇÃO DAS BIBLIOTECAS ========
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ======== LEITURA e ARMAZENAMENTIO DO DATASET ========
df = pd.read_csv('./FinalProject/Dataset.csv') ## Faz a leitura do dataset e o armazena em um DataFrame do Pandas

# ======== PRÉ-PROCESSAMENTO DOS DADOS ========
# Conversão de variáveis categóricas (Yes/No) para valores numéricos (1/0)
df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

# ======== ANÁLISE DOS DADOS ========
df.head()#Mostra as primeiras linhas do dataset
df.info()# Mostra informações gerais do dataset (tipos de dados e valores nulos)
df.describe()# Exibe estatísticas descritivas das variáveis numéricas do dataser

# ======== ANÁLISE GRÁFICA ========
# Gráfico de barras mostrando a distribuição das classes da variável alvo (Price range) pela Quantidade
df['Price range'].value_counts().plot(kind='bar')
plt.title('Distribuição da Faixa de Preço')
plt.xlabel('Faixa de Preço')
plt.ylabel('Quantidade')
plt.show()

# custo médio para duas pessoas em cada faixa de preço
df.boxplot(column='Average Cost for two', by='Price range')
plt.title('Custo Médio por Faixa de Preço')
plt.suptitle('')
plt.show()

# Avaliação agregada em cada faixa de preço
df.boxplot(column='Aggregate rating', by='Price range')
plt.title('Avaliação por Faixa de Preço')
plt.suptitle('')
plt.show()

# Delivery online com faixa de preço
pd.crosstab(df['Has Online delivery'], df['Price range']).plot(kind='bar')
plt.title('Delivery por Faixa de Preço')
plt.show()

# ======== DEFINIÇÃO DAS VARIÁVEIS DE ENTRADA E SAÍDA ========
# seleção das features(variáveis independentes)
features = [
    'Average Cost for two',
    'Votes',
    'Aggregate rating',
    'Has Table booking',
    'Has Online delivery',
    'City',
    'Cuisines'
]

X = df[features]# variáveis explicativas

y = df['Price range']# variável alvo (classe a ser prevista)

# ======== TRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS ========
# Converte variáveis categóricas (City e Cuisines) em variáveis dummy (one-hot encoding)
# drop_first=True evita multicolinearidade
X = pd.get_dummies(X, drop_first=True)

# ======== DIVISÃO DO CONJUNTO EM TREINO E TESTE ========
# Os dados serão divididos em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify=y garante que as proporções das classes sejam mantidas
)

# ======== PADRONIZAÇÃO DOS  DADOS ========
scaler = StandardScaler() # objeto de normalização
X_train_scaled = scaler.fit_transform(X_train)# ajustando o scaler nos dados de treino e transformando-os
X_test_scaled = scaler.transform(X_test)# Aplica a mesma transformação nos dados de teste

# ======== TREINAMENTO DOS MODELOS ========
# Metodo Regressão Logística
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Metodo Árvore de Decisão
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Metodo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Metodo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ======== FUNÇÃO DE AVALIAÇÃO DOS MODELOS =========
#funct p mostrar as os resultados das metricas de avaliação de cada modelo
def avaliar(y_test, y_pred, nome):
    print(f'\nModelo: {nome}')
    print('Acurácia:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Resultado da Avaliação de todos os modelos utilizados p classificação
avaliar(y_test, y_pred_lr, 'Regressão Logística')
avaliar(y_test, y_pred_knn, 'KNN')
avaliar(y_test, y_pred_dt, 'Árvore de Decisão')
avaliar(y_test, y_pred_rf, 'Random Forest')

# ======== MATRIZ DE CONFUSÃO ======================
# calculando a matriz de confusão para o Random Forest
cm = confusion_matrix(y_test, y_pred_rf)

# Gráfico da matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão – Random Forest')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()