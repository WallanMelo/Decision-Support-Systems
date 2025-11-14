import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

dados = pd.read_csv("Trabalho_02_dataset_dengue_januaria_2024.csv", sep=';')
dados.head(50)

plt.figure(figsize=(16,8))
plt.plot(dados['Semana epidem. 1º Sintomas(s)'],dados['Casos_Prováveis'], marker='o', linestyle='-', color='blue')
plt.title('Evolução Semanal dos Casos Da dengue em Januaria 2024')
plt.xlabel('Semana Epidemologica')
plt.ylabel('Casos Prováveis')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

def modelagem_dengue(variaveis, tempo, taxa_transm_mosq_hum, taxa_transm_hum_mosq, taxa_recuperacao, taxa_morte_mosq, populacao_humana, populacao_mosquitos):
    humanos_suscetiveis, humanos_infectados, humanos_recuperados, mosquitos_suscetiveis, mosquitos_infectados = variaveis

    forca_infeccao_hum = taxa_transm_mosq_hum * mosquitos_infectados / populacao_humana###DDe Moskito p Humano --- mosk -> hum
    forca_infeccao_mosq = taxa_transm_hum_mosq * humanos_infectados / populacao_humana###DDe Humano p Moskito --- hum -> mosk

    d_humanos_suscetiveis = - forca_infeccao_hum * humanos_suscetiveis
    d_humanos_infectados =   forca_infeccao_hum * humanos_suscetiveis - taxa_recuperacao * humanos_infectados
    d_humanos_recuperados =  taxa_recuperacao * humanos_infectados

    d_mosquitos_suscetiveis = taxa_morte_mosq * populacao_mosquitos - forca_infeccao_mosq * mosquitos_suscetiveis - taxa_morte_mosq * mosquitos_suscetiveis
    d_mosquitos_infectados =  forca_infeccao_mosq * mosquitos_suscetiveis - taxa_morte_mosq * mosquitos_infectados

    return [d_humanos_suscetiveis, d_humanos_infectados, d_humanos_recuperados, d_mosquitos_suscetiveis, d_mosquitos_infectados]

populacao_humana = 67000## Não existe numero exato então colocamos um valor aproximado seguindo fontes
populacao_mosquitos = 55000## Valor aproximado da população de moskitos em janu

infectados_iniciais = max(1, dados['Casos_Prováveis'].iloc[0])
recuperados_iniciais = 0
suscetiveis_iniciais = populacao_humana - infectados_iniciais - recuperados_iniciais
mosquitos_suscetiveis_iniciais = populacao_mosquitos - 1
mosquitos_infectados_iniciais = 1

condicoes_iniciais = [
    suscetiveis_iniciais, infectados_iniciais, recuperados_iniciais,
    mosquitos_suscetiveis_iniciais, mosquitos_infectados_iniciais
]

## Definição dos valores dos parametros bioçogicos
taxa_transm_mosq_hum = 0.2
taxa_transm_hum_mosq = 0.1
taxa_recuperacao = 1/7
taxa_morte_mosq = 1/14

tempo = np.linspace(0, len(dados)*7, len(dados)*7 + 1)

solucao = odeint(
    modelagem_dengue,
    condicoes_iniciais,
    tempo,
    args=(taxa_transm_mosq_hum, taxa_transm_hum_mosq, taxa_recuperacao, taxa_morte_mosq, populacao_humana, populacao_mosquitos)
)

humanos_suscetiveis, humanos_infectados, humanos_recuperados, mosquitos_suscetiveis, mosquitos_infectados = solucao.T

incidencia_diaria = taxa_transm_mosq_hum * humanos_suscetiveis * mosquitos_infectados / populacao_humana
incidencia_semanal = [incidencia_diaria[i*7:(i+1)*7].sum() for i in range(len(dados))]


plt.figure(figsize=(16,8))
plt.plot(range(len(dados)), dados['Casos_Prováveis'],  'o-', label='Casos Reais')#Casos Reais
plt.plot(range(len(incidencia_semanal)), incidencia_semanal, 's--', label='Casos Simulados')#Casos Simulados(pelo modelo e simulação temopral)
plt.xlabel('Semana epidemilogica')
plt.ylabel('Casos de Dengue')
plt.title('Grafico de Comparação dos DADOS REAIS x DADOS SIMULADOS')
plt.legend()
plt.grid(True)
plt.show()


def simular_semana(t, taxa_transm_mosq_hum, taxa_transm_hum_mosq, populacao_mosquitos):
    taxa_recuperacao = 1/7
    taxa_morte_mosq = 1/14

    infectados_iniciais = max(1, dados['Casos_Prováveis'].iloc[0])
    recuperados_iniciais = 0
    suscetiveis_iniciais = populacao_humana - infectados_iniciais - recuperados_iniciais
    mosquitos_suscetiveis_iniciais = populacao_mosquitos - 1
    mosquitos_infectados_iniciais = 1

    condicoes_iniciais = [
        suscetiveis_iniciais, infectados_iniciais, recuperados_iniciais,
        mosquitos_suscetiveis_iniciais, mosquitos_infectados_iniciais
    ]

    t_dias = np.linspace(0, len(dados)*7, len(dados)*7 + 1)
    solucao = odeint(modelagem_dengue, condicoes_iniciais, t_dias,
                     args=(taxa_transm_mosq_hum, taxa_transm_hum_mosq, taxa_recuperacao, taxa_morte_mosq, populacao_humana, populacao_mosquitos))

    humanos_suscetiveis, humanos_infectados, humanos_recuperados, mosquitos_suscetiveis, mosquitos_infectados = solucao.T
    incidencia_diaria = taxa_transm_mosq_hum * humanos_suscetiveis * mosquitos_infectados / populacao_humana
    incidencia_semanal = [incidencia_diaria[i*7:(i+1)*7].sum() for i in range(len(dados))]
    return np.array(incidencia_semanal)

t = np.arange(len(dados))
observado = dados['Casos_Prováveis'].values

p0 = [0.2, 0.1, 300000]
limites = ([0.001, 0.001, 10000], [1.0, 1.0, 2000000])

popt, pcov = curve_fit(simular_semana, t, observado, p0=p0, bounds=limites, maxfev=5000)

taxa_transm_mosq_hum_otim, taxa_transm_hum_mosq_otim, populacao_mosquitos_otim = popt
print(f"Parâmetros c Otimização:")
print(f"Taxa de transmissão moskito ---> humano: {taxa_transm_mosq_hum_otim:.4f}")
print(f"Taxa transmissão humano ---> moskito: {taxa_transm_hum_mosq_otim:.4f}")
print(f"População de mosquitos estimada: {populacao_mosquitos_otim:.0f}")


simulado = simular_semana(t, *popt)### simulação com os parâmetros otimizados
mse = mean_squared_error(observado, simulado)###Erro quadrático médio

print(f"Erro quadrático médio: {mse:.2f}")

plt.figure(figsize=(16,8))
plt.plot(t,observado,'o-',label='Casos reais')
plt.plot(t,simulado,'s--',label='Casos Simulados')
plt.title("Modelo Otimizado")
plt.xlabel('semana epidemiologica')
plt.ylabel('Casos')
plt.legend()
plt.grid(True)
plt.show()