# 📡 Churn Guard: Pipeline Preditivo de Telecomunicações

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Concluído-success.svg)]()

> Sistema preditivo end-to-end desenvolvido para antecipação de evasão de clientes (*Churn*) no setor de telecomunicações, mitigando o Custo de Aquisição de Clientes (CAC) e subsidiando campanhas proativas de retenção.

---

## 📋 Sumário
1. [Visão Geral do Problema de Negócio](#-visão-geral-do-problema-de-negócio)
2. [Arquitetura do Pipeline](#-arquitetura-do-pipeline)
3. [Tecnologias Utilizadas](#-tecnologias-utilizadas)
4. [Estrutura do Repositório](#-estrutura-do-repositório)
5. [Instalação e Execução](#-instalação-e-execução)
6. [Estratégia de Versionamento Git (Git Flow)](#-estratégia-de-versionamento-git-git-flow)
7. [Resultados e Diagnóstico de Overfitting](#-resultados-e-diagnóstico-de-overfitting)
8. [Veredito Final de Negócio](#-veredito-final-de-negócio)
9. [Propostas de Melhorias Futuras](#-propostas-de-melhorias-futuras)
10. [Vídeo Demonstrativo](#-vídeo-demonstrativo)

---

## 🎯 Visão Geral do Problema de Negócio

Em provedoras de internet e telefonia, reter um cliente ativo tem custo consideravelmente inferior ao de adquirir um novo. Este projeto implementa um classificador preditivo capaz de sinalizar contratos com alto risco de cancelamento (`cancelou = 1`), permitindo que a equipe de marketing e atendimento implemente ofertas de retenção personalizadas antes da perda efetiva do contrato.

---

## 🛠️ Arquitetura do Pipeline

O pipeline foi estruturado em 7 fases sequenciais no arquivo `pipeline_churn_telecom.ipynb`:

* **Fase 1: Análise Exploratória (EDA)**: Avaliação de distribuições, detecção de desbalanceamento amostral (73,5% ativos vs. 26,5% cancelados) e análise de correlação linear de Pearson.
* **Fase 2: Limpeza e Tratamento (Data Prep)**: Remoção de registros duplicados e imputação de valores ausentes pela **Mediana** (escolha técnica justificada pela robustez frente a distribuições assimétricas e outliers severos).
* **Fase 3: Engenharia de Recursos (Feature Engineering)**: Construção da variável `gasto_medio_mensal = valor_total / meses_contrato`, contendo salvaguarda algorítmica para contratos recentes (`meses_contrato == 0`) para evitar divisão por zero.
* **Fase 4: Divisão e Balanceamento**: Separação estratificada (80% treino / 20% teste via `stratify=y`) e aplicação da técnica **SMOTE exclusivamente nos dados de treino**, prevenindo vazamento de dados (*Data Leakage*).
* **Fase 5: Escalonamento Seletivo**: Aplicação de `StandardScaler` (`fit_transform` no treino e `transform` no teste) exclusivamente para o KNN. Os dados da Árvore de Decisão são mantidos sem alteração de escala por ser um modelo invariante a transformações monotônicas.
* **Fase 6: Combate ao Overfitting**: Avaliação de complexidade variando vizinhança no KNN ($K \in \{3, 5, 7\}$) e profundidade na Árvore (`max_depth` $\in \{3, 5, \text{None}\}$).
* **Fase 7: Avaliação e Veredito**: Cálculo das métricas na base de teste não vista e definição do modelo final para produção.

---

## 💻 Tecnologias Utilizadas

* **Python 3.10+**
* **Pandas**: Manipulação tabular e tratamento estatístico.
* **NumPy**: Operações vetorizadas e álgebra linear.
* **Matplotlib & Seaborn**: Visualizações descritivas, distribuições e matrizes de confusão.
* **Scikit-Learn**: Algoritmos de Machine Learning (KNN, Árvore de Decisão), métricas e pré-processamento.
* **Imbalanced-Learn**: Reamostragem sintética via SMOTE.
* **Jupyter Notebook**: Ambiente interativo de experimentação e documentação científica.

---

## 📁 Estrutura do Repositório

```text
├── data/
│   └── dataset_recuperacao_telecom.csv   # Base de dados original (caminho relativo)
├── pipeline_churn_telecom.ipynb          # Notebook Jupyter principal segmentado nas 7 fases
├── requirements.txt                      # Especificação estrita de dependências Python
└── README.md                             # Documentação técnica e guia de execução
```

---

## ⚙️ Instalação e Execução

### Pré-requisitos
* Python 3.10 ou superior instalado.
* Gerenciador de pacotes `pip`.

### Passo a Passo

1. **Clonar o Repositório**:
   ```bash
   git clone https://github.com/SEU_USUARIO/churn-guard-telecom.git
   cd churn-guard-telecom
   ```

2. **Criar e Ativar Ambiente Virtual**:
   * No Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * No Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Instalar Dependências**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Executar o Pipeline**:
   ```bash
   jupyter notebook pipeline_churn_telecom.ipynb
   ```
   *No menu do Jupyter, selecione `Cell -> Run All` para executar todas as etapas reprodutivelmente.*

---

## 🌿 Estratégia de Versionamento Git (Git Flow)

Para atender às boas práticas de engenharia de software, o desenvolvimento foi organizado com branches de funcionalidades isoladas a partir da branch `develop`, preservadas sem exclusão após os merges:

* `main`: Código estável e homologado para entrega.
* `develop`: Branch de integração contínua.
* `feature/fase-1-eda`: Análise exploratória e geração dos 3 gráficos obrigatórios.
* `feature/fase-2-data-prep`: Remoção de duplicatas, imputação por mediana e boxplots.
* `feature/fase-3-feature-engineering`: Criação da coluna `gasto_medio_mensal`.
* `feature/fase-4-split-balanceamento`: Divisão 80/20 com `stratify` e SMOTE no treino.
* `feature/fase-5-escalonamento`: Padronização `StandardScaler` exclusiva para KNN.
* `feature/fase-6-7-modelagem-avaliacao`: Ajuste de hiperparâmetros, gráficos e veredito.

---

## 📊 Resultados e Diagnóstico de Overfitting

| Modelo | Parâmetro | Acurácia Treino | Acurácia Teste | Diagnóstico Técnico |
| :--- | :--- | :---: | :---: | :--- |
| **KNN** | $K = 3$ | 83,98% | 70,55% | Sensível a ruídos locais pontuais |
| **KNN** | $K = 5$ | 80,57% | 71,04% | Ajuste intermediário |
| **KNN** | $K = 7$ | 78,96% | 71,54% | Maior estabilidade na família KNN |
| **Árvore de Decisão** | `max_depth = None` | **99,77%** | **67,21%** | **Overfitting Severo (queda de 32,5% no teste)** |
| **Árvore de Decisão** | `max_depth = 5` | 76,42% | 71,75% | Poda moderada |
| **Árvore de Decisão** | **`max_depth = 3`** | **71,97%** | **75,09%** | **Melhor Capacidade de Generalização** |

---

## 🏆 Veredito Final de Negócio

O modelo adotado como solução final é a **Árvore de Decisão com `max_depth=3`**:

1. **Superioridade Métrica**: Obteve a maior acurácia no conjunto de teste (**75,09%**), superando o melhor KNN (**71,54%**).
2. **Interpretabilidade Operacional (White-Box)**: Permite que analistas de marketing extraiam regras claras de decisão (ex.: contratos mensais nos primeiros meses de assinatura) para acionamento de réguas de CRM.
3. **Escalabilidade em Produção**: Complexidade de inferência $\mathcal{O}(\text{profundidade})$ que executa em frações de milissegundo, enquanto o KNN exigiria varredura e cálculo de distâncias sobre toda a base histórica.

---

## 🚀 Propostas de Melhorias Futuras

* Implementação de modelos ensemble baseados em árvores (*Random Forest*, *XGBoost* ou *LightGBM*).
* Otimização da fronteira de decisão via curva ROC-AUC e priorização de *Recall* para mitigar Falsos Negativos (clientes em risco não identificados).
* Construção de uma API de inferência em tempo real utilizando FastAPI e integração contínua (CI/CD).

---

## 📹 Vídeo Demonstrativo
O vídeo explicativo abordando o funcionamento, justificativas técnicas e organização do repositório pode ser acessado através do link:
* **Google Drive**: [Link do Vídeo Demonstrativo](https://drive.google.com/drive)
