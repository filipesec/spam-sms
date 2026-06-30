# SMS Spam
Trabalho de Reconhecimento de Padrões para classificação de mensagens SMS em spam e não spam utilizando Logistic Regression

## Estrutura do Projeto

- eda.py -> **Etapa 1**
- processing.py -> **Etapa 2**
- train_model.py -> **Etapa 3**
- evaluation.py -> **Etapa 4**

## Pré‑requisitos

- Ambiente virtual Python (`venv`)
- pandas
- matplotlib
- scikit-learn

## Para Instalar Dependências:

- pip install -r requirements.txt

## Etapa 1 (concluída)

- Dataset SMS Spam Collection baixado e armazenado em `data/spam.csv`. (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Análise exploratória inicial realizada:
  - Criação da coluna numérica `label_num` (0 = ham, 1 = spam).
  - Estatísticas geradas: total de mensagens, distribuição ham/spam, média de palavras por classe.
  - Dataset organizado salvo em `data/sms_dataset_clean.csv`.
  - Dois gráficos salvos em `outputs/`: proporção das classes e boxplot de palavras.
- Código em `src/eda.py`, executado automaticamente via `main.py`.

A **Etapa 2** deve utilizar `data/sms_dataset_clean.csv`.

## Etapa 2 (concluída)

- Utilização do dataset gerado na etapa 1: `data/sms_dataset_clean.csv`.
- Pré-processamento dos textos realizado:
  - Conversão das mensagens para letras minúsculas.
  - Remoção de links.
  - Remoção de números.
  - Remoção de caracteres especiais.
- Representação dos textos utilizando TF-IDF.
- Divisão do dataset em conjuntos de treinamento e teste utilizando `train_test_split` com estratificação das classes.
- Código implementado em `src/processing.py`.

## Etapa 3 (concluída)

- Treinamento do modelo a partir do conjunto de dados de treinamento definido na etapa 2: `src/processing.py`.
- Importação do conjunto de dados da função run_processing do arquivo `src/processing.py` para uso na Regressão Logística.
- Aplicação da Regressão Logística utilizando a biblioteca scikit-learn.
- Previsão do resultado da classificação da mensagem em spam ou não spam.
- Código implementado em `src/train_model.py`.