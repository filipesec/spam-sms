# SMS Spam
Trabalho de Reconhecimento de Padrões para classificação de mensagens SMS em spam e não spam utilizando Logistic Regression

## Estrutura do Projeto

│eda.py -> **Etapa 1**
│processing.py -> **Etapa 2**
│train_model.py -> **Etapa 3**
│evaluation.py -> **Etapa 4**

## Pré‑requisitos

- Ambiente virtual Python (`venv`)
- pandas
- matplotlib

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