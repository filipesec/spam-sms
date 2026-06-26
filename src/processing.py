import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def limpar_texto(texto):
    texto = texto.lower()

    # remove links
    texto = re.sub(r'http\S+', '', texto)
    # remove numeros
    texto = re.sub(r'\d+', '', texto)
    # remove caracteres especiais
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    return texto

def run_processing():

    # dataset gerado pela etapa 1
    df = pd.read_csv('data/sms_dataset_clean.csv')
    # limpeza
    df['message_clean'] = df['message'].apply(limpar_texto)
    # verificar resultado da limpeza
    print(df[['message', 'message_clean']].head())

    X = df['message_clean']
    y = df['label_num']

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000
    )

    X_tfidf = vectorizer.fit_transform(X)

    # treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n")
    print(f'Treino: {X_train.shape}')
    print(f'Teste: {X_test.shape}')
    print("\nPROCESSAMENTO CONCLUÍDO COM SUCESSO!")

    return X_train, X_test, y_train, y_test, vectorizer

if __name__ == "__main__":
    run_processing()