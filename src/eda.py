import pandas as pd
import matplotlib.pyplot as plt

def run_eda():
    #ler arquivo csv
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    #remove colunas desnecessárias
    df = df.iloc[:, :2]
    #renomeia colunas
    df.columns = ['label', 'message']
    #cria coluna númerica
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    #salva o dataset limpo em um novo arquivo csv
    df.to_csv('data/sms_dataset_clean.csv', index=False)

    #calcula totais e percentuais
    total = len(df)
    qtd_ham = (df['label'] == 'ham').sum()
    qtd_spam = (df['label'] == 'spam').sum()
    pct_spam = 100 * qtd_spam / total
    #exibe no terminal
    print(f'Total: {total} | Ham: {qtd_ham} | Spam: {qtd_spam} ({pct_spam:.1f}%)')
    #cria coluna com número de palavras
    df['palavras'] = df['message'].apply(lambda x: len(x.split()))
    #exibe estatísticas detalhadas
    print(df.groupby('label')['palavras'].describe())
    #gráfico de pizza
    df['label'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Ham','Spam'], colors=['green','red'])
    plt.title('Proporção de mensagens')
    plt.ylabel('')
    plt.savefig('outputs/pizza_classes.png')
    plt.show()
    #boxplot
    df.boxplot(column='palavras', by='label')
    plt.title('Número de palavras por classe')
    plt.suptitle('')
    plt.xlabel('Classe')
    plt.ylabel('Palavras')
    plt.savefig('outputs/boxplot_palavras.png')
    plt.show()

if __name__ == "__main__":
    run_eda()