from sklearn.linear_model import LogisticRegression
from processing import run_processing

# Os dados do conjunto de treinamento são importados da função run_processing para aplicá-los no LogisticRegression

# Os dados importados são executados 

def training_model():
 

 X_train, X_test, y_train, y_test, vectorizer = run_processing()

 model = LogisticRegression(max_iter=1000)
 model.fit(X_train, y_train)

 # Previsões -> Mensagem (spam/não spam)
 predictions = model.predict(X_test)
 print("\nTREINAMENTO CONCLUÍDO COM SUCESSO!")
  
 return model, X_test, y_test, predictions

if __name__ == "__main__":
    training_model()


# Para o próximo arquivo, faça: 

# from processing import run_processing, X_train, y_train, X_test
# from train_model import predictions









