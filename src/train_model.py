from sklearn.linear_model import LogisticRegression
from processing import run_processing, X_train, y_train, X_test

# Os dados do conjunto de treinamento são importados da função run_processing para aplicá-los no LogisticRegression

# Carrega o processamento e os dados dos conjuntos de treinamento e de teste
run_processing()

# Os dados importados são executados 
model = LogisticRegression()
model.fit(X_train, y_train)

# Previsões -> Mensagem (spam/não spam)
predictions = model.predict(X_test)
print(predictions)

# Para o próximo arquivo, faça: 

# from processing import run_processing, X_train, y_train, X_test
# from train_model import predictions









