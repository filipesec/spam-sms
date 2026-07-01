import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from src.train_model import training_model


def run_evaluation():
    os.makedirs("outputs", exist_ok=True)

    model, X_test, y_test, predictions = training_model()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n===== AVALIAÇÃO DO MODELO =====")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, predictions, target_names=["ham", "spam"]))

    matriz = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["ham", "spam"]
    )

    disp.plot()
    plt.title("Matriz de Confusão - Logistic Regression")
    plt.savefig("outputs/matriz_confusao.png")
    plt.show()

    metricas = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    valores = [accuracy, precision, recall, f1]

    plt.figure()
    plt.bar(metricas, valores)
    plt.ylim(0, 1)
    plt.title("Métricas de Desempenho do Modelo")
    plt.ylabel("Valor")
    plt.savefig("outputs/metricas_desempenho.png")
    plt.show()

    print("\nAVALIAÇÃO CONCLUÍDA COM SUCESSO!")


if __name__ == "__main__":
    run_evaluation()