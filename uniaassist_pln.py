"""UniAssist: comparação simples entre regras e Machine Learning.

Coloque este arquivo junto de mensagens.csv e desafios.csv do material.
Instale: python -m pip install -r requirements.txt
Execute: python uniaassist_pln.py
O programa apenas lê os CSVs e mostra os resultados no terminal.
"""
import csv
from collections import Counter
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PASTA = Path(__file__).resolve().parent
REVISAO = "Revisão humana"


def ler_dataset(arquivo):
    with (PASTA / arquivo).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classificar_regras(texto):
    """Conta termos distintos presentes; empate ou zero exige revisão."""
    regras = {
        "Financeiro": ["boleto", "mensalidade", "cobrança", "pagamento", "fatura"],
        "Matrícula": ["matrícula", "rematrícula", "disciplina", "turma", "trancamento"],
        "Estágio": ["estágio", "termo", "supervisor", "concedente", "convênio"],
        "Suporte": ["senha", "login", "portal", "acesso", "aplicativo", "erro"],
    }
    texto = texto.casefold()
    # Busca por trecho, como baseline simples: não interpreta o contexto.
    pontos = {
        categoria: sum(termo in texto for termo in termos)
        for categoria, termos in regras.items()
    }
    maior = max(pontos.values())
    vencedores = [categoria for categoria, valor in pontos.items() if valor == maior]
    if maior == 0 or len(vencedores) > 1:
        return REVISAO
    return vencedores[0]


def criar_modelo(ngram_range):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, lowercase=True)),
        ("classificador", LinearSVC(random_state=42)),
    ])


def avaliar(nome, reais, previstos, classes):
    print(f"\n=== {nome} ===")
    print(f"Acurácia: {accuracy_score(reais, previstos):.2%}")
    print(classification_report(reais, previstos, labels=classes,
                                digits=4, zero_division=0))
    print("Matriz: linhas = categoria real; colunas = previsão.")
    print("Ordem:", " | ".join(classes))
    print(confusion_matrix(reais, previstos, labels=classes))


def main():
    registros = ler_dataset("mensagens.csv")
    textos = [r["texto"] for r in registros]
    rotulos = [r["categoria"] for r in registros]
    classes = sorted(set(rotulos))
    print("scikit-learn:", sklearn.__version__)
    print("Total por categoria:", dict(sorted(Counter(rotulos).items())))

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        textos, rotulos, test_size=0.25, random_state=42, stratify=rotulos
    )
    print(f"Treino: {len(x_treino)} | Teste: {len(x_teste)}")
    print("Treino por categoria:", dict(sorted(Counter(y_treino).items())))
    print("Teste por categoria:", dict(sorted(Counter(y_teste).items())))

    # O TF-IDF aprende o vocabulário somente nos dados de treinamento.
    modelo = criar_modelo((1, 2))
    modelo.fit(x_treino, y_treino)
    previstos_ml = modelo.predict(x_teste)
    avaliar("TF-IDF + LinearSVC (unigramas e bigramas)", y_teste, previstos_ml, classes)

    previstos_regras = [classificar_regras(texto) for texto in x_teste]
    # A coluna adicional evita esconder os casos enviados para revisão.
    avaliar("Regras", y_teste, previstos_regras, classes + [REVISAO])
    revisoes = previstos_regras.count(REVISAO)
    print(f"Revisões: {revisoes}/{len(y_teste)} ({revisoes / len(y_teste):.2%})")
    print("Revisão humana conta como não acerto automatizado.")

    print("\n=== Erros e revisões no teste ===")
    for texto, real, regra, ml in zip(x_teste, y_teste, previstos_regras, previstos_ml):
        if regra != real or ml != real:
            print(f"{texto}\n  Real: {real} | Regras: {regra} | ML: {ml}")

    # Experimento solicitado: mantém exatamente a mesma divisão.
    unigramas = criar_modelo((1, 1))
    unigramas.fit(x_treino, y_treino)
    avaliar("Experimento: apenas unigramas", y_teste, unigramas.predict(x_teste), classes)
    print("Experimento exploratório: o teste não deve servir para escolher parâmetros.")

    print("\n=== Oito desafios inéditos ===")
    desafios = ler_dataset("desafios.csv")
    resultados = modelo.predict([r["texto"] for r in desafios])
    for registro, previsto in zip(desafios, resultados):
        print(f"{registro['id']}: {registro['texto']}")
        print(f"  Regras: {classificar_regras(registro['texto'])} | ML: {previsto}")
    print("Os desafios não têm rótulos oficiais; não calculamos acurácia para eles.")


if __name__ == "__main__":
    main()
