import numpy as np
import pandas as pd


def simulate_students(
    q_matrix,
    n_students=1000,
    min_interactions=50,
    max_interactions=200,
    seed=42,
):
    np.random.seed(seed)

    n_questions, n_concepts = q_matrix.shape

    data = []

    for student_id in range(n_students):
        n_inter = np.random.randint(min_interactions, max_interactions)

        mastery = np.random.beta(2, 5, size=n_concepts)  # começa baixo
        timestamp = 0

        for t in range(n_inter):
            q = np.random.randint(0, n_questions)
            concepts = q_matrix[q].nonzero()[1]

            if len(concepts) == 0:
                continue

            # dificuldade média da questão
            difficulty = np.random.normal(0, 0.5)

            # habilidade média nos conceitos
            skill = mastery[concepts].mean()

            p_correct = 1 / (1 + np.exp(-(skill - difficulty)))

            is_correct = np.random.rand() < p_correct

            data.append({
                "student_id": student_id,
                "question_id": q,
                "is_correct": int(is_correct),
                "timestamp": timestamp,
            })

            # atualização de aprendizado
            lr = 0.05
            if is_correct:
                mastery[concepts] += lr * (1 - mastery[concepts])
            else:
                mastery[concepts] -= lr * mastery[concepts]

            # esquecimento leve
            mastery *= 0.999

            timestamp += np.random.randint(30, 300)

    return pd.DataFrame(data)