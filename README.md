## 🧠 AI Core — Knowledge Tracing Pipeline

Este projeto implementa um pipeline completo de **Knowledge Tracing (KT)** para modelar o aprendizado de estudantes ao longo do tempo.

O sistema permite:

* 📊 Processar dados educacionais (questões + respostas)
* 🔁 Construir sequências de aprendizado por aluno
* 🧠 Treinar modelos de KT (DKT e LPKT)
* 🎯 Prever desempenho futuro (por questão ou por tema)

---

# ⚙️ Requisitos

* Python 3.11

Recomendado usar pyenv:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

---

# 🛠️ Setup do ambiente

```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# 📁 Estrutura de dados esperada

Coloque os arquivos em:

```
data/raw/
  ├── answers.json
  ├── questions.json
  └── tree.json
```

---

# 🚀 PIPELINE COMPLETO (STEP BY STEP)

## 1. Preparar dataset de respostas

```bash
python scripts/prepare_answers_dataset.py
```

📌 O que faz:

* limpa dados inconsistentes
* remove valores nulos
* padroniza estrutura
* filtra usuários com poucas interações

📦 Output:

```
data/processed/dataset/answers_prepared.csv
```

---

## 2. Gerar sequências por usuário (CRÍTICO)

```bash
python scripts/build_user_sequences.py
```

📌 O que faz:

* ordena interações por tempo
* constrói sequências de aprendizado
* estrutura dados para modelos de KT

📦 Output:

```
data/processed/sequences/user_sequences.json
```

⚠️ Esse arquivo é obrigatório para TODOS os modelos

---

## 3. Gerar mapeamento skill → H2 (tema)

```bash
python scripts/build_skill_to_h2_from_raw_json.py
```

📌 O que faz:

* transforma árvore de assuntos em nível H2
* agrupa skills em temas pedagógicos

📦 Output:

```
data/processed/mappings/skill_to_h2.json
```

⚠️ Necessário apenas para modelos por tema

---

# 🤖 TREINAMENTO DOS MODELOS

## 4. DKT — Next Item

```bash
python scripts/train_eval_dkt_next_item.py
```

📌 Prediz:
👉 próxima questão que o aluno irá acertar/errar

---

## 5. DKT — H2 (tema)

```bash
python scripts/train_eval_dkt_topic_h2.py
```

📌 Prediz:
👉 domínio do aluno por tema (H2)

---

## 6. LPKT — Next Item

```bash
python scripts/train_eval_lpkt_next_item.py
```

📌 Inclui:

* tempo de resposta
* recência (Δt)
* dinâmica de aprendizado

---

## 7. LPKT — H2 (tema)

```bash
python scripts/train_eval_lpkt_topic_h2.py
```

📌 Modelo mais avançado:
👉 combina sequência + tempo + tema

---

# 📊 OUTPUTS

Os resultados são salvos automaticamente em:

```
artifacts/
  ├── runs/           # DKT (item)
  ├── runs_h2/        # DKT (tema)
  ├── runs_lpkt/      # LPKT (item)
  └── runs_lpkt_h2/   # LPKT (tema)
```

Cada execução contém:

```
model.pt
metrics.json
```

---

# 🧪 Validação do dataset (opcional)

```bash
python scripts/test_kt_dataset.py
```

📌 Verifica:

* estrutura do dataset
* shapes dos tensores
* integridade dos dados

---

# ⚠️ Observações importantes

* `user_sequences.json` é obrigatório para qualquer treinamento
* `skill_to_h2.json` é obrigatório apenas para modelos H2
* Use Python 3.11 para evitar incompatibilidades
* Dataset pequeno → alta variância nos resultados
* Dataset grande → maior estabilidade

---

# 🧠 Conceitos importantes

## Knowledge Tracing (KT)

Tarefa de modelar o **estado de conhecimento do aluno ao longo do tempo**, baseado em suas interações.

## DKT (Deep Knowledge Tracing)

* Usa RNN/LSTM
* Modela sequência de respostas
* Não considera tempo explicitamente

## LPKT (Learning Process KT)

* Modela aprendizado + esquecimento
* Usa tempo entre interações (Δt)
* Mais próximo do comportamento real do aluno

## H2 (nível de tema)

* Agrupamento de habilidades
* Exemplo:

  ```
  Cardiologia > Arritmias
  ```

* Reduz granularidade e melhora interpretabilidade

---

# 🎯 Objetivo do projeto

Construir um sistema de inteligência educacional capaz de:

* prever desempenho futuro
* identificar lacunas de conhecimento
* personalizar trilhas de estudo
* suportar sistemas adaptativos (Primum)

---

# 🚀 Próximos passos

* [ ] Deploy via FastAPI (inferência online)
* [ ] Integração com backend NestJS
* [ ] Recomendação adaptativa de questões
* [ ] Modelos com embeddings de conceito
* [ ] Previsão de evasão / fadiga

---

# 👨‍💻 Autor

Guilherme Desoler
Software Engineer | Primum | AI Systems
