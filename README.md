# Embeddings e Redes Neurais - PLN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/PLN_Atividade_03_Embeddings/blob/main/03_Embbedings.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Este repositório contém a resolução dos exercícios práticos de **Embeddings** para análise de sentimentos no dataset IMDB, desenvolvidos para a disciplina de **Processamento de Linguagem Natural (PLN)**.

---

## Informações Acadêmicas

* **Curso:** Ciência de Dados (5º Semestre)
* **Disciplina:** Processamento de Linguagem Natural (PLN)
* **Professor:** Leonardo de Lellis Rossi
* **Aluno:** Rodolfo Vinicius Cima Takemoto

---

## Estrutura do Notebook

O notebook [`03_Embbedings.ipynb`](03_Embbedings.ipynb) aborda o conceito de representação vetorial densa de palavras e demonstra a **equivalência matemática e prática** entre a abordagem clássica de Bag of Words (histograma) e a camada de Embedding do PyTorch:

### 1. Preparação de Dados e Vocabulário
* **Dataset IMDB (`aclImdb`)**: Divisão em conjuntos de treino (80%), validação (20%) e teste (25.000 amostras).
* **Tokenização e Pré-processamento**: Limpeza de tags HTML (`BeautifulSoup`), remoção de pontuações/números com expressões regulares (`re`) e tokenização em minúsculas.
* **Vocabulário Personalizado (`SimpleVocab`)**: Estrutura eficiente sem dependência do `torchtext` legado, mapeando tokens para índices inteiros com tratamento de tokens desconhecidos (`default_index`).

### 2. Dataset Customizado PyTorch (`Ex3_ds`)
* **Modo `bow_hist`**: Gera o histograma de frequência de termos no vocabulário (`torch.zeros(len(vocab_train)+1)`).
* **Modo `emb`**: Gera tensores de índices de tokens com tamanho fixo (`max_pad = 200`), aplicando *padding* (`pad_idx = len(vocab_train)+1`) para sentenças menores.

### 3. Modelagem Neural e Equivalência (`Ex3_model`)
* **Modelo BoW Histograma**:
  * Primeira camada linear `torch.nn.Linear(input, hidden, bias=False)`.
  * Segunda camada linear `torch.nn.Linear(hidden, 2, bias=False)`.
* **Modelo Embedding**:
  * Camada de embedding `torch.nn.Embedding(input, hidden, padding_idx=input-1)`.
  * *Pooling* por soma (`x.sum(dim=1)`) ao longo da dimensão da sequência.
  * Segunda camada linear `torch.nn.Linear(hidden, 2, bias=False)`.
* **Equivalência Estrita**: Cópia e transposição dos pesos da camada linear do modelo BoW para a matriz de embeddings do segundo modelo (`pad_weight.T`), associados ao vetor de padding com zeros e `padding_idx` para anulação de gradientes no preenchimento.

### 4. Treinamento, Validação e Verificação
* **Otimização**: Otimizador Adam com taxa de aprendizado de `1e-4` e função de perda `CrossEntropyLoss`.
* **Checkpointing**: Salvamento automático dos pesos do melhor modelo baseado no menor *loss* de validação (`best_model.pt`).
* **Validação Numérica (`assert`)**: Verificação formal garantindo que as perdas de treino, perdas de validação e acurácias convergem exatamente para os mesmos valores numéricos em todas as épocas:
  $$\text{Train Loss (BoW)} \equiv \text{Train Loss (Emb)} \quad \text{e} \quad \text{Val Loss (BoW)} \equiv \text{Val Loss (Emb)}$$

---

## Ferramentas Utilizadas

Em conformidade com a transparência acadêmica:

* **Ferramenta:** Google Colab / Antigravity IDE
  * **Utilização:** Ambiente de desenvolvimento, execução dos loops de treinamento, testes de equivalência e edição do notebook.
* **Ferramenta:** Gemini 3.7 Flash (High)
  * **Utilização:** Apoio conceitual na formulação matemática da equivalência entre soma de embeddings e multiplicação linear por Bag-of-Words/Histograma, verificação de indexação de padding (`padding_idx`) e depuração de gradientes.
* **Ferramenta:** GitHub MCP Server
  * **Utilização:** Criação e sincronização automatizada do repositório no GitHub.

---

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/aj1no/PLN_Atividade_03_Embeddings.git
   cd PLN_Atividade_03_Embeddings
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   .\venv\Scripts\activate   # Windows
   ```

3. Instale as dependências necessárias:
   ```bash
   pip install torch numpy beautifulsoup4 matplotlib jupyter
   ```

4. Inicie o Jupyter Notebook:
   ```bash
   jupyter notebook 03_Embbedings.ipynb
   ```

---

## Licença

Este projeto está sob a licença [MIT](LICENSE).
