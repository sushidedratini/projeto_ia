# Análise de Corridas de Cavalos com IA

Este projeto utiliza um modelo de aprendizado de máquina para prever o vencedor em corridas de cavalo com base em vídeos. O programa realiza o rastreamento dos montadores, extrai características importantes e treina um modelo de classificação para identificar o vencedor.

## Instalação

Para executar este projeto, recomendamos a criação de um ambiente Conda utilizando o arquivo `projeto_ia.yml`, que já contém todas as dependências necessárias.

### Passos para criar o ambiente Conda:

1. **Instale o Conda**:
    - Caso ainda não tenha o Conda instalado, siga as instruções [neste link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Clone o repositório**:
    - Clone este repositório GitHub na sua máquina local.
    ```bash
    git clone https://github.com/usuario/projeto_ia.git
    cd projeto_ia

    ```

3. **Crie o ambiente conda**:
    - Utilize o arquivo projeto_ia.yml para criar um novo ambiente Conda.
    ```bash
    conda env create -f projeto_ia.yml
    ```

4. **Ative o ambiente**:
    - Depois de criar o ambiente, ative-o com o comando.
    ```bash
    conda activate projeto_ia
    ```

5. **Verifique a instalação**:
    - Certifique-se de que todas as dependências foram instaladas corretamente.
    ```bash
    conda list
    ```

## Estrutura do Projeto

    projeto_ia/
    ├── data/               # Diretório para armazenar dados (vídeos, CSVs, etc.)
    ├── graphs/             # Diretório onde os gráficos gerados são salvos
    ├── models/             # Diretório para armazenar os modelos treinados
    ├── src/                # Código-fonte do projeto
    │   ├── video_utils.py  # Funções de manipulação de vídeo
    │   └── model_utils.py  # Funções para treino e teste do modelo
    |── main.py             # Arquivo principal que executa o programa
    ├── projeto_ia.yml      # Arquivo de configuração do Conda
    └── README.md           # Instruções do projeto

## Etapas do Programa

O programa completo passa por várias etapas para treinar o modelo e fazer previsões com novos vídeos. Abaixo estão os passos detalhados.

1. **Processamento de Vídeos**
    O vídeo da corrida é processado utilizando a biblioteca YOLOv8 para detectar montadores e cavalos.
    As mudanças de câmera são detectadas, e os dados de rastreamento (centroides) são armazenados em um dicionário por cena e por rider_id.
    Os centroides (x e y) de cada montador são extraídos em listas que descrevem o movimento do montador ao longo do tempo.

    <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGt1YzdwZzFvanlhcXowcmhkM2pxZTUzOHpuZjd0c2F6Mm13Y2duMiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/mW1BA2YBfGvMdlwoIo/giphy-downsized.gif" width="250" />

    <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMTdkbzNlM2s1YmE4NWppMnh5NTA1MDJ3bjVkcWVvMms2OXByOWs5bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XSXPcEnS2CvVXQrZbD/giphy-downsized.gif" width="250" />

2. **Preenchimento de Centroides**
    Durante o rastreamento, algumas listas de centroides podem ser menores que outras. Uma função é utilizada para preencher essas listas com base na variação média de movimento, considerando a fadiga do cavalo, para manter a consistência de tamanho entre as listas.

3. **Treinamento do Modelo**
    - Os dados processados são organizados em um DataFrame, onde cada montador tem uma lista de valores de Y (posição vertical) ao longo do tempo.
    - Para cada montador em uma corrida, os seguintes atributos são extraídos da lista de Y:
        1. y_mean: Média dos valores de Y.
        2. y_std: Desvio padrão dos valores de Y.
        3. y_min: Valor mínimo de Y.
        5. y_max: Valor máximo de Y.
    - Estes atributos são utilizados como features para treinar um modelo de classificação. O modelo Random Forest é utilizado para prever o vencedor (winner = 1).

4. **Teste e Validação**
Após o treinamento, o modelo é avaliado utilizando o conjunto de dados de teste, e métricas como accuracy e ROC AUC são calculadas. Para verificar a performance do modelo, gráficos de ROC Curve são gerados para visualizar o desempenho em termos de separação entre as classes (vencedores e não-vencedores).

5. **Previsão em Novos Vídeos**
O modelo treinado é utilizado para prever o vencedor em novos vídeos. Os vídeos são processados da mesma forma, extraindo os centroides e características dos montadores.
As previsões do modelo são feitas e salvas em um arquivo CSV.

6. **Geração de Gráficos**
    - Gráficos de movimento (Y e X) de cada montador ao longo do tempo são gerados e salvos no diretório graphs/.
    - Gráficos da ROC Curve são gerados para visualizar a performance do modelo em separar os vencedores dos não-vencedores.


## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias e correções.

## Licença

Este projeto está licenciado sob os termos da MIT License.

### Instruções Importantes:

- **Conda Environment**: O arquivo `projeto_ia.yml` define todas as dependências necessárias para o projeto, incluindo bibliotecas como `torch`, `scikit-learn`, `pandas`, `numpy`, `moviepy`, etc.
- **Data Preprocessing**: O foco do programa é extrair centroides dos vídeos e utilizá-los como features para o modelo de aprendizado de máquina. Certifique-se de que os vídeos estejam no formato correto no diretório `data/`.

Essa estrutura de README fornece uma visão clara do fluxo de trabalho e instruções detalhadas para configurar o ambiente e rodar o programa.