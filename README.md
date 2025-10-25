<div align="center">
  <img 
    alt="Logo Equipe" 
    height="250" 
    width="250" 
    src="https://github.com/JNarimatsu/assets/raw/main/Professional%20Logo%20Design%20for%20Data%20Analysis%20Group.png"
  >
</div>

# Análise de preços de laptop
### Avanti 2025.3 
### Technologies

<div style="display: inline_block"><br>
  <img align="center" alt="Python" height="30" width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" />
  <img align="center" alt="Pandas" height="30" width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg"/>
  <img align="center" alt="Numpy" height="30" width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg"/>
  <img align="center" alt="Matplotlib" height="30" width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original.svg"/>
        
 </div>
 
## Conjunto de dados
O projeto conta com uma análise exploratória e análise de comparação de modelos de machine learning no conjunto de dados [prices_laptop(preços laptop)](https://www.kaggle.com/datasets/muhammetvarl/laptop-price), conjunto de dados disponível no Kaggle. O conjunto de dados conta com informações sobre modelos, configurações, marcas e preços de laptops. Para nossa análise comparativa utilizamos os modelos de Regressão linear (Linear Regression), K-vizinhos mais próximos (K-Nearest-Neighbors), Máquinas de vetores-suporte (Support Vector Machine), Árvores de decisão (Decision Tree), com e sem Hiper-parâmetro

## Metodologia
A metodologia utilizada será a CRISP-DM, composto por:
- Entendimento de negócio
- Entendimento de dados
- Preparação dos dados
- Modelagem

# Etapas do projeto
## Dicionário de dados
Foi criado um dicionário de dados para verificar e entender as variáveis no conjunto de dados. No dicionário classificamos os dados por:  
- Variável: Nome da coluna
- Descrição da variável: Descrição da coluna
- Tipo: Quantitativa e Qualitativa
- Subtipo: Nominal, Ordinal, Discreta e Contínua

![Dicionario](https://github.com/JNarimatsu/assets/raw/main/Dict_Avanti.png)

## Análise exploratória de dados
### Apresentação de dados
 O conjunto de dados tem 1303 unidades amostrais com 13 variáveis
#### Classificação das variáveis:
- Quantitativa contínua: Inches(polegadas), Weight (Peso),  Price-euros
- Quantitativa discreta: laptop_ID
- Qualitativa nominal: Company, Product, TypeName e OpSys.
- Qualitativa ordinal: ScreenResolution, Cpu,  Ram, Memory, Gpu
### Distribuição de Variáveis Qualitativas
#### Variável Company e Product
![Distribuicao01](https://github.com/JNarimatsu/assets/raw/main/Dist_company_Product.png)
- As três fabricantes mais frequentes são Dell(297), Lenovo(297) e HP (274);
- Dos 10 modelos mais frequentes, 06 são da fabricante Dell sendo eles: XPS 13 (30), Inspiron 3567 (29), Vostro 3568 (19), Inspiron 5570 (18),  Alienware 17 (15), Inspiron 5567 (14);
#### Variável TypeName e ScreenResolution
![Distribuicao02](https://github.com/JNarimatsu/assets/raw/main/Dist_Type_Screen.png)
 - Notebooks Aparecem com maior frequencia entre os tipos de laptops
 - A resolução de tela mais frequencia é  `Full HD 1920x1080`
### Distribuição de Variáveis Quantitativas
#### Variável Inches
![inches](https://github.com/JNarimatsu/assets/raw/main/Inches.png)
- Os tamanhos de tela mais frequentes estão entre 14" e 15"
- Temos menos frequência entre telas de 10" a 12" e 18"
#### Variável Weight
![weight](https://github.com/JNarimatsu/assets/blob/main/Weight.png?raw=true)
- A maior parte dos pesos estão entre 1.5 e 2.0
- A partir de 3.5 até 4.5 a frequência é menor
#### Variável Prices_euros
![prices](https://github.com/JNarimatsu/assets/blob/main/Prices_euros.png?raw=true)
- Os preços mais comuns estão abaixo de 2000.00 euros
- A partir de 3000.00 euros a frenquencia é menor e vai dimuindo até o preço maximo, de pouco mais de 6000.00 euros, o que foi interpretado como outliers.
- As modas para essa variável são 1099.0, 1499.0 e 1799.0 euros.
### Matriz de correlação
- Existe uma forte relação entre as variáveis `inches` e `weight`(~0.8-0.9) o que faz  sentido pois quanto mais a tela, mais pesado tende a ser o laptop.
- A variável `weigth` e a variável `princes_euros`tem uma correlação moderada (~0.3-0.4) indicando que laptop mais pesados tendem a ser um pouco mais caros , mas não é uma relação forte.
- A relação entre as variáveis `inches` e `princes_euros` tem uma correlação fraca (~0.1-0.2) Tamanho da tela quase não explica o preço diretamente.
![matriz_correlacao](https://github.com/JNarimatsu/assets/raw/main/Matriz.png)

### Dispersão preço em relação a tamanho da tela (Prices_euros x Inches)
- Laptops com telas maiores tendem a aparecer em faixas de preço mais altas.   Mas a relação não é forte: há laptops de 13" e 14" com preços bem altos, e laptops grandes que não são tão caros.
- Para cada tamanho de tela, os preços variam bastante. Isso sugere que outros fatores (processador, RAM, marca, placa de vídeo, etc.) são mais determinantes no preço do que apenas o tamanho da tela.
- Existem alguns notebooks com preços muito acima da média (acima de 5000 euros). Esses provavelmente são modelos premium/gamer ou estações de trabalho.
![dispercao](https://github.com/JNarimatsu/assets/raw/main/Dispers%C3%A3o%20preco.png)
### Box Plot de Price_euros por Cpu_Base (Ordenado por Tipo e Geração)
![boxplot01](https://github.com/JNarimatsu/assets/blob/main/Boxplot_tipo_geracao.png)
- Conseguimos verificar que os modelos geram impacto no preço do laptop, mas nos processadores Intel core, precisamos verificar, pois deve ter outros requeisitos de configuração sendo levados em consideração já que conseguimos verificar outliers.
- Temos laptops com processadores i7 sendo os que apresentam preços mais elevados.
- Intel Xeon é em sua maioria de workstation, isso pois se trata de um processador muito utilizado para Servidores.
### Preço médio por categorias
#### Análise de preços médio por marca
![Medio_company](https://github.com/JNarimatsu/assets/raw/main/Preco_medio_company.png)
 - Conseguimos verificar que a marca razer tem o preço médio mais elevado, isso pode ocorrer pois a marca é especializada em laptops gamer, com configurações mais robustas, o que como vimos analisando as cpu pode aumentar os preços.
 - Segunda marca com maior preço médio é a LG, iremos fazer mais verificações que possam explicar esse comportamento.
 - Temos a maioria das marcar com preços na faixa dos 1000.00 euros.
 - Existem cinco marcas quem tem seus preços médio abaixo do 800.00 euros, tmabém iremos verificar.

#### Análise de preços médios por tipo de computador
![Medio_type](https://github.com/JNarimatsu/assets/blob/main/Preco_medio_Type.png)
 - Worstation tem a média de preços mais elevados. Não consegui verificar se o workstation citado no dataset, se trata do computador completo com gabinete e monitor.
 - Temos `Gaming`, `Ultrabook` e `2 in 1 convertible` com preços muito próximos, isso pode ocorrer pois estamos falando de laptop com construções e configurações que tendem a ser mais roubustas visando públicos específicos e com disposição financeira.
 - Notebook e netbooks tem preço médio abaixo de 1000.00 euros.
![Medio_OS](https://github.com/JNarimatsu/assets/blob/main/Preco_medio_OS.png)

### Análise Multivariada
![Multivariada](https://github.com/JNarimatsu/assets/blob/main/Gr%C3%A1fico%20de%20barras%20agrupadas%20para%20Company%2C%20TypeName%20e%20Price_euros.png)
Análise de preços médio por por empresa e tipo de laptop
 - Observamos que as empresas que vendem `Workstation` tem seus preços médios maiores nessa categoria, como é verificado em empresas como `Dell`, `HP` e `Lenovo`
 - Exceto pela `HP` e `Lenovo` as demais empresas que contam com um linha Gaming, tem preços médios acima de 1500.00 euros, a `Acer` tem um valor médio abaixo para sua linha gaming, mas entre as demais linhas dessa marca a gaming é a com preço médio mais elevado.
 - `Razer` como tinhamos comentado anteriormente, tem a os computadores da sua linha gaming, com preços médio mais elevamos, iremos verificar as configurações, mas por se tratar de uma empresa com remomada nesse área isso pode gerar um preço agregado ao produto.
 - `Ultrabook` Tendem a ter preços mais elevados em praticamente todas as marcas, porém temos a `Samsung` como exceção, onde é observado preços médios com pouca variação entre `2 in 1 convertible`, `Ultrabook` e `Notebook`
[boxplotprice](https://github.com/JNarimatsu/assets/blob/main/Boxplot%20de%20Price_euros%20por%20CPU_Brand%20e%20TypeName.png)

## Sumário de insights e hipóteses
 -  `Dell`, `Lenovo` e `HP` são as marcas mais frequentes, também são as empresas que tem como carro-chefe a categoria `workstation`, que é a categoria com maior preço médio entre as demais catergorias dessas empresas. Isso pode ocorrer pois essa categoria é a mais adquirida por empresas.
  - `Workstation` Utilizam em sua maioria o processador Intel Xeon, o que explica os preços elevados, visto que se trata de um processador muito utilizado para servidores.
 - Apesar de ser a categoria mais frenquente, `Notebook` não é a categoria com valor médio elevado, tendo preço médio abaixo dos 1000.00 euros.
 - Laptop com telas maiores constumam ser mais pesados e verifcamos uma elavação nos preços, mas não é decisivo, a `CPU` também é um fator para aumento dos preços médios, com verificamos com `Intel Core i7`.
 - `Ultrabook`, `Gaming`e `2 in 1 convertible` apresentam preços médios na faixa dos 1500.00 euros a 1800.00 euros, isso ocorre pois utilizam em sua processadores `Intel Core i5` ou `Intel Core i7`, além disso, estamos falando de laptop que tendem a ter construção mais robusta.
 - A Razer é uma empresa que apresenta preços bastante elevados na sua linha `Gaming`.
 - Samsung apresenta pouca variação de preço entre seus modelos `Ultrabook`, `Notebook` e `2 in 1 convertible`
## Análise comparativa
Em resumo, o modelo KNN foi o que apresentou o melhor desempenho preditivo médio para este problema, seguido de perto pelo Linear Regression com desempenho estável e alta interpretabilidade. O modelo DTR(Decision Tree) foi o mais rápido, mas apresentou menos precisão. por fim o SVR Apresentou a mais baixa capacidade preditiva nesse contexto. Em suma, o KNN e LRG foram os modelos com melhor desempenho na validação cruzada.
![Comparativa](https://github.com/JNarimatsu/assets/raw/main/Comparativo.png)


