import math;
import numpy as np
#import matplotlib.pyplot as plt

def media(var):
    sum = 0
    for i in var:
        sum += i
    return sum/len(var)
	
def media_error(o1, o2, var1, var2):
    sum = 0
    for i in range(len(var1)):
        sum += (h(o1,o2,var1[i]) - var2[i])
    return sum/len(var1)	

def var1_derivative(o1,o2,var1,var2):
    return 2*media_error(o1, o2, var1, var2)

def var2_derivative(o1,o2,var1,var2):
    sum = 0
    for i in range(len(var1)):
        sum += (h(o1,o2,var1[i]) - var2[i]) * var1[i]
    return (sum*2)/len(var1)

def taxa_aprendizado(o1, o2, var1, var2, alpha, der):
    if der== 0:
        return o1 - alpha*(var1_derivative(o1, o2, var1, var2))
    elif der == 1:
        return o2 - alpha*(var2_derivative(o1, o2, var1, var2))
    else:
        print("ERROR")
        return 1

def h(o1,o2,x):
	return o1 + x*o2

def quadratic_error(o1,o2, var1, var2):
    sum = 0;
    for i in range(len(var1)):
	    sum += pow((h(o1,o2,var1[i]) - var2[i]), 2);
    return sum/len(var1);

def normalize(col):
    new_col = [];
    for i in range(len(col)):
	    new_col.append((col[i] - min(col)) / (max(col) - min(col)));
    return new_col;

def compute_cost(theta_0, theta_1, x, y):
    """
    Calcula o erro quadratico medio
    
    Args:
        theta_0 (float): intercepto da reta 
        theta_1 (float): inclinacao da reta
        data (np.array): matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    
    Retorna:
        float: o erro quadratico medio
    """
    total_cost = 0
	
    total_cost = quadratic_error(theta_0, theta_1, x, y);
	
    return total_cost;
	
def step_gradient(theta_0_current, theta_1_current, x, y, alpha):
    """Calcula um passo em direção ao EQM mínimo
    
    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo 
    
    Retorna:
        tupla: (theta_0, theta_1) os novos valores de theta_0, theta_1
    """
    
    theta_0_updated = 0;
    theta_1_updated = 0;
    
    theta_0_updated = taxa_aprendizado(theta_0_current, theta_1_current,x,y,alpha,0);
    theta_1_updated = taxa_aprendizado(theta_0_current, theta_1_current,x,y,alpha,1);

    return theta_0_updated, theta_1_updated;
	
def gradient_descent(x, y, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    """executa a descida do gradiente
    
    Args:
        data (np.array): dados de treinamento, x na coluna 0 e y na coluna 1
        starting_theta_0 (float): valor inicial de theta0 
        starting_theta_1 (float): valor inicial de theta1
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar
    
    Retorna:
        list : os primeiros dois parâmetros são o Theta0 e Theta1, que armazena o melhor ajuste da curva. O terceiro e quarto parâmetro, são vetores com o histórico dos valores para Theta0 e Theta1.
    """

    # valores iniciais
    theta_0 = starting_theta_0;
    theta_1 = starting_theta_1;

    
    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = [];
    
    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = [];
    theta_1_progress = [];
    
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, x, y));
        theta_0, theta_1 = step_gradient(theta_0, theta_1, x, y, alpha=learning_rate);
        theta_0_progress.append(theta_0);
        theta_1_progress.append(theta_1);
        
    return [theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress];

data = np.genfromtxt('house_prices_train.csv', delimiter=',')

#Extrair colunas para análise
OverallQual = np.array(data[1:,17]);
GrLivArea = np.array(data[1:,46]);
SalePrice = np.array(data[1:,80]);

GrLivArea = normalize(GrLivArea);

theta0, theta1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent(GrLivArea,SalePrice,0,0,0.1,100);

print(theta0);
print(theta1);
print(media(cost_graph));

#Gráfico dos dados
#plt.figure(figsize=(10, 6))
#plt.scatter(x,y)
#plt.xlabel('Horas de estudo')
#plt.ylabel('Nota final')
#plt.title('Dados')
#plt.show()

# dataset copiado do Quiz de Otimizacao Continua
other_data = np.array([
                 [1, 3],
                 [2, 4],
                 [3, 4], 
                 [4, 2]
             ])

new_theta0, new_theta1 = step_gradient(1, 1, other_data[:,0], other_data[:,1], alpha=0.1)
# comparacao de floats com tolerancia 1E-11
if abs(new_theta0 - 0.95) < 1E-11:
  print("Atualizacao de theta0 OK")
else:
  print("ERRO NA ATUALIZACAO DE theta0!")

if abs(new_theta1 - 0.55) < 1E-11:
  print("Atualizacao de theta1 OK")
else:
  print("ERRO NA ATUALIZACAO DE theta1!")

print();

theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent(other_data[:,0], other_data[:,1], starting_theta_0=0, starting_theta_1=0, learning_rate=0.1, num_iterations=100000)

#Imprimir parâmetros otimizados
print ('Theta_0 otimizado: ', theta_0)
print ('Theta_1 otimizado: ', theta_1)

#Imprimir erro com os parâmetros otimizados
print ('Custo minimizado: ', compute_cost(theta_0, theta_1, other_data[:,0], other_data[:,1]))
