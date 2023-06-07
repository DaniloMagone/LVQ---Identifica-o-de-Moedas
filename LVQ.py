import numpy as np
import random
import csv
t = []
init = np.zeros((5, 5),dtype=float)
weights = np.zeros((5, 5),dtype=float)

#descarrega o arquivo csv
def load_data(arquivo):
 entries = []
 with open(arquivo, 'r', newline='') as fd:
        reader = csv.reader(fd)
        for row in reader:
            values = [float(x) for x in row]
            entries.append(values)
 return np.array(entries,dtype=float)
 
#normaliza os dados
def normalize_data(data):
  transpose_data = data.T
  for i in range(data.shape[1]):
    transpose_data[i]=(transpose_data[i] - transpose_data[i].min())/(transpose_data[i].max()-transpose_data[i].min())
  return transpose_data.T

#define o neurônio vencedor com base na menor distância euclidiana
def choose_winner(inputs):
    auxiliar=[]
    for i in range(5):
      auxiliar.append(np.linalg.norm(np.subtract(inputs,weights[i])))
    yestrela=auxiliar.index(min(auxiliar))
    return yestrela

#treina a RNA
def train(dataset,alpha,alpha2,epochs):
  global weights
  global t
  global init
  alpha_step=(alpha2-alpha)/epochs
  while (alpha > 0.001):
    #Quando acerta, soma-se aos pesos a diferença entre os dados de treinamento e o valor do peso atual, multiplicado pelo alpha. 
    for i in range(dataset.shape[0]):
      idx=choose_winner(dataset[i])
      if ((idx+1)==t[i]):
        np.add(weights[idx],alpha*np.subtract(dataset[i],weights[idx]), out=weights[idx])
        #Quando erra, substrai-se dos pesos a diferença entre os dados de treinamento e o valor do peso atual, multiplicado pelo alpha.
      else:
        np.subtract(weights[idx],alpha*np.subtract(dataset[i],weights[idx]), out=weights[idx])
    alpha+=alpha_step

#escreve no csv os pesos finais
def write_weights(filename):
  global weights
  np.savetxt(filename, weights, delimiter=",")

#roda o teste 
def run_test(dataset):
  global init
  results=[]
  for i in range(dataset.shape[0]):
    x=dataset[i]
    idx=choose_winner(x)+1
    results.append(idx)
  return results

dataset=load_data("training.csv")
init=load_data("init.csv")
#Aqui unimos os dados iniciais com o dataset de treino para normalizar todos juntos.
full_data=np.concatenate((init,dataset))
t=full_data.T[5]
#normaliza os dados deixando de fora a variável target
full_data_normalized = np.insert(normalize_data(full_data[:,:5]), 5,t, axis=1)
test=np.empty([20, 5])
targets_test=[]

#Retira 20 amostras aleatórias para teste do dataset
for x in range(test.shape[0]):
  i  = random.randint(4, full_data_normalized.shape[0]) -1
  test[x, :]=full_data_normalized[i,:5]
  #a variável target do teste é colocada na variável targets_test
  targets_test.append(full_data_normalized[i,5])
  full_data_normalized=np.delete(full_data_normalized, i,axis=0)

#Separa novamente os dados inciais dos pesos e o dataset de treino, deixando de fora a variável target
dataset=full_data_normalized[5:,:5]
init=full_data_normalized[:5,:5]
weights=init
train(dataset,0.1,0.001,100)
write_weights("weights.csv")
results=run_test(test)
cont=0

#Calcula a porcentagem de acerto
for i in range(len(results)):
  if results[i]==targets_test[i]:
    cont+=1
print(results)
print(targets_test)
print(cont/len(results)*100)
