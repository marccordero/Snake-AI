import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Linear_QNet_Simple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet_Simple, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def guardar(self, nombre_archivo='modelo_simple.pth'):
        carpeta_modelo = './modelo'
        if not os.path.exists(carpeta_modelo):
            os.makedirs(carpeta_modelo)
        nombre_archivo = os.path.join(carpeta_modelo, nombre_archivo)
        torch.save(self.state_dict(), nombre_archivo)

class Linear_QNet_Moderado(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Linear_QNet_Moderado, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def guardar(self, nombre_archivo='modelo_moderado.pth'):
        carpeta_modelo = './modelo'
        if not os.path.exists(carpeta_modelo):
            os.makedirs(carpeta_modelo)
        nombre_archivo = os.path.join(carpeta_modelo, nombre_archivo)
        torch.save(self.state_dict(), nombre_archivo)

class Linear_QNet_Complejo(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Linear_QNet_Complejo, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def guardar(self, nombre_archivo='modelo_complejo.pth'):
        carpeta_modelo = './modelo'
        if not os.path.exists(carpeta_modelo):
            os.makedirs(carpeta_modelo)
        nombre_archivo = os.path.join(carpeta_modelo, nombre_archivo)
        torch.save(self.state_dict(), nombre_archivo)

class QTrainer:
    def __init__(self, modelo, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.modelo = modelo
        self.optimizador = optim.Adam(modelo.parameters(), lr=self.lr)
        self.criterio = nn.MSELoss()

    def paso_entrenamiento(self, estado, accion, recompensa, siguiente_estado, hecho):
        estado = torch.tensor(estado, dtype=torch.float)
        siguiente_estado = torch.tensor(siguiente_estado, dtype=torch.float)
        accion = torch.tensor(accion, dtype=torch.long)
        recompensa = torch.tensor(recompensa, dtype=torch.float)

        if len(estado.shape) == 1:
            estado = torch.unsqueeze(estado, 0)
            siguiente_estado = torch.unsqueeze(siguiente_estado, 0)
            accion = torch.unsqueeze(accion, 0)
            recompensa = torch.unsqueeze(recompensa, 0)
            hecho = (hecho,)

        pred = self.modelo(estado)

        objetivo = pred.clone()
        for idx in range(len(hecho)):
            Q_nuevo = recompensa[idx]
            if not hecho[idx]:
                Q_nuevo = recompensa[idx] + self.gamma * torch.max(self.modelo(siguiente_estado[idx]))

            accion_idx = torch.argmax(accion[idx]).item()
            objetivo[idx][accion_idx] = Q_nuevo

        self.optimizador.zero_grad()
        perdida = self.criterio(objetivo, pred)
        perdida.backward()
        self.optimizador.step()