import torch
import random
import numpy as np
from collections import deque
from juego import JuegoSerpienteAI, Direccion, Punto
from modelo import Linear_QNet_Simple, Linear_QNet_Moderado, Linear_QNet_Complejo, QTrainer
from results import plot, save_results
from constants import MAX_MEMORY, LR, BATCH_SIZE, EPISODES, input_size, hidden_size_simple, hidden_size1_moderado, hidden_size2_moderado, hidden_size1_complejo, hidden_size2_complejo, hidden_size3_complejo, output_size
import matplotlib.pyplot as plt


class Agente:

    def __init__(self):
        self.n_juegos = 0
        self.epsilon = 0 # aleatoriedad
        self.gamma = 0.9 # tasa de descuento
        self.memoria = deque(maxlen=MAX_MEMORY) # popleft()
        self.modelo_simple = Linear_QNet_Simple(input_size, hidden_size_simple, output_size)
        self.modelo_moderado = Linear_QNet_Moderado(input_size, hidden_size1_moderado, hidden_size2_moderado, output_size)
        self.modelo_complejo = Linear_QNet_Complejo(input_size, hidden_size1_complejo, hidden_size2_complejo, hidden_size3_complejo, output_size)
        self.entrenador = QTrainer(self.modelo_simple, lr=LR, gamma=self.gamma)


    def obtener_estado(self, juego):
        cabeza = juego.serpiente[0]
        punto_l = Punto(cabeza.x - 20, cabeza.y)
        punto_r = Punto(cabeza.x + 20, cabeza.y)
        punto_u = Punto(cabeza.x, cabeza.y - 20)
        punto_d = Punto(cabeza.x, cabeza.y + 20)
        
        dir_l = juego.direccion == Direccion.IZQUIERDA
        dir_r = juego.direccion == Direccion.DERECHA
        dir_u = juego.direccion == Direccion.ARRIBA
        dir_d = juego.direccion == Direccion.ABAJO

        estado = [
            # Peligro recto
            (dir_r and juego.hay_colision(punto_r)) or 
            (dir_l and juego.hay_colision(punto_l)) or 
            (dir_u and juego.hay_colision(punto_u)) or 
            (dir_d and juego.hay_colision(punto_d)),

            # Peligro a la derecha
            (dir_u and juego.hay_colision(punto_r)) or 
            (dir_d and juego.hay_colision(punto_l)) or 
            (dir_l and juego.hay_colision(punto_u)) or 
            (dir_r and juego.hay_colision(punto_d)),

            # Peligro a la izquierda
            (dir_d and juego.hay_colision(punto_r)) or 
            (dir_u and juego.hay_colision(punto_l)) or 
            (dir_r and juego.hay_colision(punto_u)) or 
            (dir_l and juego.hay_colision(punto_d)),
            
            # Dirección de movimiento
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Ubicación de la comida
            juego.comida.x < juego.cabeza.x,  # comida a la izquierda
            juego.comida.x > juego.cabeza.x,  # comida a la derecha
            juego.comida.y < juego.cabeza.y,  # comida arriba
            juego.comida.y > juego.cabeza.y  # comida abajo
            ]

        return np.array(estado, dtype=int)

    def recordar(self, estado, accion, recompensa, siguiente_estado, hecho):
        self.memoria.append((estado, accion, recompensa, siguiente_estado, hecho)) # popleft si se alcanza MAX_MEMORY

    def entrenar_memoria_larga(self):
        if len(self.memoria) > BATCH_SIZE:
            mini_muestra = random.sample(self.memoria, BATCH_SIZE) # lista de tuplas
        else:
            mini_muestra = self.memoria

        estados, acciones, recompensas, siguientes_estados, hechos = zip(*mini_muestra)
        self.entrenador.paso_entrenamiento(estados, acciones, recompensas, siguientes_estados, hechos)
        #for estado, accion, recompensa, siguiente_estado, hecho in mini_muestra:
        #    self.entrenador.entrenar_paso(estado, accion, recompensa, siguiente_estado, hecho)

    def entrenar_memoria_corta(self, estado, accion, recompensa, siguiente_estado, hecho):
        self.entrenador.paso_entrenamiento(estado, accion, recompensa, siguiente_estado, hecho)

    def obtener_accion(self, estado):
        # Movimientos aleatorios: equilibrio entre exploración / explotación
        self.epsilon = 80 - self.n_juegos
        movimiento_final = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            movimiento = random.randint(0, 2)  # Asegurarse de que el movimiento esté en [0, 2]
        else:
            estado0 = torch.tensor(estado, dtype=torch.float)
            prediccion = self.modelo_simple(estado0)
            print("Predicción:", prediccion)
            movimiento = torch.argmax(prediccion).item()
            print("Movimiento calculado antes de ajuste:", movimiento)
            movimiento = min(movimiento, len(movimiento_final) - 1)  # Asegurarse de que el movimiento no exceda el índice máximo permitido
            print("Movimiento calculado después de ajuste:", movimiento)

        movimiento_final[movimiento] = 1

        return movimiento_final


def entrenar():
    puntuaciones_grafico = []
    puntuaciones_medias_grafico = []
    puntuacion_total = 0
    record = 0
    
    agente = Agente()
    
    juego = JuegoSerpienteAI()
    for _ in range(EPISODES):
        hecho = False
        while  not hecho:
            # obtener estado anterior
            estado_anterior = agente.obtener_estado(juego)

            # obtener movimiento
            movimiento_final = agente.obtener_accion(estado_anterior)

            # realizar movimiento y obtener nuevo estado
            recompensa, hecho, puntuacion = juego.jugar_paso(movimiento_final)
            estado_nuevo = agente.obtener_estado(juego)

            # entrenar memoria corta
            agente.entrenar_memoria_corta(estado_anterior, movimiento_final, recompensa, estado_nuevo, hecho)

            # recordar
            agente.recordar(estado_anterior, movimiento_final, recompensa, estado_nuevo, hecho)

            if hecho:
                # entrenar memoria larga, graficar resultado
                juego.reset()
                agente.n_juegos += 1
                agente.entrenar_memoria_larga()

                

                if puntuacion > record:
                    record = puntuacion
                    agente.modelo_simple.guardar()

                print('Juego', agente.n_juegos, 'Puntuación', puntuacion, 'Record:', record)


                puntuaciones_grafico.append(puntuacion)
                puntuacion_total += puntuacion
                puntuacion_media = puntuacion_total / agente.n_juegos
                puntuaciones_medias_grafico.append(puntuacion_media)
                plot(puntuaciones_grafico, puntuaciones_medias_grafico, 'simple', save=False)
    plot(puntuaciones_grafico, puntuaciones_medias_grafico, 'simple', save=True)
    save_results(puntuaciones_grafico, puntuaciones_medias_grafico, record, 'simple')


    


if __name__ == '__main__':
    entrenar()