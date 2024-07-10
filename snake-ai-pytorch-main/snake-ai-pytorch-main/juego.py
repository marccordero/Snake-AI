import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import pygame
from constants import EPISODES

pygame.init()

fuente = pygame.font.SysFont('arial', 25)

class Direccion(Enum):
    DERECHA = 1
    IZQUIERDA = 2
    ARRIBA = 3
    ABAJO = 4

Punto = namedtuple('Punto', 'x, y')

# colores rgb
BLANCO = (255, 255, 255)
ROJO = (200, 0, 0)
AZUL1 = (0, 0, 255)
AZUL2 = (0, 100, 255)
NEGRO = (0, 0, 0)

TAM_BLOQUE = 20
VELOCIDAD = 40

class JuegoSerpienteAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # inicializar la pantalla
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Serpiente')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # inicializar el estado del juego
        self.direccion = Direccion.DERECHA

        self.cabeza = Punto(self.w/2, self.h/2)
        self.serpiente = [self.cabeza,
                          Punto(self.cabeza.x-TAM_BLOQUE, self.cabeza.y),
                          Punto(self.cabeza.x-(2*TAM_BLOQUE), self.cabeza.y)]

        self.puntaje = 0
        self.comida = None
        self._colocar_comida()
        self.iteracion_frame = 0


    def _colocar_comida(self):
        x = random.randint(0, (self.w-TAM_BLOQUE)//TAM_BLOQUE)*TAM_BLOQUE
        y = random.randint(0, (self.h-TAM_BLOQUE)//TAM_BLOQUE)*TAM_BLOQUE
        self.comida = Punto(x, y)
        if self.comida in self.serpiente:
            self._colocar_comida()


    def jugar_paso(self, accion):
        self.iteracion_frame += 1
        # 1. recolectar la entrada del usuario
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. mover
        self._mover(accion) # actualizar la cabeza
        self.serpiente.insert(0, self.cabeza)
        
        # 3. verificar si el juego ha terminado
        recompensa = 0
        juego_terminado = False
        if self.hay_colision() or self.iteracion_frame > 100*len(self.serpiente):
            juego_terminado = True
            recompensa = -10
            return recompensa, juego_terminado, self.puntaje

        # 4. colocar nueva comida o simplemente moverse
        if self.cabeza == self.comida:
            self.puntaje += 1
            recompensa = 10
            self._colocar_comida()
        else:
            self.serpiente.pop()
        
        # 5. actualizar la interfaz de usuario y el reloj
        self._actualizar_interfaz()
        self.clock.tick(VELOCIDAD)
        # 6. devolver si el juego ha terminado y el puntaje
        return recompensa, juego_terminado, self.puntaje


    def hay_colision(self, pt=None):
        if pt is None:
            pt = self.cabeza
        # choca con el límite
        if pt.x > self.w - TAM_BLOQUE or pt.x < 0 or pt.y > self.h - TAM_BLOQUE or pt.y < 0:
            return True
        # choca consigo misma
        if pt in self.serpiente[1:]:
            return True

        return False


    def _actualizar_interfaz(self):
        self.display.fill(NEGRO)

        for pt in self.serpiente:
            pygame.draw.rect(self.display, AZUL1, pygame.Rect(pt.x, pt.y, TAM_BLOQUE, TAM_BLOQUE))
            pygame.draw.rect(self.display, AZUL2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, ROJO, pygame.Rect(self.comida.x, self.comida.y, TAM_BLOQUE, TAM_BLOQUE))

        texto = fuente.render("Puntuación: " + str(self.puntaje), True, BLANCO)
        self.display.blit(texto, [0, 0])
        pygame.display.flip()


    def _mover(self, accion):
        # [recto, derecha, izquierda]

        sentido_agujas = [Direccion.DERECHA, Direccion.ABAJO, Direccion.IZQUIERDA, Direccion.ARRIBA]
        idx = sentido_agujas.index(self.direccion)

        if np.array_equal(accion, [1, 0, 0]):
            nueva_dir = sentido_agujas[idx] # sin cambios
        elif np.array_equal(accion, [0, 1, 0]):
            siguiente_idx = (idx + 1) % 4
            nueva_dir = sentido_agujas[siguiente_idx] # giro a la derecha d -> a -> i -> u
        else: # [0, 0, 1]
            siguiente_idx = (idx - 1) % 4
            nueva_dir = sentido_agujas[siguiente_idx] # giro a la izquierda d -> u -> i -> a

        self.direccion = nueva_dir

        x = self.cabeza.x
        y = self.cabeza.y
        if self.direccion == Direccion.DERECHA:
            x += TAM_BLOQUE
        elif self.direccion == Direccion.IZQUIERDA:
            x -= TAM_BLOQUE
        elif self.direccion == Direccion.ABAJO:
            y += TAM_BLOQUE
        elif self.direccion == Direccion.ARRIBA:
            y -= TAM_BLOQUE

        self.cabeza = Punto(x, y)
