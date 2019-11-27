from random import randrange
import numpy.random as rnd
import funciones as fn
import termometro as trm
import numpy as np

#########  ESTRUCTURAS DE VECINDAD  #########
"""
Vecindad definida en función del número de movimientos que se van a realizar en el algoritmo. 
Esta vecindad fomenta la diverificación a medida que el algoritmo avanza, contrarrestando la intensificación de SA.
"""


def vecindad(x, movimiento=1, intervalo=(0, 30)):
    vecindad = (intervalo[1] - intervalo[0]) * movimiento / 10
    delta = (-vecindad / 2.) + vecindad * rnd.random_sample()
    return fn.ajustar_valor(delta + x, intervalo)


#########  PROBABILIDAD DE ACEPTACIÓN  #########
"""
Probabilidad de que una solución sea aceptada como nueva solucion
"""


def probabilidad_aceptacion(y, y_nuevo, temperatura):
    # si mejora la solución actual se acepta siempre
    if y_nuevo < y:
        return 1
    # en caso contrario, la aceptación dependerá de la temperatura,
    # a medida que avanza el problema menor temperatura, menor probabilidad de aceptación
    else:
        probabilidad = np.exp(- (y_nuevo - y) / temperatura)
        return probabilidad


#########  SOLUCIÓN INICIAL  #########
"""
Valor inicial, x, con el que parte el algoritmo. Se elige un valor aleatorio en el intervalo definido
"""


def inicializador_aleatorio(INTERVALO):
    return randrange(INTERVALO[0], INTERVALO[1])


#########  ALGORITMOS  #########
"""
Versión más básica del recocido simulado
"""


def recocido_simulado(funcion, vecindad, x_inicial, probabilidad_aceptacion, max_iteraciones=10000, informacion=1):
    T = 1
    x = x_inicial
    y = funcion(x)
    valores_x, valores_y = [x], [y]
    valores_t = []
    for iteracion in range(max_iteraciones):
        estado = iteracion / float(max_iteraciones)
        T = trm.get_temperatura(iteracion=iteracion, descanso=100, temperatura_actual=T)
        valores_t.append(T)
        x_vecina = vecindad(x, estado)
        y_vecina = funcion(x_vecina)
        if informacion == 1:
            print(
                "Iteracion {:>2}/{:>2} : T = {:>4.3g}, x = {:>4.3g}, y = {:>4.3g}, x_vecina = {:>4.3g}, y_vecina = {:>4.3g}".format(
                    iteracion, max_iteraciones, T, x, y, x_vecina, y_vecina))
        if probabilidad_aceptacion(y, y_vecina, T) > rnd.random():
            x, y = x_vecina, y_vecina
            valores_x.append(x)
            valores_y.append(y)
    return x, funcion(x), valores_x, valores_y, valores_t


"""
Esta función tiene memoria de la mejor solución encontrada hasta el momento
"""


def recocido_simulado_memory(funcion, vecindad, x_inicial, probabilidad_aceptacion, max_iteraciones=10000,
                             informacion=1):
    T = 1
    x = x_inicial
    mejor_x = x
    y = funcion(x)
    mejor_y = y
    valores_x, valores_y = [x], [y]
    valores_t = []
    for iteracion in range(max_iteraciones):
        estado = iteracion / float(max_iteraciones)
        T = trm.get_temperatura(iteracion=iteracion, descanso=100, temperatura_actual=T)
        valores_t.append(T)
        x_vecina = vecindad(x, estado)
        y_vecina = funcion(x_vecina)
        if informacion == 1:
            print(
                "Iteracion {:>2}/{:>2} : T = {:>4.3g}, x = {:>4.3g}, y = {:>4.3g}, x_vecina = {:>4.3g}, y_vecina = {:>4.3g}".format(
                    iteracion, max_iteraciones, T, x, y, x_vecina, y_vecina))
        if probabilidad_aceptacion(y, y_vecina, T) > rnd.random():
            x, y = x_vecina, y_vecina
            valores_x.append(x)
            valores_y.append(y)
        if mejor_y > y:
            mejor_x, mejor_y = x, y
            if informacion > 1: print("Nueva mejor solucion, x = " + str(mejor_x) + " - f(x) = " + str(mejor_y))
    return mejor_x, mejor_y, valores_x, valores_y, valores_t
