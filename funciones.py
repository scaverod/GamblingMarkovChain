# En esta clase hay una recopilación de funciones
from math import e, cos, sin, tan, tanh, log

import numpy


def exponencial(x, base=e):
    return base ** x


def potencia(x, exponente=2):
    return x ** exponente


def abs(x):
    return x * ((x < 0) * (-1) + (x > 0))


def seno(x):
    return sin(x)


def coseno(x):
    return cos(x)


def tangente(x):
    return tan(x)


def tangente_hipervolica(x):
    return tanh(x)


def funcion_enunciado(x):
    try:
        with numpy.errstate(divide='ignore'):
            return coseno(x) / x
    except ZeroDivisionError:
        pass


def ajustar_valor(x, intervalo):
    return max(min(x, intervalo[1]), intervalo[0])
