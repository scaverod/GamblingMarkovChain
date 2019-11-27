# Temperaturas usadas
from math import e, cos, sin, tan, tanh, log


# Temepratura estática en la que referencia a Kirkpatric 1985, no encuentro el paper
def temperatura_kirkpatrick(iteracion, descanso, temperatura_actual, alpha, temperatura_inicial):
    if iteracion % descanso == 0:
        temperatura = alpha ** (iteracion / descanso) * temperatura_inicial
    else:
        temperatura = float(temperatura_actual)
    return max(0.01, min(1, temperatura))


def temperatura_lundy(temperatura, beta):
    return max(0.01, temperatura / (1 + beta * temperatura))


def temperatura_hajek(iteracion, constante):
    return constante / log(iteracion + 1)


def temperatura_adicional(estado):
    return max(0.01, 1 - estado)


def beta_lundy(temperatura_inicial, temperatura_final, n_entornos):
    return (temperatura_inicial - temperatura_final) / (n_entornos * temperatura_inicial * temperatura_final)


def get_temperatura(estado=-1, iteracion=-1, descanso=0, temperatura_actual=-1, alpha=0.95, temperatura_inicial=1,
                    constante=0, temperatura_final=-1, n_entornos=-1, beta=-1):
    if estado != -1:
        return temperatura_adicional(estado)
    elif constante != 0:
        return temperatura_hajek(iteracion, constante)
    elif beta != -1:
        return temperatura_lundy(temperatura_actual, beta)
    else:
        return temperatura_kirkpatrick(iteracion, descanso, temperatura_actual, alpha, temperatura_inicial)
