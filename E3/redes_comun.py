"""Código común de las dos redes de la E3 (CNN y BiLSTM): la atención y el entrenamiento.

Las dos redes se entrenan exactamente igual (mismo optimizador, misma pérdida, misma parada temprana).
Así, si una gana a la otra, es por su bloque central y no por cómo se ha entrenado.
La carga de datos, Optuna y el guardado de predicciones los hace cada notebook.

La red que se entrene debe devolver (logits, pesos_de_atención) en su forward.
"""
import copy
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class Atencion(nn.Module):
    """Atención aditiva (Bahdanau et al., 2015; Yang et al., 2016) para clasificar textos.

    Entrada:  H (lote, palabras, dim) y mascara (lote, palabras), True = palabra real.
    Salida:   vector (lote, dim), un resumen por texto, y pesos (lote, palabras), que suman 1 por texto.
    """
    def __init__(self, dim, dim_atn=64):
        super().__init__()
        self.W = nn.Linear(dim, dim_atn)
        self.v = nn.Linear(dim_atn, 1, bias=False)

    def forward(self, H, mascara):
        puntos = self.v(torch.tanh(self.W(H))).squeeze(-1)       # una nota por palabra
        puntos = puntos.masked_fill(~mascara, float("-inf"))     # el relleno no cuenta
        pesos = torch.softmax(puntos, dim=1)                     # suman 1 por texto
        vector = (pesos.unsqueeze(-1) * H).sum(dim=1)            # media ponderada
        return vector, pesos


def fijar_semilla(semilla):
    """Llamar antes de crear la red: fija los pesos iniciales, el orden de los lotes y el dropout."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def pesos_clase(y, n_clases):
    """Pesos "balanced": n_total / (n_clases · n_clase). Las clases raras pesan más en la pérdida."""
    n_clase = torch.bincount(y, minlength=n_clases).float()
    return len(y) / (n_clases * n_clase)


def predecir(red, X, tam_lote=256):
    """Clase predicha (número) de cada texto de X."""
    red.eval()
    with torch.no_grad():
        logits = [red(X[i:i + tam_lote])[0] for i in range(0, len(X), tam_lote)]
    return torch.cat(logits).argmax(dim=1).numpy()


def entrenar(red, X_train, y_train, X_dev, y_dev, lr, batch_size, n_clases,
             max_epocas=30, paciencia=3, trial=None):
    """Entrena con parada temprana según el F1 macro en dev (normalmente, el dev de Political).

    X_*: tensores de secuencias (textos, 150); y_*: tensores con el número de clase.
    Tras cada época mide el F1 en dev; si no mejora en `paciencia` épocas seguidas, para.
    Al final deja en la red los pesos de la mejor época y devuelve su F1.
    Con `trial` (Optuna), informa del F1 de cada época para que la poda corte los intentos malos.
    """
    perdida = nn.CrossEntropyLoss(weight=pesos_clase(y_train, n_clases))
    # solo los parámetros que se entrenan (con eswiki congelado, el embedding no)
    optimizador = torch.optim.Adam([p for p in red.parameters() if p.requires_grad], lr=lr)

    mejor_f1, mejor_estado, sin_mejora = -1.0, None, 0
    for epoca in range(max_epocas):
        red.train()
        orden = torch.randperm(len(X_train))
        for i in range(0, len(orden), batch_size):
            lote = orden[i:i + batch_size]
            optimizador.zero_grad()
            logits, _ = red(X_train[lote])
            perdida(logits, y_train[lote]).backward()
            optimizador.step()

        f1 = f1_score(y_dev.numpy(), predecir(red, X_dev), average="macro", zero_division=0)
        if trial is not None:
            trial.report(f1, epoca)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if f1 > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = f1, copy.deepcopy(red.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora == paciencia:
                break

    red.load_state_dict(mejor_estado)
    return mejor_f1
