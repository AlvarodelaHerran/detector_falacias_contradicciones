import torch
import torch.nn as nn


class Atencion(nn.Module):
    """Atención aditiva (Bahdanau; Yang et al.) para clasificar textos.

    Entrada:  H (lote, palabras, dim) y mascara (lote, palabras), True = palabra real.
    Salida:   vector (lote, dim), un resumen por texto, y pesos (lote, palabras).
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