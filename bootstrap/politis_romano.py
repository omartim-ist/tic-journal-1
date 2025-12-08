"""
Politis–Romano Stationary Bootstrap

Bootstrap method for dependent time series. Instead of using 
fixed-length blocks, it generates blocks with random lengths 
following a geometric distribution with mean l. Each block 
starts at a random position in the original series and 
continues forward with probability (1 - p), where p = 1/l. 
When the block ends, a new block starts at another random 
position. This process continues until a bootstrap series of 
the same length is formed.

The method preserves short-range dependence, works well for 
stationary series, and avoids the artificial boundary effects 
of fixed-block bootstrap methods.
"""

import numpy as np
import time
from joblib import Parallel, delayed

class PolitisRomanoBootstrap:
    def __init__(self, serie, n_boot, l=None, T=None):
        """
        serie : array-like
            - 1D: shape (T,)       -> single asset
            - 2D: shape (n_assets, T) -> multi-asset price series
        n_boot : int
            number of bootstrap series to generate
        l : float, optional
            average block length (=> p = 1/l). If None, uses T**(1/3),
            where T is the number of return steps.
            
        T : tamanho das series que se quer gerar
        """

        arr = np.asarray(serie, dtype=float)

        if arr.ndim == 1:
            # (T,) -> (1, T)
            arr = arr[None, :]

        if arr.ndim != 2:
            raise ValueError("serie must be 1D (T,) or 2D (n_assets, T).")

        self.prices = arr                         # shape: (n_assets, T_prices)
        self.n_assets, T_prices = self.prices.shape

        # log-returns ao longo do tempo
        self.lrets = np.diff(np.log(self.prices), axis=1)  # (n_assets, T_ret)
        
        if T is None:
            self.T = self.lrets.shape[1]                       # T_ret = T_prices - 1
        else:
            self.T = T

        self.n_boot = n_boot

        if l is None:
            l = self.T ** (1/3)

        self.l = float(l)
        self.p = 1.0 / self.l   # probabilidade de terminar um bloco

    def _generate_one(self):
        """Gera uma única série bootstrap multiasset, normalizada (começa em 1.0)."""
        X = self.lrets               # shape: (n_assets, T)
        n_assets = X.shape[0]
        T = self.T
        p = self.p

        # Y: log-returns bootstrap multiasset
        Y = np.empty((n_assets, T), dtype=float)
        idx = 0

        while idx < T:
            # início de bloco (escolhe índice temporal)
            pos = np.random.randint(0, T)  # 0..T-1

            # geração dentro do bloco
            while idx < T:
                # copia o vetor de retornos de TODOS os ativos nesse instante
                Y[:, idx] = X[:, pos]
                idx += 1

                # decide se o bloco termina
                if np.random.rand() < p:
                    break
                pos = (pos + 1) % T  # wrap-around

        # caminho "de preços" normalizado: começa em 1 para todos os ativos
        ratios = np.exp(np.cumsum(Y, axis=1))          # (n_assets, T)
        ones = np.ones((n_assets, 1), dtype=float)     # (n_assets, 1)
        Z = np.concatenate([ones, ratios], axis=1)     # (n_assets, T+1)

        return Z

    def generate(self, n_jobs=-1):
        """
        Gera todas as séries bootstrap.

        Retorno:
            array de shape (n_boot, n_assets, T+1),
            com caminhos normalizados (primeira coluna = 1.0).
        """
        time_init = time.time()
        print()
        print(f'Generating {self.n_boot} Politis-Romano series of '
              f'shape (n_assets={self.n_assets}, T={self.T + 1})...')

        boot_list = Parallel(n_jobs=n_jobs)(
            delayed(self._generate_one)() for _ in range(self.n_boot)
        )

        print(f'Done Politis-Romano in {round(time.time() - time_init, 3)} seconds.')
        print()

        return np.stack(boot_list, axis=0)   # (n_boot, n_assets, T+1)
    
    
''' 
serie = np.exp(np.cumsum(np.random.normal(loc=0, scale=0.01, size=10**2)))

n_assets = 2
logrets = np.random.normal(loc=0, scale=0.01, size=(n_assets, 10**2 - 1))

# construir preços normalizados (começam em 1.0)
serie = np.concatenate(
    (np.ones((n_assets, 1)), np.exp(np.cumsum(logrets, axis=1))),
    axis=1
)

pr = PolitisRomanoBootstrap(serie=serie, n_boot=4)
M = pr.generate()
'''
