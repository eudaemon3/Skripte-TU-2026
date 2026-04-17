import numpy as np
from scipy.integrate import quad

m = 1
hbar = 1

class InfSquareWell:

    def __init__(self, 
        N: int, a: int, tmax: float, 
        x_0: float, s_0:float, 
        k_0: float) -> None:

        self.N = N
        self.a = a
        self.tmax = tmax
        self.x_0 = x_0
        self.s_0 = s_0
        self.k_0 = k_0

    def _phi_n(self, n: int, x: np.ndarray) -> np.ndarray:
        kn = lambda n: np.pi/self.a * n
        f = np.sqrt(2/self.a)

        if n % 2:
            return f * np.cos(kn(n)*x) 
        else:
            return f * np.sin(kn(n)*x)

    def _phi(self, x: np.ndarray, k_0: float) -> np.ndarray:
        f = 1/np.sqrt(self.s_0 * np.sqrt(2*np.pi))
        return f * np.exp(-(x-self.x_0)**2 / (2*self.s_0)**2 + 1j*k_0*x)

    def _integrand(self, x: np.ndarray, n: int, k_0: float) -> np.ndarray:
        return np.conj(self._phi_n(n, x)) * self._phi(x, k_0)

    def get_psi(self, rx: int, tx: int) -> tuple[np.ndarray]:
        wn = lambda n: hbar / (2*m) * (np.pi/self.a)**2 * n**2

        x = np.linspace(-self.a/2, self.a/2, rx)
        t = np.linspace(0, self.tmax, tx)
        x2d = x[:, np.newaxis]
        t2d = t[np.newaxis, :]

        psi = np.zeros((len(x), len(t)), dtype=complex)

        for i in range(self.N):
            n = i + 1
            c, _ = quad(self._integrand, -self.a/2, self.a/2, args=(n, self.k_0, ), limit=200, complex_func=True)

            psi += c * self._phi_n(n, x2d) * np.exp(-1j * wn(n) * t2d)

        return psi, x, t


