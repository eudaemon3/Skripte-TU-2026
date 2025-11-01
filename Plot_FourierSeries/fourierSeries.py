import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Functon and Fourier Series Definition
# ---------------------------------------------------------------------
f = lambda t : np.where(t>0, np.sin(t), np.cos(t))

a0 = 2/np.pi
a1 = 0.5
an = lambda n : np.where(n%2, 0, 2/np.pi* 1/(1-n**2))

b1 = 0.5
bn = lambda n : np.where(n%2, 0, 2*n/np.pi* 1/(1- n**2))  

ctr = (a0, an, bn)
sing = (1, a1, b1)

t = np.linspace(-np.pi, np.pi, 1000)

# ---------------------------------------------------------------------
# Define Series
# ---------------------------------------------------------------------
def FR_Nf(t, N:int, ctr:tuple, sing:tuple|None=None):
    """
    Compute the N-term partial sum of a Fourier series at points `t`.

    Parameters
    ----------
    t : array_like
        Points where the partial sum is evaluated. Any numpy-compatible
        1D array-like (e.g. numpy.ndarray) is accepted. 
    N : int
        Number of harmonics to include (terms 1..N). Should be non-negative.
        If N == 0 the function returns the constant a0/2.
    ctr : tuple (a0, an, bn)
        a0 : scalar
            The Fourier coefficient as n=0.
        an : callable
            Function or callable that accepts an integer `n` and returns the
            coefficient a_n (scalar).
        bn : callable
            Function or callable that accepts an integer `n` and returns the
            coefficient b_n (scalar).
    sing : tuple (index, a_index, b_index) or None, optional
        Optional override for a single harmonic `index`. If provided, when
        n == index the coefficients a_index and b_index are used instead of
        an(index) and bn(index). This is useful when a_n or b_n formulas
        have a singularity at a specific `n`. If None (default), no override
        is applied.

    Returns
    -------
    numpy.ndarray
        Array with the same shape as `t` containing the partial sum values.

    Notes
    -----
    - The implementation uses the 2π-periodic basis cos(n*t), sin(n*t).
    - The `sing` override compares `n == index` using Python equality; ensure
      the `index` is an integer and matches the type used in the loop.
    """
    a0, an, bn = ctr

    if sing is None:
        def term(t, n): 
            return an(n)*np.cos(n*t) + bn(n)*np.sin(n*t)
    else:
        si, sa, sb = sing
        def term(t, n):
            if n == si:
                return sa*np.cos(si*t) + sb*np.sin(si*t)
            else:
                return an(n)*np.cos(n*t) + bn(n)*np.sin(n*t)
    
    res = a0/2 * np.ones_like(t)
    for i in range(1,N+1):
        res += term(t, i)

    return res

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
plot_range=range(0,10,2)
fig, ax = plt.subplots(1,1, figsize=(6, 4))

ax.plot(t, f(t), 'k-', linewidth=2, label='function')
for ind in plot_range:
    ax.plot(t, FR_Nf(t, ind, ctr, sing), linewidth=1, label=f'n = {ind}')

ax.set_ylabel(r'$f(t)$', fontsize=16)
ax.set_xlabel(r'$t$', fontsize=16)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()