from sympy import Symbol, sqrt, diff # Used for calculating the derivatives
import matplotlib.pyplot as plt
import numpy as np

def factorial (n: int) -> int:
    """ Computes the factorial value for the integer n """
    if (n == 0): return 1 # 0! = 1
    else: return n * factorial(n - 1) if (n > 1) else 1 # n! = n * (n-1)! until n == 1

def computeDerivatives (f, x: Symbol, a: float, degrees: int) -> [float]:
    """ Computes the derivatives of the sympy expression f at a """
    # 1. Compute the derivatives
    derivatives = [f]
    for _ in range(degrees):
        derivatives.append(diff(derivatives[-1]))
    
    # 2. Compute the values of the deriviatives at a
    return [derivative.subs(x, a) for derivative in derivatives]

def intergrateTaylor (interval: (float), derivatives: [float], degreesToSave: [int]) -> [float]:
    """ Intergrates the taylor polynomial, on the given interval, and saves values at specific orders along the way """
    values = []
    area = derivatives[0] * interval[1] - derivatives[0] * interval[0] # Compute the first part of the intergral to avoid dividing by 0
    for i, derivative in enumerate(derivatives[1:], start = 1): # The derivatives
        area += (derivative / factorial(i)) * (1/(i + 1)) * np.power(interval[1], i + 1) - (derivative / factorial(i)) * (1/(i + 1)) * np.power(interval[0], i + 1)
        if (i in degreesToSave): values.append(area) # Save the current value of the area in the values list
    return values

if (__name__ == "__main__"): 
    x = Symbol("x", real = True) # SymPy Symbol used for the expression below
    f = sqrt(1 - x**2) # Expression for f(x)
    derivatives = computeDerivatives(f = f, x = x, a = 0.0, degrees = 80) # Calculates the derivatives needed up f^(500)(a)
    print(intergrateTaylor((-1, 1), derivatives, [10, 20, 50, 100, 200, 500])) # Computes the intergral on the interval [-1; 1]
    # Output: [1.58515737734488, 1.57628203375486, 1.57225240322015, 1.57131958196737, 1.57098284813794, 1.57084374682765]

def plotDifference (xs: np.array, ys: np.array) -> None:
    """ Plots the difference between the approximated value and the true value """
    # Plotting
    plt.style.use("ggplot")
    plt.plot(xs, ys)
    # Labels:
    plt.title("Error with respect to n")
    plt.xlabel("n")
    plt.ylabel("Abs(Error)")
    
    plt.show()
    
approximateValues = intergrateTaylor((-1, 1), derivatives[:81], [*range(0, 81, 1)])
plotDifference(np.arange(0, 80), abs(np.array([np.pi/2 for _ in range(80)]) - np.array(approximateValues)))
