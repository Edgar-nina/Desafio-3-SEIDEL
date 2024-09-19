import numpy as np

def gauss_seidel(A, b, x0, tolerance, max_iterations):
    n = len(A)
    x = x0.copy()
    
    for k in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            
            sum1 = sum(A[i][j] * x[j] for j in range(n) if j != i)
            
            
            x[i] = (b[i] - sum1) / A[i][i]
        
       
        if np.linalg.norm(np.array(x) - np.array(x_old), ord=np.inf) < tolerance:
            return x, k + 1  # Retorna la solución y el número de iteraciones
    
    return x, max_iterations  


A = np.array([[0.52, 0.3, 0.18],
              [0.2, 0.5, 0.3],
              [0.25, 0.2, 0.55]])

b = np.array([4800, 5810, 5690])


x0 = [0, 0, 0]  # Estimación inicial para cada variable
tolerance = 1e-6
max_iterations = 100


solution, iterations = gauss_seidel(A, b, x0, tolerance, max_iterations)


print("Solución:", solution)
print("Iteraciones:", iterations)