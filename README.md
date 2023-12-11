# vectorfields

Vector Field and Euler Method Plotter

This Python script provides functionality to visualize vector fields and numerically solve first-order ordinary differential equations (ODEs) using the Euler method. The script utilizes the matplotlib and numpy libraries for plotting and mathematical operations.

Features

## 1. Vector Field Visualization
The vectorfield function in the script allows you to plot a vector field representing the normalized gradient vectors of a given function. You can customize the function, axis ranges, vector scaling, root plotting, and more.



### Example 1:

Plotting the vectorfield for $y(x)' = -x + a \cdot y + x^2 \cdot y$ :
```python
import numpy as np
import matplotlib.pyplot as plt

from vectorfields import vectorfield


# Define a vector field function
a = 0.1
my_ode = lambda y, x: -x + a*y + x**2 * y

# Plot the vector field
vectorfield(my_ode, y_min=0, y_max=2.5, y_step=0.1, x_min=0, x_max=2.5, x_step=0.1,
            title="$y(x)'= -x + a \cdot y + x^2 \cdot y$" , vector_scale=.1,
            plot_root=True, xlabel='X-axis', ylabel='Y-axis', show=True, color='b')

```
![Vectorfield](https://github.com/LucasMaul/vectorfields/blob/main/example1.png)


## 2. Euler Method Solver
The euler_method function allows you to numerically solve a first-order ODE using the Euler method. You can specify the ODE function, step size, initial values, and the x-value at which the solution is approximated. Additionally, the solution can be plotted. The method returns a pandas dataframe. 

### Example 2:
Plotting the euler approximation for  $x(t)'=t \cdot x$. 

Analytical exact solution is $x(t)=\frac{1}{10} e^{1/2 \cdot t^2}$.
```python
import numpy as np
import matplotlib.pyplot as plt

from vectorfields import euler_method

function1 = lambda t,x: t*x

a1 = euler_method(function1, step_h=0.4, initial_x=0, initial_y=0.1, approx_x=2, plot=True, label='h=0.4')
a2 = euler_method(function1, step_h=0.2, initial_x=0, initial_y=0.1, approx_x=2, plot=True, label='h=0.2')
a3 = euler_method(function1, step_h=0.1, initial_x=0, initial_y=0.1, approx_x=2.1, plot=True, label='h=0.1')

# manually build the exact solution
exact_sol = lambda t: 0.1 * np.e**(1/2*t**2) 
x_values = np.linspace(0,2,100)
y_values = [(exact_sol)(t) for t in x_values]

# add the exact solution
fig, ax = plt.gcf(), plt.gca()
ax.plot(x_values,y_values, label='exact')
ax.legend()

# annotate the solutions
ax.annotate(str(round(a1.iloc[-1,2],2)), (2, a1.iloc[-1,2]))
ax.annotate(str(round(a2.iloc[-1,2],2)), (2, a2.iloc[-1,2]))
ax.annotate(str(round(a3.iloc[-1,2],2)), (2, a3.iloc[-1,2]))
ax.annotate(str(round(y_values[-1],2)), (2, y_values[-1]))

plt.show()
```

![Euler Method](https://github.com/LucasMaul/vectorfields/blob/main/example2.png)

### Example 3:
Plotting the euler approximation for  $x(t)'=t \cdot x$ within vectorfield.
Analytical exact solution is $x(t)=\frac{1}{10} e^{1/2 \cdot t^2}$.
```python
import numpy as np
import matplotlib.pyplot as plt

from vectorfields import vectorfield, euler_method

function1 = lambda t,x: t*x
vectorfield(function1, vector_scale=0.2,
            x_min=-1, x_max=3, x_step=0.25,
            y_min=-1, y_max=3, y_step=0.25,
            xlabel='t', ylabel='x', plot_root=True,
            title="$x'(t)=t \\cdot x(t)$", color='b')

a1 = euler_method(function1, step_h=0.4, initial_x=0, initial_y=0.1, approx_x=2, plot=True, label='h=0.4')
a2 = euler_method(function1, step_h=0.2, initial_x=0, initial_y=0.1, approx_x=2, plot=True, label='h=0.2')
a3 = euler_method(function1, step_h=0.1, initial_x=0, initial_y=0.1, approx_x=2.1, plot=True, label='h=0.1')

# manually build the exact solution
exact_sol = lambda t: 0.1 * np.e**(1/2*t**2) 
x_values = np.linspace(0,2,100)
y_values = [(exact_sol)(t) for t in x_values]

# add the exact solution
fig, ax = plt.gcf(), plt.gca()
ax.plot(x_values,y_values, label='exact')
ax.legend()

# annotate the solutions
ax.annotate(str(round(a1.iloc[-1,2],2)), (2, a1.iloc[-1,2]))
ax.annotate(str(round(a2.iloc[-1,2],2)), (2, a2.iloc[-1,2]))
ax.annotate(str(round(a3.iloc[-1,2],2)), (2, a3.iloc[-1,2]))
ax.annotate(str(round(y_values[-1],2)), (2, y_values[-1]))

plt.show()
```

![Euler Method](https://github.com/LucasMaul/vectorfields/blob/main/example3.png)


# Heun's Method for Numerical Solution of ODEs

## Overview
Heun's method is a numerical technique used to approximate the solution of first-order ordinary differential equations (ODEs). It is an iterative method that improves upon the Euler method by using a predictor-corrector approach. This method is also known as the improved Euler method.

## Mathematical Background
Consider a first-order ODE in the form: dy/dx = f(x, y), where y is the dependent variable and f(x, y) is a given function.

Heun's method involves the following iterative steps:

1. **Predictor Step (k_1):**
   - Compute the slope at the current point (x, y) using the function f(x, y). This is denoted as k_1.
   - Predict the value of the dependent variable at the next time step using the Euler method: y_pred = y + h * k_1.

2. **Corrector Step (k_2):**
   - Use the predicted value (x + h, y_pred) to compute the slope k_2.
   - Average the slopes k_1 and k_2 to obtain a more accurate estimate of the slope over the interval.
   - Update the dependent variable using the averaged slope.

3. **Iteration:**
   - Repeat the predictor-corrector steps until the desired approximation point is reached.

## Function Parameters
The `heuns_method` function takes the following parameters:

- `function_of_xy`: A callable representing the ODE, accepting x and y as arguments.
- `step_h`: Step size for numerical integration.
- `initial_x`, `initial_y`: Initial values of the independent and dependent variables.
- `approx_x`: Value of the independent variable where the solution is approximated.
- `plot` (optional): If True, plot the numerical solution. Default is False.
- `show` (optional): If True and `plot` is True, display the plot. Default is False.
- `**kwargs`: Additional keyword arguments for customizing the plot.

## Example Usage
```python
import matplotlib.pyplot as plt

# Define the ODE dx/dt = t - x
ode_function = lambda x, t: t - x

# Set initial conditions and parameters
initial_x_value = 0
initial_t_value = 1
step_size = 0.001
target_x_value = 2

# Apply Heun's method
func = lambda t, x: t - x


vectorfield(func,
            x_min=-0.25, x_max=2.25, x_step=0.1,
            y_min=0.5, y_max=1.5, y_step=0.1,
            plot_root=True, root_size=3, show=False,
            xlabel='t', ylabel='x', title="$x'(t) = t - x$",
            vector_scale=0.07, color='b')

a3 = heuns_method(function_of_xy=func, step_h=step_size,
                    initial_x=initial_x_value, initial_y=initial_t_value,
                    approx_x=target_x_value, plot=True, show=False, color='g')

plt.show()

print(a3.tail())

# Returns:
#            k    x_k       y_k       k_1       k_2  h/2*(k_1 + k_2)
# 1997  1997.0  1.997  1.268484  0.728516  0.728788         0.000729
# 1998  1998.0  1.998  1.269213  0.728787  0.729059         0.000729
# 1999  1999.0  1.999  1.269941  0.729059  0.729329         0.000729
# 2000  2000.0  2.000  1.270671  0.729329  0.729600         0.000729
# 2001  2001.0  2.001  1.271400  0.729329  0.729600         0.000729
```


![Heuns Method](https://github.com/LucasMaul/vectorfields/blob/main/example4.png)