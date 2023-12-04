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


