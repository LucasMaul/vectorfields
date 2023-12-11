import matplotlib.pyplot as plt
import numpy as np
import inspect
import pandas as pd

def vectorfield(function_of_xy,
                y_min=-5, y_max=5, y_step=1,
                x_min=-5, x_max=5, x_step=1,
                xlabel=None, ylabel=None,
                vector_scale=1, plot_root=False, root_size=0.1,
                show=False, title=None, **kwargs):
    """
    Plot a vector field representing the normalized gradient vectors of a given function.

    Parameters:
    ----------
    function_of_xy : callable
        A function of two variables (x, y) defining the vector field.

    y_min, y_max, y_step : float, optional
        Parameters to define the y-axis range and step size.

    x_min, x_max, x_step : float, optional
        Parameters to define the x-axis range and step size.

    xlabel, ylabel : str, optional
        Labels for the x and y axes. If None, default to 'x' and 'y'.

    vector_scale : float, optional
        Scaling factor for the length of the gradient vectors.

    plot_root : bool, optional
        If True, plot the starting positions of the gradient vectors.

    root_size : float, optional
        Size of the root markers if plot_root is True.

    show : bool, optional
        If True, display the plot. If False, the plot is not displayed.

    title : str, optional
        Title for the plot. If None, the function source code is used as the title.

    **kwargs : additional keyword arguments
        Additional keyword arguments to be passed to the `quiver` function for customizing vector properties.

    Returns:
    -------
    None

    Example:
    --------
    import numpy as np
    import matplotlib.pyplot as plt

    # Define a vector field function
    my_vector_field = lambda y, x: x**2 - y**2, 2*x*y

    # Plot the vector field
    vectorfield(my_vector_field, x_min=-3, x_max=3, y_min=-3, y_max=3,
                xlabel='X-axis', ylabel='Y-axis', vector_scale=0.5,
                plot_root=True, show=True, title='My Vector Field', color='green', linewidth=1.5)
    """
    # ... (rest of the function implementation)
    
    x,y = np.meshgrid(np.arange(start=x_min, stop=x_max+x_step, step=x_step), np.arange(start=y_min, stop=y_max+y_step, step=y_step))
    
    # normalized gradient vectors
    u, v = 1, (function_of_xy)(x, y)
    magnitudes = np.sqrt(u**2+v**2)
    u_normalized = u/magnitudes
    v_normalized = v/magnitudes

    # starting position of gradient vectors
    x_root = x - 1/2*u_normalized*vector_scale
    y_root = y - 1/2*v_normalized*vector_scale

    # plotting some stuff
    fig, ax = plt.subplots()

    ax.grid(True, zorder=1, linestyle='dotted')
    if plot_root:
        ax.scatter(x, y, color='r', s=root_size)
    ax.quiver(x_root, y_root, u_normalized, v_normalized, angles='xy', scale_units='xy', scale=1/vector_scale, width=2/10**3, zorder=2, **kwargs)

    # defining windows size
    window_x_min, window_x_max = x_min - x_step, x_max + x_step
    window_y_min, window_y_max = y_min - y_step, y_max + y_step
    ax.set_xlim([window_x_min, window_x_max])
    ax.set_ylim([window_y_min, window_y_max])
    ax.set_aspect('equal', adjustable='box')

    # labeling
    if title is not None:
        title_to_plot = title
    else:
        title_to_plot = inspect.getsource(function_of_xy)
    ax.set_title(title_to_plot)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('x')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('y')

    if show:
        plt.show()


def euler_method(function_of_xy, step_h, initial_x, initial_y, approx_x, plot=False, show=False, **kwagrs):
    """
    Numerically solve a first-order ordinary differential equation (ODE) using the Euler method.

    Parameters
    ----------
    function_of_xy : callable
        A function representing the ODE in the form f(x, y).
    step_h : float
        The step size for the Euler method.
    initial_x : float
        The initial value of x.
    initial_y : float
        The initial value of y corresponding to initial_x.
    approx_x : float
        The x-value at which the solution is approximated.
    plot : bool, optional
        If True, plot the solution. Default is False.
    show : bool, optional
        If True and plot is True, display the plot. Default is False.
    **kwargs
        Additional keyword arguments to be passed to the plot function.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the iteration information, including 'k', 'x_k', 'y_k',
        'f(x_k, y_k)', and 'h*f(x_k, y_k)'.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Define the ODE: dy/dx = x + y
    >>> lambda x, y: np.cos(x + y) + np.sin(x - y)
    >>>
    >>> # Solve the ODE using the Euler method
    >>> result = euler_method(ode_function, 0.1, 0, 1, 2, plot=True, show=True, color='b', label='Euler Method')
    >>> print(result)
       k  x_k       y_k  f(x_k, y_k)  h*f(x_k, y_k)
    0  0  0.0  1.000000         1.0            0.1
    1  1  0.1  1.100000         1.1            0.11
    2  2  0.2  1.210000         1.2            0.12
    ...
    20 20  2.0  6.727500         3.0            0.3
    """
    iteration_list = pd.DataFrame({'k':[], 'x_k':[], 'y_k':[], 'f(x_k, y_k)':[], 'h*f(x_k, y_k)':[]})
    k = 0
    x = initial_x
    y = initial_y
    while x <= approx_x:
        f = (function_of_xy)(x, y)
        h = step_h * f
        iteration_list.loc[len(iteration_list)] = [k, x, y, f, h]
        k += 1
        y += h
        x += step_h
    
    if plot:
        fig, ax = plt.gcf(), plt.gca()
        ax.plot(iteration_list['x_k'], iteration_list['y_k'], **kwagrs)

        if show:
            plt.show()

    return iteration_list

def heuns_method(function_of_xy, step_h, initial_x, initial_y, approx_x, plot=False, show=False, **kwagrs):
    """
    Apply Heun's method to numerically solve a first-order ordinary differential equation (ODE).

    Parameters
    ----------
    function_of_xy : callable
        A function representing the ODE, accepting x and y as arguments.
    step_h : float
        Step size for the numerical integration.
    initial_x : float
        Initial value of the independent variable.
    initial_y : float
        Initial value of the dependent variable.
    approx_x : float
        Value of the independent variable where the solution is approximated.
    plot : bool, optional
        If True, plot the numerical solution. Default is False.
    show : bool, optional
        If True and plot is True, display the plot. Default is False.
    **kwargs
        Additional keyword arguments for customizing the plot.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the iteration history with columns ['k', 'x_k', 'y_k', 'k_1', 'k_2', 'h/2*(k_1 + k_2)'].

    Example
    -------
    import matplotlib.pyplot as plt

    # Define the ODE dy/dx = x + y
    lamda x, y: x + y

    # Set initial conditions and parameters
    initial_x_value = 0
    initial_y_value = 1
    step_size = 0.1
    target_x_value = 2

    # Apply Heun's method
    iteration_results = heuns_method(
        function_of_xy=ode_function,
        step_h=step_size,
        initial_x=initial_x_value,
        initial_y=initial_y_value,
        approx_x=target_x_value,
        plot=True,
        show=True,
        label='Heun\'s Method'
    )

    print(iteration_results)
    """
    iteration_list = pd.DataFrame({'k':[], 'x_k':[], 'y_k':[], 'k_1':[], 'k_2':[], 'h/2*(k_1 + k_2)':[]})
    k = 0
    x = initial_x
    y = initial_y
    while x <= approx_x:
        f = (function_of_xy)(x, y)
        k_1 = f
        k_2 = (function_of_xy)(x + step_h, y + step_h * k_1)
        h = step_h / 2 * (k_1 + k_2)
        iteration_list.loc[len(iteration_list)] = [k, x, y, k_1, k_2, h]
        k += 1
        y += h
        x += step_h

    if plot:
        fig, ax = plt.gcf(), plt.gca()
        ax.plot(iteration_list['x_k'], iteration_list['y_k'], **kwagrs)

        if show:
            plt.show()

    return iteration_list


if __name__ == '__main__':
    # func = lambda t, x: np.cos(t + x) + np.sin(t - x)
    # vectorfield(func,
    #             x_min=-np.pi,x_max=np.pi, x_step=np.pi/16,
    #             y_min=-np.pi, y_max=np.pi, y_step=np.pi/16,
    #             title="$x'(t)=t \\cdot x(t)$", xlabel='t', ylabel='x',
    #             vector_scale=0.2, color='b',
    #             show=False, plot_root=True)
    # euler_method(function_of_xy=func, step_h=0.001, initial_x=0, initial_y=0, approx_x=1, plot=True, show=False, color='r')
    # plt.show()
    func = lambda t, x: t - x
    a1 = heuns_method(function_of_xy=func, step_h=0.1  , initial_x=0, initial_y=1, approx_x=2, plot=True)
    a2 = heuns_method(function_of_xy=func, step_h=0.01 , initial_x=0, initial_y=1, approx_x=2, plot=True)
    a3 = heuns_method(function_of_xy=func, step_h=0.001, initial_x=0, initial_y=1, approx_x=2, plot=True)

    print(a1.tail())
    print(a2.tail())
    print(a3.tail())

    plt.show()