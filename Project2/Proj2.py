import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from matplotlib import rc
import numpy as np
import timeit

# package dependencies: typ1cm & dvipng for LaTeX formatting
rc('text', usetex=True)
axes = Axes3D(plt.figure())
LaTeX_z = r"""$z = \frac{sin(x^2+3y^2)}{0.1+r^2}+(x^2+5y^2)\cdot\frac{e^{(1-r^2)}}{2},r=\sqrt{x^2+y^2}"""
sigfigs = "{0:.3f}"
#MAX_TEMP = 500
COOLING_RATE = 1

'''def main():
#    f = SA_Wrapper(z, 0.025, MAX_TEMP, -2.5, 2.5, -2.5, 2.5, True, LaTeX_z)
#    print(timeit.timeit(f, number=1))
#    simulated_annealing(z, 0.025, MAX_TEMP, -2.5, 2.5, -2.5, 2.5, True, LaTeX_z)
    hill_climb_random_restart(z, 0.015, 8, -2.5, 2.5, -2.5, 2.5, True, LaTeX_z)
    d = generateDomain(0, 2.5, 0.025)
    totalPoints = len(d) * len(d) * 4
    yes = 0
    X = 0
    while X < len(d):
        Y = 0
        while Y < len(d):
            minimum = easy_hill_climb(z, d, d, X, Y)
            if z(minimum[0], minimum[1]) <= -0.150:
                yes += 1
            Y += 1
        X += 1
        print(X)
    print("total points: " + str(totalPoints))
    print("points that lead to global min: " + str(yes * 4))'''
    
# functions to optimize

# paraboloid
def r(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y,2))

# whatever the heck this surface is ?_?
def z(x, y):
    term1 = np.sin(np.power(x, 2) + (3 * np.power(y, 2))) / (0.1 + np.power(r(x, y), 2))
    term2 = np.power(x, 2) + (5 * np.power(y, 2))
    term3 = np.exp(1 - np.power(r(x,y), 2)) / 2
    return term1 + (term2 * term3)



# generates a list of uniformly spaced values
def generateDomain(minValue, maxValue, step_size):
    return np.arange(minValue, maxValue, step_size)

# generates a random index >= 0 and < len(list)
def randomIndex(aList):
    return (np.random.randint(1, len(aList)) - 1)

# hill climbing bootstrap
def hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax, displayGraph=False, LaTeX_f=''):
    xvalues = generateDomain(xmin, xmax, step_size)
    yvalues = generateDomain(ymin, ymax, step_size)
    results = easy_hill_climb(function_to_optimize, xvalues, yvalues, randomIndex(xvalues), randomIndex(yvalues))
    if(displayGraph):
        generate_graph(function_to_optimize, xvalues, yvalues, results[2], results[3], results[0], results[1], "Hill Climbing", LaTeX_f)
    return (results[0], results[1])

# hill climbing with random restarts bootstrap
def hill_climb_random_restart(function_to_optimize, step_size, num_restarts, xmin, xmax, ymin, ymax, displayGraph=False, LaTeX_f=''):
    xvalues = generateDomain(xmin, xmax, step_size)
    yvalues = generateDomain(ymin, ymax, step_size)
    results = random_hill_climb(function_to_optimize, xvalues, yvalues, num_restarts, displayGraph)
    if(displayGraph):
        generate_graph(function_to_optimize, xvalues, yvalues, results[2], results[3], results[0], results[1], "Hill Climbing with Random Restarts", LaTeX_f)
    return (results[0], results[1])

def simulated_annealing(function_to_optimize, step_size, max_temp, xmin, xmax, ymin, ymax, displayGraph=False, LaTeX_f=''):
    xvalues = np.arange(xmin, xmax, step_size)
    yvalues = np.arange(ymin, ymax, step_size)
    results = annealing(function_to_optimize, xvalues, yvalues, max_temp)
    if(displayGraph):
        generate_graph(function_to_optimize, xvalues, yvalues, results[2], results[3], results[0], results[1], "Simulated Annealing", LaTeX_f)
    return (results[0], results[1])

# heavy lifting
def generate_graph(f, xvalues, yvalues, xpath, ypath, xmin, ymin, title, LaTeX_f):
    plot_surface(f, xvalues, yvalues)
    plot_3D_line(f, xpath, ypath)
#    plot_point(f, xmin, ymin)
    string = "minimum found at: (" + str.format(sigfigs, xmin) + ", " + str.format(sigfigs, ymin) + ", " + str.format(sigfigs, f(xmin, ymin)) + ')'
    LaTeX_title = r"""\Large{""" + title + """} \n \large{""" + string + """}"""
    plt.title(LaTeX_title)
    plt.subplots_adjust(top=3.0)
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.text2D(0.5, 0.85, LaTeX_f, transform=axes.transAxes, fontsize=18, ha="center")
    plt.show()

# plots the min point
'''
def plot_point(f, x, y):
#   does not work well with surfaces
    axes.scatter([x], [y], zs=[f(x, y) + 0.25], marker='*', s=200, c='yellow')
'''
# plots the path
def plot_3D_line(f, xvalues, yvalues):
    axes.plot(xvalues, yvalues, f(xvalues, yvalues), c='r')

# plots the surface
def plot_surface(f, xvalues, yvalues):
    X, Y = np.meshgrid(xvalues, yvalues)
    Z = f(X, Y)
    axes.plot_surface(X, Y, Z, cmap=cm.winter, linewidth=0)
    
# finds the best neighbor of (X, Y) and returns index directions to it
# format: ([-1 | 0 | 1], [-1 | 0 | 1])
# (0, 0) indicates (X, Y) is the best solution compared to neighbors
def get_direction(f, xvalues, yvalues, X, Y):
    minimum = f(xvalues[X], yvalues[Y])
    direction = (0, 0)
    if(X < len(xvalues) - 1):
        if(f(xvalues[X + 1], yvalues[Y]) < minimum):
            minimum = f(xvalues[X + 1], yvalues[Y])
            direction = (1, 0)
        if(Y < len(yvalues) - 1):
            if(f(xvalues[X + 1], yvalues[Y + 1]) < minimum):
                minimum = f(xvalues[X + 1], yvalues[Y + 1])
                direction = (1, 1)
        if(Y > 0):
            if(f(xvalues[X + 1], yvalues[Y - 1]) < minimum):
                minimum = f(xvalues[X + 1], yvalues[Y - 1])
                direction = (1, -1)
    if(X > 0):
        if(f(xvalues[X - 1], yvalues[Y]) < minimum):
            minimum = f(xvalues[X - 1], yvalues[Y])
            direction = (-1, 0)
        if(Y < len(yvalues) - 1):
            if(f(xvalues[X - 1], yvalues[Y + 1]) < minimum):
                minimum = f(xvalues[X - 1], yvalues[Y + 1])
                direction = (-1, 1)
        if(Y > 0):
            if(f(xvalues[X - 1], yvalues[Y - 1]) < minimum):
                minimum = f(xvalues[X - 1], yvalues[Y - 1])
                direction = (-1, -1)
    if(Y < len(yvalues) - 1):
        if(f(xvalues[X], yvalues[Y + 1]) < minimum):
            minimum = f(xvalues[X], yvalues[Y + 1])
            direction = (0, 1)
    if(Y > 0):
        if(f(xvalues[X], yvalues[Y - 1]) < minimum):
            direction = (0, -1)
    return direction

# hill climbing
def easy_hill_climb(f, xvalues, yvalues, X, Y):
    minimum = f(xvalues[X], yvalues[Y])
    pathx = [xvalues[X]]
    pathy = [yvalues[Y]]
    
    direction = get_direction(f, xvalues, yvalues, X, Y)
    while(direction != (0, 0)):
        X = X + direction[0]
        Y = Y + direction[1]
        minimum = f(xvalues[X], yvalues[Y])
        pathx.append(xvalues[X])
        pathy.append(yvalues[Y])
        direction = get_direction(f, xvalues, yvalues, X, Y)
    
    return(xvalues[X], yvalues[Y], pathx, pathy)


def random_hill_climb(f, xvalues, yvalues, num_restarts, displayGraph):
    startX = randomIndex(xvalues)
    startY = randomIndex(yvalues)
    globalmin = easy_hill_climb(f, xvalues, yvalues, startX, startY)
    if(displayGraph):
        plot_3D_line(f, globalmin[2], globalmin[3])
    count = 1;
    while(count < num_restarts):
        count += 1
        startX = randomIndex(xvalues)
        startY = randomIndex(yvalues)
        localmin = easy_hill_climb(f, xvalues, yvalues, startX, startY)
        if(f(localmin[0],localmin[1]) < f(globalmin[0], globalmin[1])):
            globalmin = localmin
        if(displayGraph):
            plot_3D_line(f, localmin[2], localmin[3])
    return globalmin


def annealing(f, xvalues, yvalues, max_temp):
    X = randomIndex(xvalues)
    Y = randomIndex(yvalues)

    globalmin = f(xvalues[X], yvalues[Y])
    globalX = xvalues[X]
    globalY = yvalues[Y]
    T = max_temp
    pathx = [xvalues[X]]
    pathy = [yvalues[Y]]
    
    while(T > 0.0001):
        direction = random_direction()
        while(X + direction[0] > len(xvalues) - 1 or X + direction[0] < 0 or Y + direction[1] > len(yvalues) - 1 or Y + direction[1] < 0):
            direction = random_direction()
        if(f(xvalues[X + direction[0]], yvalues[Y + direction[1]]) < f(xvalues[X], yvalues[Y])):
            X = X + direction[0]
            Y = Y + direction[1]
            if f(xvalues[X], yvalues[Y]) < globalmin:
                globalmin = f(xvalues[X], yvalues[Y])
                globalX = xvalues[X]
                globalY = yvalues[Y]
            pathx.append(xvalues[X])
            pathy.append(yvalues[Y])
        else:
            if(accept(f, xvalues[X], yvalues[Y], xvalues[X + direction[0]], yvalues[Y + direction[1]], T)):
                X = X + direction[0]
                Y = Y + direction[1]
                pathx.append(xvalues[X])
                pathy.append(yvalues[Y])
#        T -= COOLING_RATE
        T -= (T/max_temp)/COOLING_RATE
    
    return (globalX, globalY, pathx, pathy)


def random_direction():
    direction = (0, 0)
    while(direction == (0,0)):
        direction = (np.random.random_integers(-1, 1), np.random.random_integers(-1, 1))
    return direction

def accept(f, x1, y1, x2, y2, T):
    # print(str(f(x1,y1)) + ' -> ' + str(f(x2,y2)))
    probability = np.exp((f(x1, y1) - f(x2,y2)) / T)
    # print(str(probability) + '\n')
    return (probability > np.random.random())
