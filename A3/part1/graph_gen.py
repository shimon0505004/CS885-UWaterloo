from matplotlib import pyplot as plt

import ppo
import ppopenalty

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    BETA_PREVIOUS = ppopenalty.BETA
    BETA_VALUES = [1, 5, 10]
    COLOR_VALUES = ['g', 'r', 'k']
    for i in range(3):
        BETA = BETA_VALUES[i]
        # Train for different seeds
        curves = []
        curvesc = []
        for seed in ppopenalty.SEEDS:
            R, Rc = ppopenalty.train(seed)
            curves += [R]
            curvesc += [Rc]

        label = "ppo-penalty-" + str(BETA)
        # Plot the curve for the given seeds
        ppopenalty.plot_arrays(ax[0], curves, COLOR_VALUES[i], label)
        ppopenalty.plot_arrays(ax[1], curvesc, COLOR_VALUES[i], label)

    BETA = BETA_PREVIOUS

    # Train for different seeds
    curves = []
    curvesc = []
    for seed in ppo.SEEDS:
        R, Rc = ppo.train(seed)
        curves += [R]
        curvesc += [Rc]

    # Plot the curve for the given seeds
    ppo.plot_arrays(ax[0], curves, 'b', 'ppo')
    ppo.plot_arrays(ax[1], curvesc, 'b', 'ppo')

    plt.legend(loc='best')
    plt.show()