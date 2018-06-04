# import numpy as np
# from matplotlib import pyplot as plt
#
# x = np.linspace(0, 2*np.pi, 400)
# y1 = np.sin(x**2)
# y2 = np.sin(x)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(x, y1)
# ax2.plot(x, y2)
# ax1.set_title("Simple plot (sin(x))")
# ax2.set_title("Simple plot (sin(x^2))")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


mat = np.random.random((1000, 50, 50))
print(mat[0].shape)
def quick_play(dT = 10):
    fig, ax = plt.subplots()
    im = ax.imshow(mat[0], cmap = "gray")

    def init():
        im.set_data(mat[0])
        return im,

    def animate(i):
        im.set_data(mat[i])
        return im,

    ani = animation.FuncAnimation(fig, animate, frames = 100, init_func = init, interval = dT, blit = True)
    plt.show()

quick_play()