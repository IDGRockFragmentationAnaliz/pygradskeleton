import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects


def main():
    vizulize_simples()


def vizulize_separant():
    arr1 = np.array([
        [0, 7, 5],
        [1, 5, 9],
        [3, 3, 6]
    ])

    arr2 = np.array([
        [2, 3, 4],
        [5, 4, 4],
        [1, 2, 7]
    ])

    arr3 = np.array([
        [3, 7, 4],
        [6, 5, 8],
        [2, 9, 4]
    ])

    arr4 = np.array([
        [5, 1, 6],
        [1, 4, 3],
        [4, 2, 9]
    ])

    fig = plt.figure(figsize=(8, 2.5))
    axs = [fig.add_subplot(1, 4, 1),
           fig.add_subplot(1, 4, 2),
           fig.add_subplot(1, 4, 3),
           fig.add_subplot(1, 4, 4)]

    plot_table(axs[0], arr1)
    plot_table(axs[1], arr2)
    plot_table(axs[2], arr3)
    plot_table(axs[3], arr4)

    add_corner_label(axs[0], "a)")
    add_corner_label(axs[1], "b)")
    add_corner_label(axs[2], "c)")
    add_corner_label(axs[3], "d)")

    plt.tight_layout()
    plt.savefig("pictures/separant.png", dpi=300, bbox_inches="tight")
    plt.show()



def vizulize_simples():
    arr1 = np.array([
        [0, 7, 5],
        [1, 5, 9],
        [3, 3, 6]
    ])

    arr2 = np.array([
        [2, 3, 4],
        [5, 4, 4],
        [1, 2, 7]
    ])

    arr3 = np.array([
        [3, 7, 4],
        [6, 5, 8],
        [2, 9, 4]
    ])

    arr4 = np.array([
        [5, 1, 6],
        [1, 9, 3],
        [4, 2, 6]
    ])

    fig = plt.figure(figsize=(8, 2.5))
    axs = [fig.add_subplot(1, 4, 1),
           fig.add_subplot(1, 4, 2),
           fig.add_subplot(1, 4, 3),
           fig.add_subplot(1, 4, 4)]

    plot_table(axs[0], arr1)
    plot_table(axs[1], arr2)
    plot_table(axs[2], arr3)
    plot_table(axs[3], arr4)

    add_corner_label(axs[0], "a)")
    add_corner_label(axs[1], "b)")
    add_corner_label(axs[2], "c)")
    add_corner_label(axs[3], "d)")

    plt.tight_layout()
    plt.savefig("pictures/simple_and_not.png", dpi=300, bbox_inches="tight")
    plt.show()

def vizulize_clicue():
    arr1 = np.array([
        [-1, 7, -1],
        [-1, 5, 9],
        [-1, 8, -1]
    ])

    arr2 = np.array([
        [-1, 3, 4],
        [-1, 4, 4],
        [-1, -1, 7]
    ])

    arr3 = np.array([
        [-1, -1, 7],
        [-1, 6, 8],
        [-1, 9, -1]
    ])

    arr4 = np.array([
        [-1, 3, -1],
        [-1, 3, 6],
        [-1, -1, 9]
    ])

    fig = plt.figure(figsize=(8, 2.5))
    axs = [fig.add_subplot(1, 4, 1),
           fig.add_subplot(1, 4, 2),
           fig.add_subplot(1, 4, 3),
           fig.add_subplot(1, 4, 4)]

    plot_table(axs[0], arr1)
    plot_table(axs[1], arr2)
    plot_table(axs[2], arr3)
    plot_table(axs[3], arr4)

    # axs[0].set_title("a)")
    # axs[0].set_title("b)")
    # axs[0].set_title("c)")

    plt.tight_layout()
    plt.savefig("pictures/result.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_table(ax, arr, h=None):
    center = arr[1, 1]
    if h is None:
        h = center

    colors = np.zeros((3, 3), dtype=int)

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                colors[i, j] = 1          # центр — серый
            elif arr[i, j] < 0:
                colors[i, j] = 0          # отрицательные — белая база + штриховка
            elif arr[i, j] >= h:
                colors[i, j] = 2          # больше либо равно центру — черный
            else:
                colors[i, j] = 0          # меньше центра — белый

    cmap = ListedColormap(["white", "gray", "black"])

    ax.imshow(
        colors,
        cmap=cmap,
        vmin=0,
        vmax=2,
        zorder=0
    )

    # Штриховка для отрицательных значений
    for i in range(3):
        for j in range(3):
            if arr[i, j] < 0 and not (i == 1 and j == 1):
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="black",
                    hatch="///",
                    linewidth=0,
                    zorder=2
                )
                ax.add_patch(rect)

    # Сетка поверх всех ячеек, включая штрихованные
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)

    ax.set_axisbelow(False)

    ax.grid(
        which="minor",
        color="black",
        linewidth=2,
        zorder=5
    )

    # Убираем подписи осей
    ax.set_xticks([])
    ax.set_yticks([])

    # Убираем маленькие деления по краям
    ax.tick_params(which="minor", bottom=False, left=False)

    # Квадратные ячейки
    ax.set_aspect("equal")

    # Фиксируем границы изображения
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)

    # Подписи внутри ячеек
    for i in range(3):
        for j in range(3):
            label = "x" if arr[i, j] < 0 else str(arr[i, j])

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=18,
                color="black",
                path_effects=[
                    path_effects.withStroke(linewidth=3, foreground="white")
                ],
                zorder=10
            )
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_corner_label(ax, text):
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            boxstyle="square",
            linewidth=0
        ),
        zorder=20
    )

if __name__ == "__main__":
    main()