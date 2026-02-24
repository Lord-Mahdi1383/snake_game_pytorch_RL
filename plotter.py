import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('training')
    plt.xlabel('num of games')
    plt.ylabel('scores')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.show(block=False)
    plt.pause(.1)