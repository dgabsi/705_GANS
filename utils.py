from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os


def show_images_grid(images, title='', figsize=(10,6)):
    #images = images.cpu()
    # Denoromalize
    fig = plt.figure(figsize=figsize)
    images_grid = make_grid(images[:], padding=2, normalize=True)
    plt.imshow(images_grid.permute(1, 2, 0).squeeze())
    plt.title(title)
    plt.show()

    return images_grid

def plot_all_losses(all_gen_losses ,all_discr_losses, title, dir_name, filename,x_label):

    fig = plt.figure(figsize=(12, 8))
    plot_file = os.path.join(dir_name, filename + '.png')
    plt.plot(range(len(all_gen_losses)), all_gen_losses, color='blue', label='Generator')
    plt.plot(range(len(all_discr_losses)), all_discr_losses, color='orange', label='Discriminator')
    plt.xlabel(x_label)
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)
    plt.savefig(plot_file, dpi=fig.dpi)
    plt.show()


def plot_all_accuracies(all_fake_accuracies ,all_real_accuracies, title, dir_name, filename, x_label):
    fig = plt.figure(figsize=(12, 8))
    plot_file = os.path.join(dir_name, filename + '.png')
    plt.plot(range(len(all_fake_accuracies)), all_fake_accuracies, color='blue', label='Fake')
    plt.plot(range(len(all_real_accuracies)), all_real_accuracies, color='green', label='Real')
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(title)
    plt.savefig(plot_file, dpi=fig.dpi)
    plt.show()



def plot_inception_scores(all_inception_scores, labels, title, dir_name, filename, y_label):
    fig = plt.figure(figsize=(12, 8))
    plot_file = os.path.join(dir_name, filename + '.png')
    for scores, label in zip(all_inception_scores,labels):
        plt.plot(range(len(scores)), scores, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(plot_file, dpi=fig.dpi)
    plt.show()

