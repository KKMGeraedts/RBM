import numpy as np
import matplotlib.pyplot as plt

def draw_network(w, nv, nh, figsize=(7,7)):
    """
    Given the weights and biases of a two-layered network, draw it. 

    Parameters:
        w - weights + biases
        nv - visible layer size
        nh - hidden layer size
    """

    # Create figure
    plt.figure(figsize=figsize)
    n = nv + nh
    
    # Unpack weights
    a = w[:nv]
    b = w[nv:n]
    w = w[n:]
    
    # lines between visible and hidden
    k = 0
    for i in range(nv):
        for j in range(nh):
            c = 'b' if w[k] > 0 else 'r'
            plt.plot([0,2],[-(nv-1)/2+i,-(nh-1)/2+j],'-',c=c, lw=abs(w[k]))
            k+=1

    # visible nodes
    for i in range(nv):
        c = 'b' if a[i] > 0 else 'r'
        plt.plot(0, -(nv-1)/2+i, 'o', ms=10*abs(a[i]), c=c)

    # hidden nodes
    for i in range(nh):
        c = 'b' if b[i] > 0 else 'r'
        plt.plot(2, -(nh-1)/2+i, 'o', ms=10*abs(b[i]), c=c)

def plot_weights_and_reconstruction_during_learning(weights, updates, cost, recon, figsize=(15, 10)):
    """
    Plot the norms of the weights and biases, and reconstruction during learning.

    Parameters:
        r - tuple containing (reconstructions errors, visible biases, hidden biases, weights) 
    """
    # Unpack weights and updates
    a, b, w = weights
    da, db, dw = updates

    # x-range for plots
    epochs = len(cost)
    x_epoch = np.arange(epochs)
    iterations = len(a)
    x_iterations = np.arange(iterations)

    # Create figure
    fig, ax = plt.subplots(2,2, figsize=figsize)
    ax = ax.ravel()

    # Reconstruction error
    ax[0].plot(x_epoch, recon)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("reconstruction error")
    ax[0].set_yscale("log")
    ax[0].set_title("Reconstruction error during learning")

    # Cost
    ax[1].plot(x_epoch, cost)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("cost")
    ax[1].set_title("Cost during learning")

    # Weight updates
    ax[2].plot(x_iterations, da, label="visible bias")
    ax[2].plot(x_iterations, db, label="hidden bias")
    ax[2].plot(x_iterations, dw, label="weights")
    ax[2].set_xlabel("epoch*batch_idx")
    ax[2].set_ylabel("L1 of weight gradients")
    ax[2].set_yscale("log")
    ax[2].set_title("Norms of the weight gradients during learning")

    # Weights
    ax[3].plot(x_iterations, a, label="visible bias")
    ax[3].plot(x_iterations, b, label="hidden bias")
    ax[3].plot(x_iterations, w, label="weights")
    ax[3].set_xlabel("epoch*batch_idx")
    ax[3].set_ylabel("L1 of weights")
    ax[3].set_title("Norms of the weights during learning")
    plt.legend()

def receptive_fields_v(W, nv, nh, figsize=(7, 8)):
  # Unpack weights
  n = nv + nh
  a = W[:nv]
  w = W[n:].reshape(nv, nh)

  # Image and plot size 
  plot_size = int(np.floor(np.sqrt(nh)))
  im_size = int(np.ceil(np.sqrt(nv)))

  # Create subplot
  fig, axs = plt.subplots(plot_size, plot_size, figsize=figsize)
  axs = axs.ravel()

  # Min and max values for colorbar
  vmin = np.min(w)
  vmax = np.max(w)

  for i in range(plot_size*plot_size):
      # Receptive field v_i
      wh = w[i, :]
      img = np.zeros(im_size*im_size)
      img[:len(wh)] = wh + a[i]
      img = img.reshape(im_size, im_size)
      
      # Plot
      im = axs[i].imshow(img, vmin=vmin, vmax=vmax)
      axs[i].set_title(f"v{i+1}")
      axs[i].set_xticks([])
      axs[i].set_yticks([])


  # Add a colorbar to the figure
  fig.colorbar(im, ax=axs.tolist())

def receptive_fields_h(W, nv, nh, figsize=(7, 8)):
    # Unpack weights
    n = nv + nh
    b = W[nv:n] * 0
    w = np.copy(W[n:].reshape(nv, nh))
    
    # Image and plot size 
    plot_size = int(np.floor(np.sqrt(nh)))
    im_size = int(np.ceil(np.sqrt(nv)))
    
    # Create subplot
    fig, axs = plt.subplots(plot_size, plot_size, figsize=figsize)
    axs = axs.ravel()

    # Renormalize the weights to be between [0, 1]
    for i, wi in enumerate(w.T):
        wi = np.abs(wi)
        wi = wi - min(wi)
        wi = wi / np.max(wi)
        w[:, i] = wi

    # Plot receptive fields
    for i in range(plot_size*plot_size):
        # Receptive field h_i
        wh = w[:, i]
        img = np.zeros(im_size*im_size)
        img[:len(wh)] = wh + b[i]
        img = img.reshape(im_size, im_size)
        
        # Plot
        im = axs[i].imshow(img)
        axs[i].set_title(f"h{i+1}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # Add a colorbar to the figure
    fig.colorbar(im, ax=axs.tolist())
    plt.show()

    # Summed weights in a single image
    w = np.sum(w, axis=1)
    vmax = np.max(w)
    im = plt.imshow(w.reshape(im_size, im_size), vmin=0, vmax=vmax)
    plt.colorbar(im)

def free_energies_during_learning(rbm, figsize=(18, 6)):
    # Extract free energies
    f_trains = rbm.f_trains
    f_valids = rbm.f_valids
    f_diffs = rbm.f_diffs

    x = np.arange(len(f_diffs))

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs = axs.ravel()

    # Overfitting
    axs[0].plot(x, f_diffs)
    axs[0].set_ylabel("$f_{valid} - f_{train}$")
    axs[0].set_xlabel("epoch")
    axs[0].set_title("Overfitting")

    # Free energy of train data
    axs[1].plot(x, f_trains)
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("free energy")
    axs[1].set_title("Free energy of train set during learning")

    # Free energy of validation data
    axs[2].plot(x, f_valids)
    axs[2].set_xlabel("epoch")
    axs[2].set_ylabel("free energy")
    axs[2].set_title("Free energy of test set during learning")

def receptive_histogram_and_spectra(rbm, data):
    nv = rbm.nv
    nh = rbm.nh
    W = rbm.torch_to_w()
    W = W[nv+nh:].reshape(nv, nh)

    fig, axs = plt.subplots(int(np.sqrt(nh)), int(np.sqrt(nh)), figsize=(16, 15))
    _, axs_eig = plt.subplots(1, 1)
    axs = axs.ravel()
    for i, wj in enumerate(W.T):
        axs[i].hist(wj)
        axs[i].set_title(f"$h_{{{i}}}$, mean $w_{{{i}}}$: {np.mean(wj):.2f}")

        idx = np.argwhere(wj[wj > 0.5 * np.mean(wj)]).reshape(-1)
        data_w = data[:, idx].T

        c = np.corrcoef(data_w)
        eigv, _= np.linalg.eigh(c)

        axs_eig.plot([i]*len(eigv), eigv, ".")

    axs_eig.set_yscale("log")
    
