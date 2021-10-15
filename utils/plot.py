import numpy as np
import matplotlib.pyplot as plt

def plot_fields_tri(tri, show=False, **kwargs):
    """
    :param fields of shape (n, m, *).
    """
    assert len(fields.shape) > 2
    nrows, ncols = fields.shape[:2]
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols*5, nrows*5))
    

    for i, j in np.ndindex(fields.shape[:2]):
        axes[i, j].tripcolor(tri, fields[i, j], shading='gouraud')
        axes[i, j].set_xlim(left=-0.5, right=1.5)
        axes[i, j].set_ylim(bottom=-1, top=1)     
    
    if title:fig.suptitle(title)
    if show: fig.show()
    return fig

def plot_fields_image(fields: np.ndarray, show=False, **kwargs):
    """
    :param fields of shape (n, m, h, w).
    """
    nrows, ncols = fields.shape[:2]
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols*5, nrows*5))

    for i, j in np.ndindex(fields.shape[:2]):
        axes[i, j].imshow(fields[i, j])        
    
    if title:fig.suptitle(title)
    if show: fig.show()
    return fig

def plot_fields_quad(X:np.ndarray, fields: np.ndarray, show=False, **kwargs): 
    """
    :param X: x and y coordinates of the quad mesh. 2*h*w.
    :param fields: n*m*h*w array.
    """
    xcoord, ycoord = X
    nrows,ncols = fields.shape[:2]
    xlim = kwargs.get('xlim', (-1, 2))
    ylim = kwargs.get('ylim', (-1, 1))
    title = kwargs.get('title', '')
    xlabels = kwargs.get('xlabels')
    ylabels = kwargs.get('ylabels')
    if xlabels: assert len(xlabels)==ncols
    if ylabels: assert len(ylabels)==nrows

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols*5, nrows*5))
    for i, j in np.ndindex(fields.shape[:2]):
        # axes[i, j].contour(xcoord, ycoord, uvp, levels=20, linestyles='-', linewidths=0.4, colors='k')
        axes[i, j].pcolormesh(xcoord, ycoord, fields[i, j], cmap='jet', shading='gouraud', antialiased=True, snap=True, rasterized=True)
        axes[i, j].set_xlim(xlim)
        axes[i, j].set_ylim(ylim)
    
    if xlabels:
        for j in range(ncols):
            axes[-1, j].set_xlabel(xlabels[j])
    if ylabels:
        for j in range(ncols):
            axes[i, 0].set_ylabel(xlabels[j])        
    
    if title:fig.suptitle(title)
    if show: fig.show()
    return fig