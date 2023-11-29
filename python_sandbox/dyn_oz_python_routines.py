import numpy as np
import matplotlib.pyplot as plt


class Wellenfeld:
    def __init__(self,xdim,ydim):
        self.mesh = np.meshgrid(np.arange(ydim),np.arange(xdim))
        self.eta = np.zeros_like(self.mesh[0])
        self.u   = np.zeros_like(self.mesh[0])
        self.v   = np.zeros_like(self.mesh[0])
        self.g = 9.81
        self.f = 1e-4


    def set_initial(self, initial_type = None, position = 'center', amplitude = 1, std = 1):
        if position == 'center':
            pos = [len(self.mesh[0][0])/2,len(self.mesh[0])/2]
        else: pos = position
        self.eta = initial_type(self.mesh,pos,amplitude = amplitude,std = std)
        self.u = -self.g/self.f * derive('y',self.eta)
        self.v = self.g/self.f * derive('x',self.eta)
        
    def plot_initals(self,separate_dims = False):
        n_plots = 2
        if separate_dims: n_plots+=1
        fig,ax = plt.subplots(1,n_plots,figsize = (n_plots*10,7))
        titles = ['Surface elevation', 'velocity field']
        if separate_dims: titles = titles = ['Surface elevation', 'u field','v field']
        ax[0].contourf(self.eta)
        if separate_dims: 
            ax[1].quiver(self.u,np.zeros_like(self.u))
            ax[2].quiver(np.zeros_like(self.v),self.v)
        else: ax[1].quiver(self.v,self.u)

        for i in range(n_plots):
            ax[i].set_title(titles[i],fontsize = 20)

def gaussian(mesh,position, amplitude, std):
    def ensure_tuple(param):
        return param if type(param) == list else [param,param]
    
    param_list =  [ensure_tuple(param) for param in [position, std]]

    return amplitude * np.exp( -1/2 * ( (mesh[0]-param_list[0][0])**2/param_list[1][0]+(mesh[1]-param_list[0][1])**2/param_list[1][1]))

def phi(mesh,position, amplitude, std):
    g = 9.81
    f = 1e-4
    return f/g * amplitude * (np.sin(2*np.pi*mesh[0]/(2*position[0]*std))*np.sin(2*np.pi*mesh[1]/(2*position[1]*std))) 

def derive(dim,var):
    der = np.zeros_like(var)
    if dim == 'x':
        der[1:-1] = (var[2:]-var[:-2])/2
    elif dim == 'y': 
        der[:,1:-1] = (var[:,2:]-var[:,:-2])/2
    return der

def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

        .. versionadded:: 1.6.0
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.

        .. versionadded:: 1.6.0
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.

        .. versionadded:: 1.17.0

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.zeros_like(y)
    array([0.,  0.,  0.])

    """
    res = empty_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    # needed instead of a 0 to get same result as zeros for for string dtypes
    z = zeros(1, dtype=res.dtype)
    multiarray.copyto(res, z, casting='unsafe')
    return res
