
import importlib

import numpy as np


def safe_import(name, package=None):
    try:
        module = importlib.import_module(name, package)
    except (ImportError, ModuleNotFoundError):
        module = None
    return module


def cast_to_compatible_arrays(*args):
    # .. cast inputs to np.array with, at least, dimension 1, except when
    #    they are of type str or bool, that they are left unchanged
    args_ar = [a if isinstance(a, (str, bool)) else np.atleast_1d(a)
               for a in args]

    # .. get the shape of the input arguments
    def get_shape(a):
        try:
            return a.shape
        except AttributeError:
            return None

    input_shapes = list(set(filter(None, map(get_shape, args_ar))))

    # Now, there are various options:
    #  1. All inputs have the same shape. Then, len(input_shapes) == 1
    #  2. There are inputs with different shapes. Then:
    #     A. If len(input_shapes) == 2:
    #        a. One of the shapes is (1,). Thus, one or various
    #           inputs are scalar
    #        b. None of the shapes is (1,). Thus, there are
    #           arrays with mismatching shapes. Raise error.
    #     B. If len(input_shapes) > 2:
    #        There are arrays with mismatching shapes. Raise error.

    if len(input_shapes) > 2:
        msg = ', '.join(map(repr, input_shapes))
        raise ValueError(f'mismatch in input arrays with shapes {msg}')

    # .. reshape the scalar inputs to the shape of the array inputs.
    #    Because the fortran routine only admits 1d arrays, I directly
    #    expand the scalar inputs as a 1-d array with the appropriate
    #    number of elements.
    #    Within this block of code, I also convert nan values to MISSING_VALUE
    #    so that missings are properly accounted for in the fortran routine
    if len(input_shapes) == 2:
        if (1,) not in input_shapes:
            msg = ', '.join(map(repr, input_shapes))
            raise ValueError(f'mismatch in input arrays with shapes {msg}')

        input_shapes.pop(input_shapes.index((1,)))

    input_shape = input_shapes[0]

    for k, this_arg in enumerate(args_ar):
        if isinstance(this_arg, (str, bool)):
            continue
        if this_arg.size == 1:
            args_ar[k] = np.full(input_shape, this_arg)
        else:
            args_ar[k] = this_arg.reshape(input_shape)

    def restore_shape(ar):
        if input_shape == (1,):
            return ar.item()
        return np.reshape(ar, input_shape)

    return args_ar + [restore_shape]


def altitude_to_pressure(z, zo=0., Po=1013.25):
    """
    Calculate atmospheric pressure from altitude with the hypsometric
    equation (hydrostatic atmosphere + ideal gas state equation)

    Parameters
    ----------
      z : array-like
          Ground altitude, in meters above mean sea level
      zo : float
          Ground altitude reference level, in meters above mean sea level
      Po : float
          Atmospheric pressure at z=zo, in hPa
    """
    return Po * np.exp(-(z-zo)/8419.)  # T=15 K
