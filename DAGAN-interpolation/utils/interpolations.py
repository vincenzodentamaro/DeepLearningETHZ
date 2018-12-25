import numpy as np
import tensorflow as tf
from scipy.stats import norm

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def lerp_gaussian(val, low, high):
    """Linear interpolation with gaussian CDF"""
    low_gau = norm.cdf(low)
    high_gau = norm.cdf(high)
    lerped_gau = lerp(val, low_gau, high_gau)
    return norm.ppf(lerped_gau)

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    threshold = 1e-7
    dot_prod = np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high))
    dot_prod = np.maximum(-(1 - threshold) * np.ones(dot_prod.shape),
                          np.minimum(dot_prod, (1 - threshold) * np.ones(dot_prod.shape)))
    omega = np.arccos(dot_prod)
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def slerp_tf(val, low, high):
    if val <= 0:
        return low
    elif val >= 1:
        return high
    low_flat = tf.reshape(low,[-1,])
    high_flat = tf.reshape(high, [-1,])
    dot_prod = tf.reduce_sum( tf.multiply( low_flat/tf.norm(low_flat, axis=0,keepdims=True),
                                           high_flat/tf.norm(high_flat, axis=0,keepdims=True) ), 0, keepdims=True)
    threshold = 1e-7
    dot_prod = tf.maximum(-(1-threshold)*tf.ones(dot_prod.shape),
                          tf.minimum(dot_prod, (1-threshold)*tf.ones(dot_prod.shape)))
    omega = tf.acos(dot_prod)
    so = tf.sin(omega)
    interpolation = tf.sin((1.-val)*omega)/so*low_flat + tf.sin(val*omega)/so*high_flat
    return tf.reshape(interpolation, high.shape)



def slerp_gaussian(val, low, high):
    """Spherical interpolation with gaussian CDF (generally not useful)"""
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
    epsilon = 0.001
    clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
    result = norm.ppf(clipped_sum)
    return result

def get_interpfn(spherical, gaussian):
    """Returns an interpolation function"""
    if spherical and gaussian:
        return slerp_gaussian
    elif spherical:
        return slerp
    elif gaussian:
        return lerp_gaussian
    else:
        return lerp

def create_mine_grid(rows, cols, dim, space, anchors, spherical, gaussian, scale=1.):
    """Create a grid of latents with splash layout"""
    lerpv = get_interpfn(spherical, gaussian)

    u_list = np.zeros((rows, cols, dim))
    # compute anchors
    cur_anchor = 0
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space == 0:
                if anchors is not None and cur_anchor < len(anchors):
                    u_list[y,x,:] = anchors[cur_anchor]
                    cur_anchor = cur_anchor + 1
                else:
                    u_list[y,x,:] = np.random.normal(0,scale, (1, dim))
    # interpolate horizontally
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space != 0:
                lastX = space * (x // space)
                nextX = lastX + space
                fracX = (x - lastX) / float(space)
#                 print("{} - {} - {}".format(lastX, nextX, fracX))
                u_list[y,x,:] = lerpv(fracX, u_list[y, lastX, :], u_list[y, nextX, :])
    # interpolate vertically
    for y in range(rows):
        for x in range(cols):
            if y%space != 0:
                lastY = space * (y // space)
                nextY = lastY + space
                fracY = (y - lastY) / float(space)
                u_list[y,x,:] = lerpv(fracY, u_list[lastY, x, :], u_list[nextY, x, :])

    u_grid = u_list.reshape(rows * cols, dim)

    return u_grid

#print(create_mine_grid(rows=16, cols=16, dim=100, space=1, anchors=None, spherical=True, gaussian=True))

def create_interpolation_interval(input_a, input_b, batch_size):
    vals = np.linspace(0., 1., batch_size)
    interpolated_latent_space = []

    for i, val in enumerate(vals):
        if i == 0:
            interpolated_latent_space.append(input_a)
        elif i == batch_size-1:
            pass
        else:
            interpolated_latent_space.append(slerp_tf(val,input_a,input_b))
    interpolated_latent_space.append(input_b)
    interpolated_latent_space = tf.stack(interpolated_latent_space,axis=0)
    return interpolated_latent_space

def create_mine_vector(num_generations, z_dim, space=None):
    """
    :param num_generations: number of random latent space vectors in output
    :param z_dim: dimension of random latent space
    :param space: number of interpolations between the random latent space vectors
    :return:
    """
    if space == None:
        space = int(num_generations/3.)
    u_list = np.zeros((num_generations,z_dim))
    z_high = np.random.normal(size=(z_dim),loc=0.,scale=1.)
    for i in range(num_generations):
        if i%space == 0:
            z_low = z_high
            z_high = np.random.normal(size=(z_dim),loc=0.,scale=1.)
            u_list[i,:] = z_low
        else:
            val = float(i%space)/float(space)
            u_list[i,:] = slerp(val, z_low, z_high)
    return u_list