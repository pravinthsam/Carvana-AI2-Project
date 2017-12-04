import numpy as np
#import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import skimage as skimage
from skimage import data, io, filters, transform

input_size = 512
#random flip
def random_flip(img, mask, u=1):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
        mask = image.flip_axis(mask, 1)
    return img, mask

#rotate util
'''
this takes a theta in radians
ht is along x axis
wd is along y axis
'''
def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
#rotate that uses rotate Util
'''
optioanlly takes rotate limits
'''
def random_rotate(img, mask, rotate_limit=(-30, 30), u=0.8):
    if np.random.random() < u:
        theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = rotate(img, theta)
        mask = rotate(mask, theta)
    return img, mask

#shift util
def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

#shift method
'''
takes shift limits for ht and wd
'''
def random_shift(img, mask, w_limit=(-0.2, 0.2), h_limit=(-0.2, 0.2), u=0.7):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        mask = shift(mask, wshift, hshift)
    return img, mask

#zoom util
def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
#zoom method
'''
uses zoom util to zoom in ht and wd along the center
'''
def random_zoom(img, mask, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = zoom(img, zx, zy)
        mask = zoom(mask, zx, zy)
    return img, mask

#shear util
def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

#method for shear
'''
takes in shear intensity range and shears ht and wd along the center
'''
def random_shear(img, mask, intensity_range=(-0.3, 0.3), u=0.6):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = shear(img, sh)
        mask = shear(mask, sh)
    return img, mask

#random blur
def random_blur(img, mask, blurSigma = 4):
    blur_sigma = np.random.uniform(1, blurSigma)
    if blur_sigma > 0:
        img2 = skimage.filters.gaussian(img, sigma=blur_sigma, multichannel=True)
        mask2 = skimage.filters.gaussian(mask, sigma=blur_sigma, multichannel=True)
    return img2, mask2


#generic method for plotting
'''
takes image,mask,transfromed image, transformed mask as input
'''
def plot_img_and_mask_transformed(img, mask, img_tr, mask_tr):
    fig, axs = plt.subplots(ncols=4, figsize=(16, 4), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(mask[:, :, 0])
    axs[2].imshow(img_tr)
    axs[3].imshow(mask_tr[:, :, 0])
    for ax in axs:
        ax.set_xlim(0, input_size)
        ax.axis('off')
    fig.tight_layout()
    plt.show()
    

keyTransformation = {0:random_flip,1:random_rotate,2:random_shift,3:random_zoom,4:random_shear,5:random_blur}

def TransformImageMask(img, mask):
    images = []
    masks = []
    
    for i in range(6):
        resImage,resMask = keyTransformation[i](img,mask)
        images.append(resImage)
        masks.append(resMask)

    #for i in range(6):
     #   plot_img_and_mask_transformed(img,mask,images[i],masks[i])
    return images, masks

#img = image.load_img(r'C:\ML\image_augmentation\train_masks.csv\train\11fcda0a9e1c_07.jpg',target_size=(512, 512))
#img = image.img_to_array(img)
#mask = image.load_img(r'C:\ML\image_augmentation\train_masks.csv\train_masks\11fcda0a9e1c_07_mask.gif',grayscale=True, target_size=(512, 512))
#mask = image.img_to_array(mask)
#img, mask = img / 255., mask / 255.
  
#TransformImageMask(img,mask)
