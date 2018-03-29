from PIL import Image
import numpy as np


class Filter(object):
    kernels = {'identity':     np.array([[0,  0, 0], [ 0, 1,  0], [0,  0,  0]]),
               'sharpen':      np.array([[0, -1, 0], [-1, 5, -1], [0, -1,  0]]),
               'other_kernel': np.array([[0,  0, 0], [ 0, 1,  0], [0,  0, -1]]),
               'blur':         np.array([[1,  1, 1], [ 1, 1,  1], [1,  1,  1]]),
               'mean': np.array([[1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9]])
               }

    channels = {'k': [0, 0, 0], 'b': [0, 0, 1],
                'g': [0, 1, 0], 'c': [0, 1, 1],
                'r': [1, 0, 0], 'm': [1, 0, 1],
                'y': [1, 1, 0], 'w': [1, 1, 1]}


def channel(image, channel_array):
    channel_array = np.array(channel_array, dtype='uint8')
    product = np.copy(image)
    product[:, :, :3] = image[:, :, :3] * channel_array[:3]
    return Image.fromarray(product, 'RGBA')


def add_shine(image, operand):
    result = np.int16(np.copy(image))
    result[:, :, :3] += operand
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)), ('RGBA' if result.shape[2] == 4 else 'RGB'))


def mult_shine(image, operand):
    result = np.float64(np.copy(image))
    result[:, :, :3] *= operand
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)), ('RGBA' if result.shape[2] == 4 else 'RGB'))


def negative(image):
    # keeps alpha channel's values
    result = np.copy(image)
    result[:, :, :3] = 255 - result[:, :, :3]
    return Image.fromarray(result, ('RGBA' if result.shape[2] == 4 else 'RGB'))


def apply_kernel(image, kernel):
    kernel = np.flipud(np.fliplr(np.copy(kernel)))
    result = np.int16(np.copy(image))

    # adding margins
    hks = (kernel.shape[0]-1)/2  # half_kernel_size
    image_padded = np.zeros([image.shape[0]+2*hks, image.shape[1]+2*hks, image.shape[2]])
    image_padded[hks:-hks, hks:-hks] = image

    # extension padding on edges
    image_padded[0:hks, :] = image_padded[hks:hks+1, :]
    image_padded[:, 0:hks] = image_padded[:, hks:hks+1]
    image_padded[-1: -(hks+1):-1, :] = image_padded[-(hks+1):-(hks+2):-1, :]
    image_padded[:, -1: -(hks+1):-1] = image_padded[:, -(hks+1):-(hks+2):-1]
    image_padded = np.int16(image_padded)

    # kernel application
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            for c in range(3):
                result[y, x, c] = (image_padded[y:y+kernel.shape[0], x:x+kernel.shape[0], c] * kernel).sum()

    return Image.fromarray(np.uint8(np.clip(result, 0, 255)), ('RGBA' if result.shape[2] == 4 else 'RGB'))


def apply_kernel_float(image, kernel):
    kernel = np.flipud(np.fliplr(np.copy(kernel)))
    result = np.float64(np.copy(image))

    # adding margins
    hks = (kernel.shape[0]-1)/2  # half_kernel_size
    image_padded = np.zeros([image.shape[0]+2*hks, image.shape[1]+2*hks, image.shape[2]])
    image_padded[hks:-hks, hks:-hks] = image

    # extension padding on edges
    image_padded[0:hks, :] = image_padded[hks:hks+1, :]
    image_padded[:, 0:hks] = image_padded[:, hks:hks+1]
    image_padded[-1: -(hks+1):-1, :] = image_padded[-(hks+1):-(hks+2):-1, :]
    image_padded[:, -1: -(hks+1):-1] = image_padded[:, -(hks+1):-(hks+2):-1]
    image_padded = np.float64(image_padded)

    # kernel application
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            for c in range(3):
                result[y, x, c] = (image_padded[y:y+kernel.shape[0], x:x+kernel.shape[0], c] * kernel).sum()

    return Image.fromarray(np.uint8(np.clip(result, 0, 255)), ('RGBA' if result.shape[2] == 4 else 'RGB'))


# def set_y_luma(img, x, y):
#     if len(Filter.args) >= 1 and Filter.args[0] <= 255.0:
#         vec = img.matrix[x, y]
#         y_luma, i, q = to_yiq(vec[0], vec[1], vec[2])
#         img.matrix[x, y] = tuple(to_rgb(Filter.args[0], i, q) + (vec[3:] if len(vec) > 2 else []))


# def to_yiq(r, g, b):
#     y = 0.299 * r + 0.587 * g + 0.114 * b
#     i = 0.596 * r - 0.274 * g - 0.322 * b
#     q = 0.211 * r - 0.523 * g + 0.312 * b
#     return y, i, q


# def to_rgb(y, i, q):
#     r = valid_rgb(int(1.000 * y + 0.956 * i + 0.621 * q))
#     g = valid_rgb(int(1.000 * y - 0.272 * i - 0.647 * q))
#     b = valid_rgb(int(1.000 * y - 1.106 * i + 1.703 * q))
#     return r, g, b


def adapt(filename):
    image = Image.open(filename)
    if image.format == 'JPEG':
        if image.mode != 'RGB':
            image = image.convert('RGB')
    elif image.format == 'PNG':
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
    return np.array(image)
