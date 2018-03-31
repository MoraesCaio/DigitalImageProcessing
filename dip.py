from PIL import Image
import numpy as np


class Filter(object):

    kernels = {'identity':     np.array([[0,  0,  0], [ 0,  1,  0], [ 0,  0,  0]]),
               'sharpen':      np.array([[0, -1,  0], [-1,  5, -1], [ 0, -1,  0]]),
               'other_kernel': np.array([[0,  0,  0], [ 0,  1,  0], [ 0,  0, -1]]),
               'blur':         np.array([[1,  1,  1], [ 1,  1,  1], [ 1,  1,  1]]),
               'gx':           np.array([[1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]]),
               'gy':           np.array([[1,  2,  1], [ 0,  0,  0], [-1, -2, -1]]),
               'log4n':        np.array([[0,  1,  0], [ 1, -4,  1], [ 0,  1,  0]]),
               'log8n':        np.array([[1,  1,  1], [ 1, -8,  1], [ 1,  1,  1]]),
               }

    kernels_float = {'mean': np.array([[1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9]])
                     }

    channels = {'k': [0, 0, 0], 'b': [0, 0, 1],
                'g': [0, 1, 0], 'c': [0, 1, 1],
                'r': [1, 0, 0], 'm': [1, 0, 1],
                'y': [1, 1, 0], 'w': [1, 1, 1]}


class ImageMatrix(np.ndarray):

    # Subclassing numpy.array
    def __new__(cls, *args, **kwargs):
        return super(ImageMatrix, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        pass

    # Interface methods
    @staticmethod
    def from_file(filename):
        image = Image.open(filename)
        if image.format == 'JPEG':
            if image.mode != 'RGB':
                image = image.convert('RGB')
        elif image.format == 'PNG':
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
        return np.array(image).view(ImageMatrix)

    def get_image(self):
        mode = 'RGBA' if self.shape[2] == 4 else 'RGB'
        return Image.fromarray(ImageMatrix.format_image_array(self), mode)

    @staticmethod
    def format_image_array(img_array):
        return np.uint8(np.clip(img_array, 0, 255)).view(ImageMatrix)

    # Processing methods
    def channel(self, channel_array):
        channel_array = np.array(channel_array, dtype='uint8')
        product = np.copy(self)
        product[:, :, :3] = self[:, :, :3] * channel_array[:3]
        return ImageMatrix.format_image_array(product)

    def add_shine(self, operand):
        result = np.int16(np.copy(self))
        result[:, :, :3] += operand
        return ImageMatrix.format_image_array(result)

    def mult_shine(self, operand):
        result = np.float64(np.copy(self))
        result[:, :, :3] *= operand
        return ImageMatrix.format_image_array(result)

    def negative(self):
        # keeps alpha channel's values
        result = np.copy(self)
        result[:, :, :3] = 255 - result[:, :, :3]
        return ImageMatrix.format_image_array(result)

    def extension_padded(self, kernel):
        # adding margins
        hks = (kernel.shape[0]-1)/2  # half_kernel_size
        image_padded = np.zeros([self.shape[0] + 2 * hks, self.shape[1] + 2 * hks, self.shape[2]])
        image_padded[hks:-hks, hks:-hks] = self

        # extension padding on edges
        image_padded[0:hks, :] = image_padded[hks:hks+1, :]
        image_padded[:, 0:hks] = image_padded[:, hks:hks+1]
        image_padded[-1: -(hks+1):-1, :] = image_padded[-(hks+1):-(hks+2):-1, :]
        image_padded[:, -1: -(hks+1):-1] = image_padded[:, -(hks+1):-(hks+2):-1]

        return image_padded

    def run_kernel_loop(self, image_padded, kernel):
        # kernel application
        for x in range(self.shape[1]):
            for y in range(self.shape[0]):
                for c in range(3):
                    self[y, x, c] = (image_padded[y:y+kernel.shape[0], x:x+kernel.shape[0], c] * kernel).sum()

        return self

    def apply_kernel(self, kernel):
        kernel = np.flipud(np.fliplr(np.copy(kernel)))
        output = np.int16(np.copy(self)).view(self.__class__)

        image_padded = np.int16(self.extension_padded(kernel))

        return ImageMatrix.format_image_array(output.run_kernel_loop(image_padded, kernel))

    def apply_kernel_float(self, kernel):
        kernel = np.flipud(np.fliplr(np.copy(kernel)))
        output = np.float64(np.copy(self)).view(self.__class__)

        image_padded = np.float64(self.extension_padded(kernel))

        return ImageMatrix.format_image_array(output.run_kernel_loop(image_padded, kernel))

    def median(self, kernel):
        hks = (kernel.shape[0] - 1) / 2
        output = np.int16(np.copy(self))

        image_padded = np.int16(self.extension_padded(kernel))

        # median application
        for x in range(self.shape[1]):
            for y in range(self.shape[0]):
                for c in range(3):
                    copy1d = image_padded[y:y+kernel.shape[0], x:x+kernel.shape[0], c].flatten()
                    copy1d.sort()
                    output[y, x, c] = copy1d[hks+1]
                    # does not work! kernel must not be applied over a backup array
                    # image_padded[y+hks, x+hks, c] = copy1d[hks+1]

        return ImageMatrix.format_image_array(output)

    def to_yiq(self):
        t = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
        fl_copy = np.float64(np.copy(self[:, :, :3]))
        result = fl_copy.dot(t.T) / 255
        return result.view(self.__class__)

    def to_rgb(self):
        t = np.array([[1.000, 0.956, 0.621], [1.000, -0.272, -0.647], [1.000, -1.106, 1.703]])
        fl_copy = np.float64(np.copy(self[:, :, :3]))
        result = fl_copy.dot(t.T) * 255.0
        return ImageMatrix.format_image_array(result)

# def set_y_luma(img, x, y):
#     if len(Filter.args) >= 1 and Filter.args[0] <= 255.0:
#         vec = img.matrix[x, y]
#         y_luma, i, q = to_yiq(vec[0], vec[1], vec[2])
#         img.matrix[x, y] = tuple(to_rgb(Filter.args[0], i, q) + (vec[3:] if len(vec) > 2 else []))
