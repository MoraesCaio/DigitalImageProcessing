from PIL import Image
import numpy as np
from os.path import splitext


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
    """
    Main class of dip, subclass of np.ndarray, contains methods for image processing. Uses PIL.Image and np.ndarrays.
     All of its methods do NOT alter the calling ImageMatrix object (self), instead a copy with the result processing is
     returned. This allows chaining methods without losing the original data of the image.
     (e.g.: img_mtx.grey().sobel() will NOT alter img_mtx object!)

     For creation of ImageMatrix objects, is recommended to use one of the following methods:
        - Using from_file(). (e.g.: my_img_mtx = ImageMatrix.from_file('party.jpg') or
        - Using np.array constructor and np.array.view() method. (e.g.: np.array(my_list).view(ImageMatrix))

     When working with YIQ system it's mandatory to use to_rgb() and to_yiq() when necessary!

     At the end of your work, you may want to see or save the result. With my_img = img_mtx.get_image(),
     you get a PIL.Image object. By doing so, you can see or save it using my_img.save('myEditedPhoto.jpg') and
     my_img.show().

    """

    # Subclassing numpy.array
    def __new__(cls, *args, **kwargs):
        return super(ImageMatrix, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        pass

    # INTERFACE METHODS
    @staticmethod
    def from_file(filename):
        """
        Creates an ImageMatrix object corresponding to filename.\n
        :param filename: string name of the file (including its extension).
        :return: ImageMatrix (subclass of numpy.ndarray) object of the RGB or RGBA pixels.
        """
        image = Image.open(filename)
        if image.format == 'JPEG':
            if image.mode != 'RGB':
                image = image.convert('RGB')
        elif image.format == 'PNG':
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
        return np.array(image).view(ImageMatrix)

    def get_image(self):
        """
        Creates a PIL.Image object with RGB or RGBA mode.\n
        :return: a PIL.Image object with RGB or RGBA mode
        """
        mode = 'RGBA' if self.shape[2] == 4 else 'RGB'
        return Image.fromarray(ImageMatrix.format_image_array(self), mode)

    # FORMATTING METHODS
    @staticmethod
    def format_image_array(image_array):
        """
        Formats ImageMatrix image_array (RGB/RGBA mode) onto a uint8 numpy.ndarray with values between 0 and 255.\n
        :param image_array: (RGB/RGBA mode) numpy.ndarray object
        :return: Formatted uint8 ImageMatrix
        """
        return np.uint8(np.clip(image_array, 0, 255)).view(ImageMatrix)

    @staticmethod
    def format_image_array_yiq(image_array):
        """
        Formats ImageMatrix image_array (YIQ mode) to YIQ's system thresholds. \n
        :param image_array: ImageMatrix with float64 values.
        :return: ImageMatrix with valid YIQ's values.
        """
        minimum = 255. * np.array([0.0, -0.596, -0.523])
        maximum = 255. * np.array([1.0, 0.596, 0.523])
        
        copy = np.copy(image_array)
        
        y = copy[:, :, 0]
        y[y < minimum[0]] = minimum[0]
        y[y > maximum[0]] = maximum[0]
        
        i = copy[:, :, 1]
        i[i < minimum[1]] = minimum[1]
        i[i > maximum[1]] = maximum[1]

        q = copy[:, :, 2]
        q[q < minimum[2]] = minimum[2]
        q[q > maximum[2]] = maximum[2]

        return np.float64(copy.view(ImageMatrix))

    # PROCESSING METHODS
    def channel(self, channel_array):
        """
        Iterates through ImageMatrix self multiplying each pixel's colors by int np.ndarray channel_array.\n
        :param channel_array: int np.array with shape (>=3, 1) (only 3 first elements will have effect).
        :return: Formatted uint8 ImageMatrix
        """
        channel_array = np.array(channel_array, dtype='uint8')
        product = np.copy(self)
        product[:, :, :3] = self[:, :, :3] * channel_array[:3]
        return ImageMatrix.format_image_array(product)

    def add_shine(self, operand):
        """
        Iterates through ImageMatrix self adding each pixel's colors an int operand.\n
        :param operand: int value
        :return: Formatted uint8 ImageMatrix
        """
        copy = np.int16(np.copy(self))
        copy[:, :, :3] += operand
        return ImageMatrix.format_image_array(copy)

    def add_shine_y(self, operand):
        """
        Iterates through ImageMatrix self adding each pixel's Y component (luma) an float operand.\n
        :param operand: float value
        :return: Formatted float64 ImageMatrix (YIQ mode)
        """
        copy = np.float64(np.copy(self))
        copy[:, :, 0] += operand
        return ImageMatrix.format_image_array_yiq(copy)

    def mult_shine(self, operand):
        """
        Iterates through ImageMatrix self multiplying each pixel's colors by an float operand.\n
        :param operand: float value
        :return: Formatted uint8 ImageMatrix
        """
        copy = np.float64(np.copy(self))
        copy[:, :, :3] *= operand
        return ImageMatrix.format_image_array(copy)

    def mult_shine_y(self, operand):
        """
        Iterates through ImageMatrix self multiplying each pixel's Y component (luma) by an float operand.\n
        :param operand: float value
        :return: Formatted uint8 ImageMatrix
        """
        copy = np.float64(np.copy(self))
        copy[:, :, 0] *= operand
        return ImageMatrix.format_image_array_yiq(copy)

    def negative(self):
        """
        Creates a ImageMatrix with complementary RGB components of self ImageMatrix.\n
        :return: Formatted uint8 ImageMatrix
        """
        # keeps alpha channel's values
        copy = np.copy(self)
        copy[:, :, :3] = 255 - copy[:, :, :3]
        return ImageMatrix.format_image_array(copy)

    def negative_y(self):
        """
        Creates a ImageMatrix with complementary Y component of self ImageMatrix.\n
        :return: Formatted float64 ImageMatrix
        """
        copy = np.copy(self.to_yiq()).view(self.__class__)
        copy[:, :, 0] = 1.0 - copy[:, :, 0]
        return copy

    def extension_padded(self, kernel):
        """
        Creates an ImageMatrix with a centralized self copy and edges with repeated values.\n
        :param kernel: np.ndarray kernel (won't be applied, for size references only).
        :return: Formatted uint8/float64 ImageMatrix
        """
        # adding margins
        hks = (kernel.shape[0]-1)/2  # half_kernel_size
        image_padded = np.zeros([self.shape[0] + 2 * hks, self.shape[1] + 2 * hks, self.shape[2]])
        image_padded[hks:-hks, hks:-hks] = np.copy(self)

        # extension padding on edges
        image_padded[0:hks, :] = image_padded[hks:hks+1, :]
        image_padded[:, 0:hks] = image_padded[:, hks:hks+1]
        image_padded[-1: -(hks+1):-1, :] = image_padded[-(hks+1):-(hks+2):-1, :]
        image_padded[:, -1: -(hks+1):-1] = image_padded[:, -(hks+1):-(hks+2):-1]

        return image_padded.view(self.__class__)

    def run_kernel_loop(self, image_padded, kernel):
        """
        Iterates through self ImageMatrix applying convolutional kernel.\n
        :param image_padded: ImageMatrix with adequate shape and content (see extension_padded()).
        :param kernel: np.ndarray kernel
        :return: uint8/float64 ImageMatrix
        """
        copy = np.copy(self)
        # kernel application
        for x in range(copy.shape[1]):
            for y in range(copy.shape[0]):
                for c in range(3):
                    copy[y, x, c] = (image_padded[y:y+kernel.shape[0], x:x+kernel.shape[0], c] * kernel).sum()

        return copy.view(self.__class__)

    def apply_kernel(self, kernel):
        """
        Routine for convolution with int kernels: flips the kernel, pads (extension) self ImageMatrix, applies kernel's
        convolution and formats the output.
        :param kernel: int np.ndarray kernel
        :return: uint8 ImageMatrix
        """
        kernel = np.flipud(np.fliplr(np.copy(kernel)))
        output = np.int16(np.copy(self)).view(self.__class__)

        image_padded = np.int16(self.extension_padded(kernel))

        return ImageMatrix.format_image_array(output.run_kernel_loop(image_padded, kernel))

    def apply_kernel_float(self, kernel):
        """
        Routine for convolution with float kernels: flips the kernel, pads (extension) self ImageMatrix, applies
        kernel's convolution and formats the output.\n
        :param kernel: float np.ndarray kernel
        :return: float64 ImageMatrix
        """
        kernel = np.flipud(np.fliplr(np.copy(kernel)))
        output = np.float64(np.copy(self)).view(self.__class__)

        image_padded = np.float64(self.extension_padded(kernel))

        return ImageMatrix.format_image_array(output.run_kernel_loop(image_padded, kernel))

    def median(self, kernel):
        """
        Apply median filter (kernel's content does not matter, only its contents).\n
        :param kernel: np.ndarray kernel
        :return: uint8 ImageMatrix
        """
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
        """
        Converts RGB/RGBA ImageMatrix into YIQ ImageMatrix.\n
        :return: float64 ImageMatrix
        """
        t = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
        fl_copy = np.float64(np.copy(self[:, :, :3]))
        result = fl_copy.dot(t.T) / 255
        return result.view(self.__class__)

    def to_rgb(self):
        """
        Converts YIQ ImageMatrix into RGB/RGBA ImageMatrix.\n
        :return: uint8 ImageMatrix
        """
        t = np.array([[1.000, 0.956, 0.621], [1.000, -0.272, -0.647], [1.000, -1.106, 1.703]])
        fl_copy = np.float64(np.copy(self[:, :, :3]))
        result = fl_copy.dot(t.T) * 255.0
        return ImageMatrix.format_image_array(result)

    def gx_component(self):
        """
        Returns self's Gx component (Sobel filter).\n
        :return: uint32 ImageMatrix
        """
        print('Calculating Gx component...')
        kernel = np.flipud(np.fliplr(np.copy(Filter.kernels['gx'])))
        gx = np.int32(np.copy(self)).view(self.__class__)
        image_padded = np.int32(self.extension_padded(kernel))
        return gx.run_kernel_loop(image_padded, kernel)

    def gy_component(self):
        """
        Returns self's Gy component (Sobel filter).\n
        :return: uint32 ImageMatrix
        """
        print('Calculating Gy component...')
        kernel = np.flipud(np.fliplr(np.copy(Filter.kernels['gy'])))
        gy = np.int32(np.copy(self)).view(self.__class__)
        image_padded = np.int32(self.extension_padded(kernel))
        return gy.run_kernel_loop(image_padded, kernel)

    def sobel(self, gx_gy_component=None):
        """
        Apply Sobel filter.\n
        :param gx_gy_component: ImageMatrix 2-tuple containing Gx and Gy components.
            If is None, calls gx_component() and gy_component().
        :return: uint8 ImageMatrix
        """
        if gx_gy_component is None:
            gx = self.gx_component()
            gy = self.gx_component()
        else:
            gx = gx_gy_component[0]
            gy = gx_gy_component[1]

        print('Calculating G...')
        g = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
        return ImageMatrix.format_image_array(g)

    def grey(self):
        """
        Returns a ImageMatrix monocromatic channel of self.\n
        :return: uint8 ImageMatrix
        """
        copy = np.copy(self)
        factor = np.array([299.0/1000, 587.0/1000, 114.0/1000])
        for x in range(self.shape[1]):
            for y in range(self.shape[0]):
                l = (copy[y, x] * factor).sum()
                copy[y, x, :3] = np.array([l, l, l])
        return ImageMatrix.format_image_array(copy)

    @staticmethod
    def is_float(data):
        """
        Auxiliary method for checking numpy.dtype is any kind of float.\n
        :param data: np.dtype value
        :return: Boolean True if data is any kind of numpy float.
        """
        if   isinstance(data, np.float16):
            return True
        elif isinstance(data, np.float32):
            return True
        elif isinstance(data, np.float64):
            return True
        return False

    def threshold_y(self, minimum=0.0, maximum=1.0):
        """
        Applies threshold values over Y (luma) component outliers.\n
        :param minimum: float minimum value for Y component.
        :param maximum: float maximum value for Y component.
        :return: float YIQ ImageMatrix
        """
        copy = np.copy(self).view(self.__class__)

        if not ImageMatrix.is_float(self[0, 0, 0]):
            return copy

        y = copy[:, :, 0]
        y[y < minimum] = minimum
        y[y > maximum] = maximum

        return copy

    def threshold_mean_y(self):
        """
        Replaces Y (luma) component values by its mean.\n
        :return: float YIQ ImageMatrix
        """
        copy = np.copy(self).view(self.__class__)

        if not ImageMatrix.is_float(self[0, 0, 0]):
            return copy

        y = copy[:, :, 0]
        mean = np.average(y)
        y[:, :] = mean

        return copy


