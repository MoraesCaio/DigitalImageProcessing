from PIL import Image


class Filter(object):
    matrix = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
    args = []


def channel(img, x, y, i):
    img.matrix[x, y] = tuple([0] * i + [img.matrix[x, y][i]] + ([0] * (len(img.matrix[x, y]) - (i + 1))))


def channel_r(img, x, y):
    # (r,g,b,[a]) -> (r,0,0,[0])
    channel(img, x, y, 0)


def channel_g(img, x, y):
    # (r,g,b,[a]) -> (0,g,0,[0])
    channel(img, x, y, 1)


def channel_b(img, x, y):
    # (r,g,b,[a]) -> (0,0,b,[0])
    channel(img, x, y, 2)


def add_shine(img, x, y):
    if len(Filter.args) >= 1 and Filter.args[0] >= 0:
        img.matrix[x, y] = tuple(
            (int(z) + Filter.args[0]) if int(z) + Filter.args[0] <= 255 else 255 for z in img.matrix[x, y])


def mult_shine(img, x, y):
    if len(Filter.args) >= 1 and Filter.args[0] >= 0.0:
        img.matrix[x, y] = tuple(
            int(float(z) * Filter.args[0]) if float(z) * Filter.args[0] <= 255.0 else 255 for z in img.matrix[x, y])


def negative(img, x, y):
    img.matrix[x, y] = tuple(255 - x for x in img.matrix[x, y])


def set_y_luma(img, x, y):
    if len(Filter.args >= 1) and Filter.args[0] <= 255.0:
        vec = img.matrix[x, y][0]
        y_luma, i, q = to_yiq(vec[0], vec[1], vec[2])
        img.matrix[x, y] = tuple(to_rgb(Filter.args[0], i, q) + (vec[3:] if len(vec > 2) else []))


def to_yiq(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b
    return y, i, q


def to_rgb(y, i, q):
    r = max(int(1.000 * y + 0.956 * i + 0.621 * q), 255)
    g = max(int(1.000 * y - 0.272 * i - 0.647 * q), 255)
    b = max(int(1.000 * y - 1.106 * i + 1.703 * q), 255)
    return r, g, b


class Editor(object):
    image = None
    matrix = None

    def __init__(self, image):
        self.image = image
        self.matrix = self.image.load()

    def set_image(self, image):
        self.image = image
        self.matrix = image.load()

    def iterate(self, function):
        for y in range(self.image.height):
            for x in range(self.image.width):
                function(self, x, y)

    @staticmethod
    def show_pixel(pixel):
        print("RGB: %s, %s, %s" % (str(pixel[0]), str(pixel[1]), str(pixel[2])))


def float_list(start, stop, step):
    result = [start]
    while start + step < stop:
        start = start + step
        result.append(start)
    return result
