from PIL import Image
import dip
from dip import Editor
from dip import Filter

filename = ['test.jpeg', 'test2.png']

# for i in range(0, 100, 10):
# x = float(i)*2.0/10.0 + 1.0

petala = Editor(Image.open(filename[0]))
Filter.args = [2]
petala.iterate(dip.negative)
petala.image.save('Negative.jpeg', 'JPEG')


# petala.image.show()

