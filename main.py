import dip
from dip import Filter
from dip import ImageMatrix
from dip import Routine
import numpy as np


filename = ['test.jpeg', 'test2.png', '1px.png', 'petala.png', 'SnPnoise.png', 'valve.png', 'bike.jpg',
            'xray.jpeg', 'xray2.jpeg']

# petala = ImageMatrix.from_file(filename[3])
# noise = ImageMatrix.from_file(filename[4])
# valve = ImageMatrix.from_file(filename[5])
# bike = ImageMatrix.from_file(filename[6])
# xray1 = ImageMatrix.from_file(filename[7])
# xray2 = ImageMatrix.from_file(filename[8])

petala = Routine("petala.png")
petala.threshold_y()
petala.threshold_mean_y()
