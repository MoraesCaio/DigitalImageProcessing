import sys
import dip
from dip import Filter
from dip import ImageMatrix
from dip import Routine
import numpy as np


img = Routine(sys.argv[1])
expression = 'img.{0}({1})'.format(sys.argv[2], ', '.join(sys.argv[3:]))
eval(expression)
