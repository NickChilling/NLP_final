import sys
import os 

path = os.path.dirname(os.path.dirname(__file__))
# print(path)
# print(__file__)
# 实现时把文件名和基类改成要用的文件和类即可
sys.path.append(path)
import Common.Utils.sample_generator as sp
from Common.Models.model import BaseModel
samples = sp.wordseg_sample_generator(50,'train')
print(next(samples))






# raise NotImplementedError()
