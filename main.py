import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from features import *
from specialCharacterRemove import *

file_path = "/C:/Users/ASUS/Desktop/Thesis/dundee_corpus/english/sf03ma1p.dat"

# Assuming your data file has a header row, you can skip it with skiprows=1
sf03ma1p = pd.read_csv(file_path, sep='\s+', skiprows=1, names=['WORD', 'TEXT', 'LINE', 'OLEN', 'WLEN', 'XPOS', 'WNUM', 'FDUR', 'OBLP', 'WDLP', 'LAUN', 'TXFR'], encoding='windows-1252')

# Display the DataFrame
print(sf03ma1p.head())
