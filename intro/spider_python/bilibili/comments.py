from bilbili_spider import *

bv = "BV1Re4y1x72f"
file = f'D:/temp_files/{bv}_comments.csv'
# get_comments_n_save(bv)
import pandas as pd
a = pd.read_csv(file)
print(type(a))


