from bilbili_spider import *
if __name__ == '__main__':
    bv = "BV11P4y1S7do"
    file = f'D:/temp_files/{bv}_replies.csv'
    get_root_replies_n_save(bv)
    import pandas as pd
    a = pd.read_csv(file)