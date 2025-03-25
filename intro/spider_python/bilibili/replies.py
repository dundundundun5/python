from bilbili_spider import *
if __name__ == '__main__':
    bv = "BV1Re4y1x72f"
    file = f'D:/temp_files/{bv}_replies.csv'
    # get_root_replies_n_save(bv)
    import pandas as pd
    a = pd.read_csv(file)
    input("")