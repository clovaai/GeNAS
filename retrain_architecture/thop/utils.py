'''
Copied from https://github.com/megvii-model/RLNAS/blob/main/darts_search_space/cifar10/rlnas/retrain_architecture/thop/utils.py
'''

def clever_format(num, format="%.2f"):
    if num > 1e12:
        return format % (num / 1e12) + "T"
    if num > 1e9:
        return format % (num / 1e9) + "G"
    if num > 1e6:
        return format % (num / 1e6) + "M"
    if num > 1e3:
        return format % (num / 1e3) + "K"