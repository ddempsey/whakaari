
import os, shutil
from glob import glob

MOVE_DIR = r'U:\EruptionForecasting\eruptions\predictions'

def collate_consensus():
    dirs = glob('../predictions/transformed2_e*hires')

    for d in dirs:
        d1 = MOVE_DIR+os.sep+d.split(os.sep)[-1]
        os.makedirs(d1, exist_ok=True)
        fls = glob(d+os.sep+'consensus*.pkl')
        for fl in fls:
            shutil.copyfile(fl, d1+os.sep+fl.split(os.sep)[-1])

def main():
    collate_consensus()

if __name__ == "__main__":
    main()