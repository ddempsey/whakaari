import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, to_nztimezone, datetimeify
from datetime import timedelta, datetime

def plot_gaps():
    f = plt.figure(figsize=[8,6])
    ax1 = plt.axes([0.1, 0.60, 0.8, 0.25])
    ax2 = plt.axes([0.1, 0.15, 0.8, 0.25])

    plt.savefig('data_gaps.png', dpi=400)

def main():
    plot_gaps()

if __name__ == "__main__":
    main()