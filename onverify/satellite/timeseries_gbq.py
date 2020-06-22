import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.font_manager import FontProperties
from pandas import read_gbq
import pandas as pd

from onverify.satellite import utils

# from onverify.satellite.utils import altnames

sys.path.append("../")


altnames = {
    6: "ENVISAT",
    7: "TOPEX-POSEIDON",
    1: "GFO",
    4: "SARAL",
    7: "TOPEX",
    9: "ERS-2",
    0: "CRYOSAT-2",
    8: "ERS-1",
    1: "JASON-1",
    3: "JASON-3",
    2: "JASON-2",
}


# altnames = {
    # 1: "ERS1",
    # 2: "ERS2",
    # 3: "ENVISAT",
    # 4: "TOPEX",
    # 5: "POSEIDON",
    # 6: "JASON1",
    # 7: "GFO",
    # 8: "JASON2",
    # 9: "CRYOSAT",
# }


def test():
    # dset = "wave.glob05_era5_prod_monthly_stats"
    # dset = "wave.metocean_wind_stats"
    #dset = "wave.oceanum_wave_glob05_era5_v1_monthly_stats"
    dset = "wave.cawcr_wave_glob04_cfsr_monthly_stats"
    df = loadStats(dset)
    # df = pd.read_csv('test_data.csv', parse_dates=['time'], index_col=['time']).sort_index()
    plot_set(df, minobs=10)
    plt.savefig("out.png")
    plt.show()


def loadStats(dset, projectid="oceanum-dev", use_bqstorage_api=False):
    return read_gbq(
        "SELECT * FROM `{}`".format(dset),
        project_id=projectid,
        use_bqstorage_api=use_bqstorage_api,
        index_col="time",
    )
    # return pd.from_csv('test_data.csv')


def plot_set(
    df,
    minobs=400000,
    fig=None,
    stats=dict(bias="Bias (m)", si="Scatter Index", rmsd="RMSE (m)", r="R", N="N"),
    ** kwargs,
):
    if not fig:
        fig = plt.figure(figsize=(10, 10))
    # stats = dict(bias = dict(label="Bias (m)", ax=None),
    # si = dict(label = "Scatter Index", ax=None),
    # rmsd = dict(label="RMSE (m)", ax=None),
    # N = dict(label="N", ax=None)
    nn = 1
    length = len(stats)
    for key, value in stats.items():
        ax = fig.add_subplot(f"{length}1{nn}")
        for satellite in df.groupby("satellite"):
            try:
                label = altnames[int(satellite[0])]
            except Exception as e:
                label = "unknown"
            mask = satellite[1]["N"] > minobs
            satellite[1][key][mask].plot(label=label, ax=ax, grid=True, legend=False)
        plt.ylabel(value)
        nn += 1
    fontP = FontProperties()
    fontP.set_size("medium")
    lg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        fancybox=True,
        prop=fontP,
        shadow=False,
    )
    # lg=plt.legend(borderaxespad=0)
    # lg.get_frame().set_linewidth(0)
    lg.get_frame().set_alpha(0.2)
    return fig


def main():
    parser = utils.default_parser()
    parser.add_argument(
        "t1", metavar="t1", type=str, help="Start date: [20100101, 20130501]"
    )
    parser.add_argument(
        "t2", metavar="t2", type=str, help="End date: [20120101, 20140101]"
    )
    parser.add_argument("-o", "--output", default="timeseries", help="png output")
    args = parser.parse_args()
    sdate = datetime.datetime.strptime(args.t1, "%Y%m")
    edate = datetime.datetime.strptime(args.t2, "%Y%m")
    cases = args.cases if args.cases else os.walk(args.basedir).next()[1]
    fig = None
    if len(cases) > 1:
        labels = cases
        colors = ["r", "b", "g", "c", "m", "y", "k", "w"]
    else:
        labels = [None]
        colors = [None]
    print(
        "%s\nPopulating document with testcases in %s:\n%s\n %s\n"
        % (80 * "=", args.basedir, 80 * "=", cases)
    )
    for label, test, color in zip(cases, labels, colors):
        jsonglob = os.path.join(test, "colocs", "%Y%m.json")
        fig = utils.plot_set(
            sdate, edate, jsonglob, lw=1.5, label=label, color=color, fig=fig
        )
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    # main()
    tmp = test()
    # plt.show()
