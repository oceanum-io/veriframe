import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import json
import datetime
import numpy as np
import numpy.ma as ma
from matplotlib.dates import DateFormatter
from matplotlib.font_manager import FontProperties
import argparse

altnames = {
    1: "ERS1",
    2: "ERS2",
    3: "ENVISAT",
    4: "TOPEX",
    5: "POSEIDON",
    6: "JASON1",
    7: "GFO",
    8: "JASON2",
    9: "CRYOSAT",
    10: "SARAL",
}


def getStats(sdate, edate, jsonglob, stat):
    tmpdate = sdate
    stats = {}
    print ("    Getting %s" % stat)
    ii = 0
    while tmpdate < edate:
        stats[ii] = {}
        stats[ii]["date"] = tmpdate
        stats[ii]["alts"] = {}
        try:
            with open(tmpdate.strftime(jsonglob)) as jsonf:
                data = json.load(jsonf)
                for alt in data.keys():
                    stats[ii]["alts"][altnames[float(alt)]] = data[alt][stat]
        except Exception as e:
            print(e)
        tmpdate = addmonth(tmpdate)
        ii += 1
    return stats


def getcla():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonglob", help="Path to json files")
    parser.add_argument(
        "-r2", "--jsonglob2", help="Path to json files of second run to compare"
    )
    parser.add_argument(
        "t1", metavar="t1", type=str, help="Start date: [20100101, 20130501]"
    )
    parser.add_argument(
        "t2", metavar="t2", type=str, help="End date: [20120101, 20140101]"
    )
    parser.add_argument(
        "-o", "--output", default="./timeseries.png", help="path for output file"
    )
    args = parser.parse_args()
    return args


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show default values in help and preserve line break in epilog"""

    pass


def default_parser():
    parser = argparse.ArgumentParser(
        description="Generates PDF documents with verification plots",
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "basedir", help="path for where validation data. PDF is saved here"
    )
    parser.add_argument(
        "-t", "--title", default="Satellite Verification", help="Slide title"
    )
    parser.add_argument(
        "-s", "--subtitle", default="Some Location", help="Slide subtitle"
    )
    parser.add_argument(
        "-c",
        "--cases",
        nargs="*",
        help="Test case names, space-separated. Live blank to use all folder in BASEDIR",
    )
    return parser


def addmonth(date, n=1):
    """
    add n+1 months to date then subtract 1 day
    to get eom, last day of target month
    """
    OneDay = datetime.timedelta(days=1)
    q, r = divmod(date.month + n, 12)
    eom = datetime.datetime(date.year + q, r + 1, 1) - OneDay
    if date.month != (date + OneDay).month or date.day >= eom.day:
        return eom
    return eom.replace(day=date.day)


def plot_timeseries(
    sdate, edate, run, stat, fig=None, legend=True, minobs=10, ylabel=None, **kwargs
):
    if fig == None:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.gcf()
    assignlabel = True
    if kwargs.has_key("label"):
        assignlabel = False
    stats = getStats(sdate, edate, run, stat)
    N = getStats(sdate, edate, run, "N")
    alts = []
    for key in stats.keys():
        for alt in stats[key]["alts"].keys():
            alts.append(alt)
    alts = set(alts)
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    for alt, col in zip(alts, colors):
        data = []
        dates = []
        for key in stats.keys():
            dates.append(stats[key]["date"])
            if stats[key]["alts"].has_key(alt):
                if N[key]["alts"][alt] > minobs:
                    data.append(stats[key]["alts"][alt])
                else:
                    data.append(np.NAN)
            else:
                data.append(np.NAN)
        if assignlabel:
            kwargs["label"] = alt
        kwargs["color"] = kwargs.get("color", col)
        plt.plot(dates, data, **kwargs)
        if not assignlabel:
            kwargs["label"] = None
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter("%d %b %Y "))
    fig.autofmt_xdate()
    if "ylabel" != None:
        plt.ylabel(ylabel)
    if legend:
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
    # plt.title(stat)
    return fig


def plot_percentiles(sdate, edate, run, **kwargs):
    ii = 512
    nn = 0
    fig = plt.figure(figsize=(10, 10))
    modlabel = "Model"
    obslabel = "Altimeter"
    legend = True
    for pc in (
        "50",
        "80",
        "90",
        "95",
    ):
        ax = fig.add_subplot(str(ii))
        plot_timeseries(
            sdate,
            edate,
            run,
            "Qy_" + pc,
            fig=fig,
            label=modlabel,
            legend=False,
            color="k",
            lw=1.5,
        )
        ax = plt.gca()
        colors = [line.get_color() for line in ax.lines]
        plot_timeseries(
            sdate,
            edate,
            run,
            "Qx_" + pc,
            fig=fig,
            label=obslabel,
            legend=legend,
            ylabel="$H_s (m)$",
            color="r",
            lw=1.5,
        )
        legend = False
        modlabel = None
        obslabel = None
        ax = plt.gca()
        # for kk,line in enumerate(ax.lines[len(colors):]):
        #     line.set_color(colors[kk])
        plt.title("$" + pc + "^{th}$ Percentile")
        ii += 1
        nn += 1


def plot_set(sdate, edate, run, minobs=2000, fig=None, **kwargs):
    if not fig:
        fig = plt.figure(figsize=(10, 10))
    fig.add_subplot("411")
    plot_timeseries(
        sdate,
        edate,
        run,
        "bias",
        legend=False,
        minobs=minobs,
        fig=fig,
        ylabel="Bias (m)",
        **kwargs
    )
    # plt.title("Bias (m)")
    fig.add_subplot("412")
    plot_timeseries(
        sdate,
        edate,
        run,
        "si",
        legend=False,
        minobs=minobs,
        fig=fig,
        ylabel="Scatter Index",
        **kwargs
    )
    # plt.title("Scatter Index")
    fig.add_subplot("413")
    plot_timeseries(
        sdate,
        edate,
        run,
        "rmsd",
        legend=False,
        minobs=minobs,
        fig=fig,
        ylabel="RMSE (m)",
        **kwargs
    )
    # plt.title("RMSE (m)")
    fig.add_subplot("414")
    plot_timeseries(
        sdate,
        edate,
        run,
        "r",
        legend=True,
        minobs=minobs,
        fig=fig,
        ylabel="R",
        **kwargs
    )
    # plt.title("R")
    return fig


def main():
    args = hc.getcla()


def test():
    t1 = datetime.datetime(2000, 1, 1)
    t2 = datetime.datetime(2000, 12, 1)
    return getStats(t1, t2, "bob", "bias")


def testts():
    t1 = datetime.datetime(2000, 1, 1)
    t2 = datetime.datetime(2000, 12, 1)
    return plot_timeseries(t1, t2, "bob", "bias", minobs=500)


def testset():
    t1 = datetime.datetime(2012, 1, 1)
    t2 = datetime.datetime(2012, 12, 1)
    return plot_set(t1, t2, "../out/st2/%Y%m.json")


if __name__ == "__main__":
    # bias=test()
    # testts()
    testset()
