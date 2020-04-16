import sys

sys.path.append("../")
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.font_manager import FontProperties
import glob
import os

# from verify.core.site_tom import Verify
from onverify.satellite import utils


# def test():
# t1=datetime.datetime(2000,1,1)
# t2=datetime.datetime(2000,12,1)
# return plot_all(t1,t2,'bayofbengal',)

# def plot_all(sdate,edate,path):
# tmplfile=path+'/%Y%m%d.hdf5'
# files=[]
# tmpdate=sdate
# while tmpdate < edate:
# files.append(tmpdate.strftime(tmplfile))
# tmpdate=utils.addmonth(tmpdate)
# tmp=Verify(files)
# return tmp


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
    main()
    # tmp=test()
    # plt.show()
