import unittest
import numpy as np
import datetime
import pandas as pd
import logging
import matplotlib.pyplot as plt
from onverify.core.verifyframe import VerifyFrame, VerifyFrameMulti
from onverify.core.taylorDiagram import df2taylor
from create_test_data import create_test_data, create_test_data_sin

savepdf=True
if savepdf:
    from matplotlib.backends.backend_pdf import PdfPages
    plt.switch_backend('pdf')


class TestVerifySingle(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if savepdf: self.pdf_pages = PdfPages('test_single.pdf')

    def setUp(self):
        self.df = create_test_data_sin(n=100, nmod=1)
        # At least plot_density_scatter KDE seems to need a bit of noise...
        self.df['m1'] = self.df['m1'] + np.random.rand(self.df.shape[0]) * 0.01 - 0.005
        self.df['obs'] = self.df['obs'] + np.random.rand(self.df.shape[0]) * 0.01 - 0.005
        self.v = VerifyFrame(self.df[['obs','m1']],
                    lat=-50, lon=100,
                    ref_col='obs',
                    verify_col='m1',
                    verify_label='m1label',
                    )

    def test_scatter(self):
        ax = self.v.plot_scatter()
        self.v.add_regression(ax=ax)
        self.v.add_stats(ax=ax)
        ax.set_title('test_scatter')
        if savepdf: self.pdf_pages.savefig()

    def test_density_hexbin(self):
        ax = self.v.plot_density_hexbin()
        self.v.add_stats(ax)
        ax.set_title('test_hexbin')
        if savepdf: self.pdf_pages.savefig()

    def test_qq(self):
        ax = self.v.plot_qq()
        ax.set_title('test_qq')
        if savepdf: self.pdf_pages.savefig()

    def test_qq2(self):
        ax = self.v.plot_qq(marker='s', color='r')
        self.v.add_stats(ax=ax)
        ax.set_title('test_qq2')
        if savepdf: self.pdf_pages.savefig()

    def test_cdf(self):
        ax = self.v.plot_cdf()
        ax.set_title('test_cdf')
        if savepdf: self.pdf_pages.savefig()

    def test_density_contour(self):
        ax = self.v.plot_density_contour()
        ax.set_title('test_contour')
        if savepdf: self.pdf_pages.savefig()

    def test_scatter_density(self):
        ax = self.v.plot_density_scatter()
        self.v.add_regression(ax=ax,show_eqn=True)
        self.v.add_stats(ax=ax, loc=3)
        ax.set_title('test_scatter_density')
        if savepdf: self.pdf_pages.savefig()

    def test_contour_regression(self):
        ax = self.v.plot_density_contour()
        self.v.add_regression(ax)
        ax.set_title('test_contour_regression')
        if savepdf: self.pdf_pages.savefig()

    def test_timeseries(self):
        ax = self.v.plot_timeseries()
        ax.set_title('test_timeseries')
        if savepdf: self.pdf_pages.savefig()

    def test_timeseries_fillunder(self):
        ax = self.v.plot_timeseries(fill_under_obs=True)
        ax.set_title('test_timeseries_fillunder')
        if savepdf: self.pdf_pages.savefig()

    def test_stats_table(self):
        tb = self.v.stats_table()
        logging.info(tb)
        tb.to_csv('test_single_stats_table.csv')

    def tearDown(self):
        if not savepdf:
            plt.show()

    @classmethod
    def tearDownClass(self):
        if savepdf:
            self.pdf_pages.close()
        plt.close('all')


class TestVerifyMulti(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if savepdf: self.pdf_pages = PdfPages('test_multi.pdf')

    def setUp(self):
        self.df = create_test_data_sin(n=100, nmod=3)
        # At least plot_density_scatter KDE seems to need a bit of noise...
        self.df['m1'] = self.df['m1'] + np.random.rand(self.df.shape[0]) * 0.01 - 0.005
        self.df['obs'] = self.df['obs'] + np.random.rand(self.df.shape[0]) * 0.01 - 0.005
        self.v = VerifyFrameMulti(self.df[['obs','m1', 'm2', 'm3']],
                lat=-50, lon=100, var='tp',
                ref_col='obs',
                verify_cols=['m1','m2', 'm3'],
                verify_labels=['m1label','m2label', 'm3label'])

    def test_multi_qq(self):
        ax = self.v.plot_qq()
        ax.set_ylabel('%s [%s]' % (self.v.var,self.v.units))
        if savepdf: self.pdf_pages.savefig()

    def test_multi_cdf(self):
        self.v.plot_cdf()
        if savepdf: self.pdf_pages.savefig()

    def test_multi_timeseries(self):
        self.v.plot_timeseries()
        if savepdf: self.pdf_pages.savefig()

        self.v.plot_timeseries(fill_under_obs=True)
        if savepdf: self.pdf_pages.savefig()

    def test_multi_set(self):
        self.v.plot_set()
        if savepdf: self.pdf_pages.savefig()

    def test_multi_set_scatter_density(self):
        self.v.plot_set_scatter_density()
        if savepdf: self.pdf_pages.savefig()

    def test_multi_stats_table(self):
        tb = self.v.stats_table()
        logging.info(tb)
        tb.to_csv('test_multi_stats_table.csv')

    def test_multi_taylor(self):
        plot_colors = ['b', 'r', 'g', 'c', 'y']
        df2taylor(self.v, obslabel=self.v.ref_col, colors=plot_colors)

    def tearDown(self):
        if not savepdf:
            plt.show()

    @classmethod
    def tearDownClass(self):
        if savepdf:
            self.pdf_pages.close()
        plt.close('all')


if __name__ == '__main__':
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)

    unittest.main()
