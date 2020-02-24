import os
import pytest
import pandas as pd

from onverify.veriframe import VeriFrame, VeriFrameMulti

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '../sample_files')

class TestStatDataFrame(object):
    """Test that stats attributes are maintained after slicing."""

    @classmethod
    def setup_class(self):
        """Read data and define objects."""
        self.df = pd.read_pickle(os.path.join(FILES_DIR, 'collocs.pkl'))
        self.df['hs_mod2'] = self.df['hs_mod'] * 2
        self.vf = VeriFrame(self.df, ref_col='hs_obs', verify_col='hs_mod')
        self.vfm = VeriFrameMulti(self.df,
                                    ref_col='hs_obs',
                                    verify_cols=['hs_mod', 'hs_mod2'])

    def test_type_veriframe(self):
        sliced = self.vf.iloc[0:10]
        assert isinstance(sliced, VeriFrame)

    def test_type_veriframemulti(self):
        sliced = self.vfm.iloc[0:10]
        assert isinstance(sliced, VeriFrameMulti)

    @pytest.mark.parametrize('attr_name', [
        'ref_col',
        'verify_col',
        'n',
        'bias',
        'rmsd',
        'si',
        'mad',
        'mrad',
        'ks',
        'plot_cdf',
        'plot_density_contour',
        'plot_density_hexbin',
        'plot_density_scatter',
        'plot_map',
        'plot_pdf',
        'plot_qq',
        'plot_regression',
        'plot_scatter',
        'plot_scatter_polar',
        'plot_scatter_qq',
        'plot_subtimeseries',
        'plot_timeseries',
        'stats_table',
        'pretty_stats_table',
        'add_stats',
        'add_regression',
        'from_file',
        'set_xylimit',
        'legend',
        'add_text',
    ])
    def test_slice_veriframe(self, attr_name):
        sliced = self.vf.iloc[0:10]
        assert getattr(sliced, attr_name, None) is not None

    @pytest.mark.parametrize('attr_name', [
        'ref_col',
        'verify_cols',
        'n',
        'bias',
        'rmsd',
        'si',
        'mad',
        'mrad',
        'ks',
        'plot_cdf',
        'plot_density_contour',
        'plot_density_hexbin',
        'plot_density_scatter',
        'plot_map',
        'plot_pdf',
        'plot_qq',
        'plot_regression',
        'plot_scatter',
        'plot_scatter_polar',
        'plot_scatter_qq',
        'plot_subtimeseries',
        'plot_timeseries',
        'stats_table',
        'pretty_stats_table',
        'add_stats',
        'add_regression',
        'from_file',
        'set_xylimit',
        'legend',
        'add_text',
    ])
    def test_slice_veriframemulti(self, attr_name):
        sliced = self.vfm.iloc[0:10]
        assert getattr(sliced, attr_name, None) is not None

