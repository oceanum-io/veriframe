import os
import pytest
import pandas as pd

from onverify.veriframe import VeriFrame

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


class TestStatDataFrame(object):
    """Test stats methods from VerifyFrame."""

    @classmethod
    def setup_class(self):
        """Read data and define objects."""
        self.df = pd.read_pickle(os.path.join(FILES_DIR, "collocs.pkl"))
        self.vf = VeriFrame(self.df, ref_col="hs_obs", verify_col="hs_mod")
        self.stats = {
            "bias": 0.101,
            "nbias": 0.054,
            "rmsd": 0.331,
            "nrmsd": 0.177,
            "si": 0.168,
            "mad": 0.243,
            "mrad": 0.148,
        }

    def test_bias(self):
        stat = self.vf.bias()
        assert stat == pytest.approx(self.stats["bias"], rel=0.01)

    def test_nbias(self):
        stat = self.vf.bias(norm=True)
        assert stat == pytest.approx(self.stats["nbias"], rel=0.01)

    def test_rmsd(self):
        stat = self.vf.rmsd()
        assert stat == pytest.approx(self.stats["rmsd"], rel=0.01)

    def test_nrmsd(self):
        stat = self.vf.rmsd(norm=True)
        assert stat == pytest.approx(self.stats["nrmsd"], rel=0.01)

    def test_si(self):
        stat = self.vf.si()
        assert stat == pytest.approx(self.stats["si"], rel=0.01)

    def test_mad(self):
        stat = self.vf.mad()
        assert stat == pytest.approx(self.stats["mad"], rel=0.01)

    def test_mrad(self):
        stat = self.vf.mrad()
        assert stat == pytest.approx(self.stats["mrad"], rel=0.01)
