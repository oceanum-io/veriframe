import os
import pytest
import datetime
import pandas as pd

from onverify.veriframe import VeriFrame

HERE = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(HERE, '../sample_files')

class TestStatDataFrame(object):
    """Test stats methods from StatsDataFrame."""

    @classmethod
    def setup_class(self):
        """Attributes to instantiate object."""
        self.pickle_file = os.path.join(FILES_DIR, 'collocs.pkl')
        self.csv_file = os.path.join(FILES_DIR, 'collocs.csv')
        self.ref_col = 'hs_obs'
        self.verify_col = 'hs_mod'
        self.var = 'hs'

    def test_read_pickle(self):
        vf = VeriFrame.from_file(filename=self.pickle_file,
                                   kind='pickle',
                                   ref_col=self.ref_col,
                                   verify_col=self.verify_col,
                                   var=self.var,
                                   )

    def test_read_csv(self):
        dparser = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        vf = VeriFrame.from_file(filename=self.csv_file,
                                   kind='csv',
                                   ref_col=self.ref_col,
                                   verify_col=self.verify_col,
                                   var=self.var,
                                   sep='\t',
                                   parse_dates=[0],
                                   date_parser=dparser,
                                   index_col=0,
                                   )
