import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from onverify.core.taylorDiagram import TaylorDiagram, df2taylor

savepdf=True
if savepdf:
    from matplotlib.backends.backend_pdf import PdfPages
    plt.switch_backend('pdf')

class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if savepdf: self.pdf_pages = PdfPages('test_taylor.pdf')

    def setUp(self):

        # Reference dataset
        self.x = np.linspace(0,4*np.pi,100)
        self.data = np.sin(self.x)

        # Models
        self.m1 = self.data + 0.2*np.random.randn(len(self.x))    # Model 1
        self.m2 = 0.8*self.data + .1*np.random.randn(len(self.x)) # Model 2
        self.m3 = np.sin(self.x-np.pi/10)                    # Model 3

    def tearDown(self):
        if not savepdf:
            plt.show()

    @classmethod
    def tearDownClass(self):
        if savepdf:
            self.pdf_pages.close()
        plt.close('all')

    def test_plot(self):

        refstd = self.data.std(ddof=1)           # Reference standard deviation
        # Compute stddev and correlation coefficient of models
        samples = np.array([ [m.std(ddof=1), np.corrcoef(self.data, m)[0,1]]
                             for m in (self.m1,self.m2,self.m3)])

        fig = plt.figure(figsize=(10,4))

        ax1 = fig.add_subplot(1,2,1, xlabel='X', ylabel='Y')
        # Taylor diagram
        dia = TaylorDiagram(refstd, 1.2*refstd, fig=fig, rect=122, label="Reference")

        colors = plt.matplotlib.cm.jet(np.linspace(0,1,len(samples)))

        ax1.plot(self.x,self.data,'ko', label='Data')
        for i,m in enumerate([self.m1,self.m2,self.m3]):
            ax1.plot(self.x,m, c=colors[i], label='Model %d' % (i+1))
        ax1.legend(numpoints=1, prop=dict(size='small'), loc='best')

        # Add samples to Taylor diagram
        for i,(stddev,corrcoef) in enumerate(samples):
            dia.add_sample(stddev, corrcoef, marker='s', ls='', c=colors[i],
                           label="Model %d" % (i+1))

        # Add RMS contours, and label them
        contours = dia.add_contours(colors='0.5')
        plt.clabel(contours, inline=1, fontsize=10)

        # Add a figure legend
        fig.legend(dia.samplePoints,
                   [ p.get_label() for p in dia.samplePoints ],
                   numpoints=1, prop=dict(size='small'), loc='upper right')
        if savepdf: self.pdf_pages.savefig()

    def test_df2taylor(self):
        df = pd.DataFrame
        d = {'obs': self.data, 'm1': self.m1, 'm2': self.m2, 'm3': self.m3}
        df = pd.DataFrame(data=d)
        df2taylor(df)
        if savepdf: self.pdf_pages.savefig()
        df2taylor(df, mod_cols=['m1', 'm3'])
        if savepdf: self.pdf_pages.savefig()
        df2taylor(df, mod_cols=['m1', 'm3'], mod_labels=['label', 'label2'])
        if savepdf: self.pdf_pages.savefig()


if __name__=='__main__':
    unittest.main()
