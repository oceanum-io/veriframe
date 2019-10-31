import os
import yaml
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
import logging
from matplotlib import rc
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#from onverify.stats import add_stats,stats_table,stats_table_monthly, stats_table_groups
from onverify.core.regression import linear_regression
from onverify.core.taylorDiagram import df2taylor

from dateutil.relativedelta import relativedelta
import matplotlib.dates as dates


rc('text', usetex=False)


# Verify class ---------------------------


class Verify(object):
    """
    Create verifications plots from a pandas dataframe
    """
    def __init__(self, df,
                 obsname='obs',
                 modname='forecast',
                 obslabel=None,
                 modlabel=None,
                 var='hs',
                 vmin=None,
                 vmax=None,
                 sitelat=None,
                 sitelon=None,
                 sitename='None'
                 ):
        self.df = df
        self.obsname = obsname
        self.modname = modname
        self.modlabel = modlabel or modname
        self.obslabel = obslabel or obsname
        self.var = var
        self.vmin = vmin if vmin is not None else min(df[modname].min(), df[obsname].min())
        self.vmax = vmax if vmax is not None else max(df[modname].max(), df[obsname].max())
        self.sitelat, self.sitelon = sitelat, sitelon
        self.sitename = sitename
        # Put standard variables in qdict
        vardef = os.path.join(os.path.dirname(globals()['__file__']), 'vardef.yml')
        self.vardef = yaml.load(open(vardef))
        try:
            self.units = self.vardef['vars'][var]['units']
            self.long_name = self.vardef['vars'][var]['long_name']
        except KeyError:
            raise KeyError("Variable not implemented")

        self._rename_df_columns()

    def _rename_df_columns (self):
        ''' rename columns because plot dataframes doesn't recognize labels'''
        self.df = self.df.rename(columns = {self.obsname:self.obslabel, self.modname: self.modlabel})
        self.obsname = self.obslabel
        self.modname = self.modlabel

    def plot_map(self, buff=20, marker='ro', markercolor='r', markersize=8,
                 ax=None):
        logging.debug("Plotting map")
        plot_map(self.sitelat, self.sitelon, buff=buff, marker=marker,
                 markercolor=markercolor, markersize=markersize, ax=ax)


    def plot_timeseries(self, grid=True, plotobs=True, fill_under_obs=False, fill_under_alpha=0.3,
                        **kwargs):
        # if not double [[]] is a pd.Series and, with pandas 0.19.2:
        # 1. adds to previous existing figure
        # 2. error recognizing index if not previous figure

        logging.debug("Plotting timeseries")
        if plotobs:
            kwargs2 = {x: kwargs[x] for x in kwargs if x not in ['color', 'ls', 'lw']}
            if fill_under_obs:
                ax = self.df[[self.obsname]].plot(alpha=fill_under_alpha, color='k', label=self.obslabel, **kwargs2)
                # needs ax and doesn't add a label to legend
                ax.fill_between(self.df.index, 0, self.df[self.obsname], alpha=fill_under_alpha, color='k')
            else:
                ax = self.df[[self.obsname]].plot(grid=grid, color='k', ls='--', lw=2, label=self.obslabel, **kwargs2)
            kwargs.update(dict(ax=ax))

        ax = self.df[[self.modname]].plot(grid=grid, label=self.modlabel, **kwargs)
        L = legend(ax=ax)
        return ax


    def plot_scatter( self, grid=True, **kwargs):
        logging.debug("Plotting scatter")
        ax = self.df.plot(x=self.obsname,
                          y=self.modname,
                          kind='scatter',
                          xlim=[self.vmin, self.vmax],
                          ylim=[self.vmin, self.vmax],
                          grid=grid,
                          **kwargs)
        self.label(ax)
        return ax


    def label(self, ax, diag=True):
        tmp = np.arange(self.vmin,2 * self.vmax)
        if diag:
            ax.plot(tmp, tmp, color='gray', zorder=1, linestyle='--')
        ax.set_xlabel('%s [%s]' % (self.obslabel,self.units))
        ax.set_ylabel('%s [%s]' % (self.modlabel,self.units))

    def add_stats(self, ax, **kwargs):
        pos, step = add_stats(self.df, ax, self.obsname, self.modname, **kwargs)
        return pos, step

    def plot_hexbin(self, bins=None, gridsize=20, cmap=plt.get_cmap('gray_r'), grid=True, **kwargs):
        logging.debug("Plotting hexbin")
        ax = self.df.plot(
            x=self.obsname,
            y=self.modname,
            kind='hexbin',
            xlim=[self.vmin, self.vmax],
            ylim=[self.vmin, self.vmax],
            bins=bins,
            gridsize=gridsize,
            cmap=cmap,
            grid=grid,
            **kwargs)
        self.label(ax)
        return ax

    def plot_density(self, grid=True, plotobs=True, **kwargs):
        # if not double [[]] is a pd.Series and add to previous an existing figure (pandas 0.19.2)
        logging.debug("Plotting density")
        if plotobs:
            kwargs2 = {x: kwargs[x] for x in kwargs if x not in ['color', 'ls', 'lw']}
            kwargs.update(dict(ax=self.df[[self.obsname]].plot(kind='density', grid=grid, color='k', ls='--', lw=2,
                                                             label=self.obslabel, **kwargs2)))
        ax = self.df[[self.modname]].plot(kind='density', lw=1.5, grid=grid, label=self.modlabel, **kwargs)
        ax.set_xlim([self.vmin, self.vmax])
        L = legend(ax=ax, loc=1)
        # L.get_texts()[0].set_text(modlab+' '+self.units)
        # L.get_texts()[1].set_text(obslab+' '+self.units)
        return ax


    def plot_qq(self, grid=True, increment=0.01, **kwargs):
        logging.debug("Plotting qq")
        P = np.arange(0, 1+increment, increment)
        qq = self.df[[self.obsname, self.modname]].quantile(P)
        ax = qq.plot(
            x=self.obsname,
            y=self.modname,
            kind='scatter',
            xlim=[self.vmin, self.vmax],
            ylim=[self.vmin, self.vmax],
            grid=grid,
            **kwargs)
        plotmax = max(qq.max())*1.05
        plotmin = min(0,min(qq.min()))
        self.label(ax)
        return ax

    def add_regression(self, ax, print_pos=[None,None], color='r', lw=1,
                        show_equation=False, xpos=0.05, ypos=0.8,
                        **kwargs):

        """ Add regression line to plot"""
        coeffs, modcorr = linear_regression(self.df[self.obsname], self.df[self.modname], 1)
        x = np.arange(min(2*self.vmin, 0), 2*self.vmax)
        ax.plot(x, coeffs[0]*x+coeffs[1],color=color, lw=lw)
        ax.set_xlim([self.vmin, self.vmax])
        ax.set_ylim([self.vmin, self.vmax])

        if print_pos[0] is not None:
            ax.text(print_pos[0], print_pos[1], "reg: a=%.2f, b=%.2f" % (coeffs[0],coeffs[1]))

        # another way of printing the coefs
        if show_equation:
            r2 = np.corrcoef(self.df[self.obsname],self.df[self.modname])[0,1]**2
            n = self.df.shape[0]
            eqnstr = 'y= %.3f x + %.3f\nR2 = %.3f\nN = %i' % (coeffs[0], coeffs[1], r2, n)
            ax.text(xpos,ypos,eqnstr, transform=ax.transAxes, va='center', fontsize='small')


    def plot_contour(self, mincnt=1, binsize=0.5, cmap=plt.get_cmap('gray_r'),
                     cbar=True, **kwargs):
        logging.debug("Plotting contour")
        x,y,density=calc_scat_density(self.df, self.obsname, self.modname,
                                      binsize=binsize, mincnt=0)
        pden = density/density.max()
        mincnt /= density.max()
        #xx = yy = x[0:-1]+(x[0]+x[1])/2
        xx = yy = x[0:-1]
        ax = kwargs.pop('ax', None)
        if not ax:
            plt.figure(**kwargs)
            ax = plt.subplot(111)
        levels = np.arange(mincnt,1.1,0.1)
        levels[0] = 0.01
        ax.contour(xx,yy,pden,colors='gray', levels=levels, alpha=0.4)
        map = ax.contourf(xx,yy,pden, cmap=cmap, levels=levels, **kwargs)
        self.label(ax)
        ax.grid(True)
        ax.set_xlim(self.vmin, self.vmax)
        ax.set_ylim(self.vmin, self.vmax)
        if cbar:
            plt.colorbar(mappable=map)
        return ax

    def plot_scatter_density(self, grid=True, **kwargs):
        logging.debug("Plotting scatter with density")
        xs,ys,zs = calc_gaussian_kde(self.df, self.obsname, self.modname)
        ax = kwargs.pop('ax') if kwargs.has_key('ax') else None
        if not ax:
            plt.figure(**kwargs)
            ax = plt.subplot(111)
        p = ax.scatter(xs, ys, c=zs, s=20, edgecolor='')
        self.label(ax)
        ax.grid(True)
        ax.set_xlim(self.vmin, self.vmax)
        ax.set_ylim(self.vmin, self.vmax)
        return ax


    def stats_table(self, **kwargs):
        tb = stats_table(self.df, self.obsname, self.modname, **kwargs)
        return pd.DataFrame({self.modlabel : tb})

    def stats_table_monthly(self, **kwargs):
        tb = stats_table_monthly(df=self.df,
                                 obslabel=self.obsname,
                                 modlabel=self.modname,
                                 **kwargs)
        return tb

    def stats_table_groups(self, groups, **kwargs):
        tb = stats_table_groups(df=self.df, groups=groups,
                                 obslabel=self.obsname,
                                 modlabel=self.modname,
                                 **kwargs)
        return tb


class VerifyMulti(Verify):

    def __init__(self, df,
                 obsname='obs',
                 modnames=['m1','m2'],
                 obslabel=None,
                 modlabels=None,
                 var = 'hs',
                 vmin=None,
                 vmax=None,
                 sitelat=None,
                 sitelon=None,
                 sitename='None',
                 **kwargs
                 ):
        self.colors = ['b', 'r', 'g', 'c', 'y']
        self.modnames = modnames
        self.modlabels = modlabels or modnames

        if self.modlabels:
            if len(self.modlabels) != len(self.modnames):
                raise Exception("Length of modnames and modlabels differ")

        for modname in self.modnames:
            vminall = vmin if vmin is not None else min(df[modname].min(), df[obsname].min())
            vmaxall = vmax if vmax is not None else  max(df[modname].max(), df[obsname].max())

        super(VerifyMulti, self).__init__(df,
                                            obsname=obsname,
                                            obslabel=obslabel,
                                            var=var,
                                            vmin=vminall,
                                            vmax=vmaxall,
                                            sitelat=sitelat,
                                            sitelon=sitelon,
                                            sitename=sitename)


    def _rename_df_columns (self):
        ''' rename columns because plot dataframes doesn't recognize labels'''
        newcols = {self.obsname : self.obslabel}
        for i in range(len(self.modnames)):
            newcols[self.modnames[i]] = self.modlabels[i]

        self.df = self.df.rename(columns = newcols)
        self.obsname = self.obslabel
        self.modnames = self.modlabels

    def stats_table(self, **kwargs):
        frames = []
        for self.modname, self.modlabel in zip(self.modnames, self.modlabels):
            tbm = stats_table(self.df, self.obsname, self.modname, **kwargs)
            frames += [pd.DataFrame({self.modlabel : tbm})]
        tb = pd.concat(frames,axis=1)
        return tb

    def plot_qq(self, ax=None, **kwargs):
        if not ax:
            plt.figure()
            ax = plt.axes()
        ii = 0
        for self.modname, self.modlabel in zip(self.modnames, self.modlabels):
            ax = super(VerifyMulti, self).plot_qq(ax=ax, color=self.colors[ii],
                                                  label=self.modlabel, **kwargs)
            ii += 1
        legend(ax=ax,loc=4)
        ax.set_ylabel('%s [%s]' % (self.long_name, self.units))
        return ax


    def plot_timeseries(self, ax=None, **kwargs):
        if not ax:
            plt.figure()
            ax = plt.axes()
        ii = 0
        plotobs=True
        for self.modname, self.modlabel in zip(self.modnames, self.modlabels):
            ax = super(VerifyMulti, self).plot_timeseries(ax=ax, color=self.colors[ii],
                                                           plotobs=plotobs, **kwargs)
            ii += 1
            plotobs=False
        legend(ax)
        ax.set_ylabel('%s [%s]' % (self.long_name, self.units))
        return ax


    def plot_density(self, ax=None, **kwargs):
        if not ax:
            plt.figure()
            ax = plt.axes()
        ii = 0
        plotobs = True
        for self.modname, self.modlabel in zip(self.modnames, self.modlabels):
            ax = super(VerifyMulti, self).plot_density(ax=ax, color=self.colors[ii], plotobs=plotobs, **kwargs)
            plotobs = False
            ii += 1
        legend(ax=ax, loc=1)

    def plot_set(self,):
        import cartopy.crs as ccrs
        logging.info("   Plotting set ... ")
        nmods = len(self.modnames)
        nr = max(nmods+1,3)
        nc = 3

        # Set up axes
        self.fig = plt.figure(figsize=(14, 6 * nr))
        mapax = plt.subplot2grid((nr, nc), (0, 0),
                                 # projection=ccrs.PlateCarree())
                                 projection=ccrs.Orthographic(central_longitude=self.sitelon,
                                                              central_latitude=self.sitelat))
        self.plot_map(ax=mapax)
        self.tsax = plt.subplot2grid((nr, nc), (0, 1), colspan=2)
        self.denax = plt.subplot2grid((nr, nc), (2, 1), colspan=2)
        self.qqax = plt.subplot2grid((nr, nc), (1, 1), aspect=1)
        self.plot_timeseries(ax=self.tsax, alpha=0.6, lw=2)
        self.plot_density(ax=self.denax, alpha=0.6)
        self.plot_qq(ax=self.qqax, alpha=0.6)

        #df2taylor(self.df, fig=plt.gcf(), obslabel=self.obslabel, rect=100*nr+30+((nr-1)*3), colors=self.colors)
        df2taylor(self.df, fig=plt.gcf(), obslabel=self.obslabel, rect='%i%i%i' % (nr,nc,6), colors=self.colors)

        ii = 1
        for self.modname, self.modlabel in zip(self.modnames, self.modlabels):
            self.scatax = plt.subplot2grid((nr, nc), (ii, 0), aspect=1)
            self.plot_contour(ax=self.scatax)
            #self.plot_scatter(ax=self.scatax, alpha=0.6)
            self.add_stats(ax=self.scatax)
            ii += 1

        plt.tight_layout()
        return


    def plot_set_scatter_density(self,):
        import cartopy.crs as ccrs
        logging.info("   Plotting set scatter density... ")
        nmods = len(self.modnames)
        nr = 2 + int(np.ceil(nmods/3.))
        nc = 3

        # Set up axes
        self.fig = plt.figure(figsize=(18, 6 * nr))
        mapax = plt.subplot2grid((nr, nc), (0, 0),
                                 # projection=ccrs.PlateCarree())
                                 projection=ccrs.Orthographic(central_longitude=self.sitelon,
                                                              central_latitude=self.sitelat))
        self.plot_map(ax=mapax)
        self.tsax  = plt.subplot2grid((nr, nc), (0, 1), colspan=2)
        self.denax = plt.subplot2grid((nr, nc), (1, 0))
        self.qqax  = plt.subplot2grid((nr, nc), (1, 1), aspect=1)
        self.plot_timeseries(ax=self.tsax, alpha=0.6, lw=2)
        self.plot_density(ax=self.denax, alpha=0.6)
        self.plot_qq(ax=self.qqax, alpha=0.6)

        #df2taylor(self.df, fig=plt.gcf(), obslabel=self.obslabel, rect=100*nr+30+((nr-1)*3), colors=self.colors)
        df2taylor(self.df, fig=plt.gcf(), obslabel=self.obslabel, rect='%i%i%i' % (nr,nc,6), colors=self.colors)

        ii = 7
        for self.modname, self.modlabel in zip(self.modnames, self.modlabels):
            ax = plt.subplot('%i%i%i' % (nr,nc,ii))
            ax.set_aspect('equal')
            self.plot_scatter_density(ax=ax, alpha=0.6)
            self.add_regression(ax=ax, show_equation=False, color='k', lw=2)
            self.add_stats(ax=ax, loc='tr')
            ii += 1

        # plt.tight_layout()
        return



    # def plot_composite(self,figsize=(16, 9)
    #                    grid = 2, 3
    #                    plots = {plot_map,
    #                    ):
    #     logging.info("   Plotting... ")

    #     # Set up axes
    #     fig = plt.figure(figsize=(16, 9))
    #     mapax = plt.subplot2grid((2, 3), (0, 0),
    #                              projection=ccrs.Orthographic(central_longitude=self.sitelon,
    #                                                           central_latitude=self.sitelat))
    #     tsax = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    #     scatax = plt.subplot2grid((2, 3), (1, 0), aspect=1)
    #     denax = plt.subplot2grid((2, 3), (1, 1),)
    #     qqax = plt.subplot2grid((2, 3), (1, 2), aspect=1)

    #     # Add plots
    #     self.plot_map(buff=30, markersize=6,ax=mapax)
    #     self.plot_timeseries(ax=tsax)
    #     self.plot_hexbin(ax=scatax)
    #     self.plot_density(ax=denax)
    #     self.plot_qq(ax=qqax)

    #     plt.tight_layout()
    #     return fig


    # def compare_models(self,**kwargs):
    #     ntot=len(self.modnames)
    #     ii=0
    #     self.fig=plt.figure(figsize=(15,14))
    #     mapax = plt.subplot(ntot+1,3,1, projection=ccrs.Orthographic(central_longitude=self.sitelon,
    #                                                          central_latitude=self.sitelat))
    #     self.plot_map(buff=30,markersize=6, ax=mapax)
    #     self.tsax = plt.subplot2grid((ntot+1,3), (0,1), colspan=2)
    #     self.plot_timeseries(ax=self.tsax, **kwargs)
    #     plotobs=True # Only plot obs if first model

    #     while ii < ntot:
    #         nmod = ii + 1
    #         scatax = plt.subplot2grid((ntot+1,3), (nmod, 0),aspect=1)
    #         denax = plt.subplot2grid((ntot+1,3), (nmod, 1),)
    #         qqax = plt.subplot2grid((ntot+1,3), (nmod, 2),aspect=1)

    #         # Add plots
    #         self.plot_hexbin(modno=ii, ax=scatax,**kwargs)
    #         self.plot_density(modno=ii, ax=denax,**kwargs)
    #         self.plot_qq(modno=ii, ax=qqax,**kwargs)
    #         plt.grid(True)
    #         plt.tight_layout()
    #         ii+=1

    #     # for mod,label in zip(self.modnames,labellist):
    #     #     logging.info('PROCESSING %s'%mod)
    #     #     self.qdict.update({'dset':mod})
    #     #     self.add_plot_set(ii,ntot,obslab=obslab,modlab=label,**kwargs)
    #     #     ii+=1

def plot_map(lat, lon, buff=20, marker='ro', markercolor='r', markersize=12,
             ax=None, subplot=111, label=None):
    """
    Plots map of obs location
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    if not ax:
        ax = plt.subplot(subplot,
                         projection=ccrs.Orthographic(central_latitude=lat,
                                                      central_longitude=lon))
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_extent([lon-buff, lon+buff, lat-buff,
                   lat+buff])
    ax.plot(lon, lat, marker, label=label, color=markercolor,
            markersize=markersize, transform=ccrs.PlateCarree())
    if label:
        plt.title(label)
    gl = ax.gridlines(draw_labels=False)
    return ax

def calc_scat_density(df, obsname, modname, binsize=0.5, mincnt=1):
    """
    Calculates density for a scatter plot
    """
    import numpy.ma as ma
    xx = yy = np.arange(min(df[obsname].min(), df[modname].min())-binsize, max(df[obsname].max(), df[modname].max())+(2*binsize), binsize)
    density, xedges, yedges = np.histogram2d(df[modname], df[obsname],
                                             bins=(xx,yy))
    return xx, yy, ma.masked_less(density, mincnt)

# a bit slow, and seems to do exactly the same as the above function but for each point instead of bins
def calc_gaussian_kde(df, obsname, modname):
    from scipy.stats import gaussian_kde
    x = df[obsname]
    y = df[modname]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    xs, ys, zs = x[idx], y[idx], z[idx]
    return xs,ys,zs

def legend(ax=None, fontsize='medium',**kwargs):
    fontP = FontProperties()
    fontP.set_size(fontsize)
    if ax:
        lg=ax.legend(borderaxespad=0,prop=fontP,**kwargs)
    else:
        lg=plt.legend(borderaxespad=0,prop=fontP,**kwargs)
    lg.get_frame().set_linewidth(0)
    lg.get_frame().set_alpha(0.2)
    return lg

def plot_timeseries_monthly(df, figsize=(10,20),fill_between=False,
                            format_xaxis=True, **kwargs):

    grouped = df.groupby(pd.TimeGrouper(freq='M'))
    fig,axs = plt.subplots(len(grouped),1, sharey=True, figsize=figsize)
    i=0
    for gname, dfm in grouped:
        ax = axs[i]
        dfm.plot(ax=ax, rot=0, legend=False, **kwargs)
        t1 = datetime.datetime(gname.year, gname.month,1)
        t2 = t1 + relativedelta(months=+1)
        ax.set_xlim([t1,t2])
        if fill_between:
            ax.fill_between(dfm.index, 0, dfm, where=dfm>=0, facecolor='red', alpha=0.5)
            ax.fill_between(dfm.index, 0, dfm, where=dfm<0, facecolor='blue', alpha=0.5)
        if format_xaxis:
            ax.set_xlabel('')
            ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
            ax.xaxis.set_major_locator(dates.DayLocator(interval=7))
            ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))
        i+=1
        ax.set_title(gname.strftime('%b'))

    axs[0].legend(loc=1, ncol=df.shape[1], framealpha=0, fontsize='small')
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.5)
    fig.autofmt_xdate(rotation=0, ha='center')#bottom=0.2, rotation=30, ha='right')

    return fig,axs
