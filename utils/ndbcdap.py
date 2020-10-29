#!/usr/bin/env python

# Small module to interface with NDBC's opendap server


import general as gen
import netCDF4
import matplotlib.pyplot as plt
from netCDF4 import num2date, date2num
from matplotlib.dates import DateFormatter
import datetime
import numpy as np
import numpy.ma as ma
import glob
import config
from StringIO import StringIO
import unittest
import pandas as pd
import os

# TODO - add lat/lons to bom data (will have to create lookup table)

# Remote NDBC server
server='http://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/'

# Set for local data
#import config
#server=config.buoydir

# Define locations for buoys that do not include information in meta-data
buoylocs={
        'cdec':(-36.07,136.62),
        'cpso':(-42.08,145.01),
        }


class ndbc:
    """ Class for reading in ndbc netcdf data from
    plotting, and super-obbing it"""

    def __init__(self, stationlist,yearlist, **kwargs):
        """Create initial"""
        self.urll=[]
        self.stationlist=stationlist
        for station in stationlist:
            for year in yearlist:
                self.urll.append(server+station+'/'+station+'h'+year+'.nc')
        exclude=['gust',
                'air_pressure', 'air_temperature', 'sea_surface_temperature',
                'dewpt_temperature', 'visibility', 'water_level']
        if len(self.urll) == 1:
            print "Loading data from ", self.urll[0]
            self.ncfile = netCDF4.Dataset(self.urll[0],exclude=exclude)
        else:
            print "    Reading in %s files..." % len(self.urll)
            for fname in self.urll:
                print "        ",fname
            self.ncfile = netCDF4.MFDataset(self.urll,aggdim='time',exclude=exclude)

    def extract_var(self, var='wave_height',qc=True,correctwndsp=True, **kwargs):
        """Extract data and apply qc flags"""
        data={}
        ncvar=self.ncfile.variables[var]
        tmp=ma.masked_array(ncvar[:].squeeze())
        mask=tmp.mask
        #flag=self.flag_bad(tmp,thres=30)
        if var=='wind_spd':
            if correctwndsp:
                if len(self.stationlist)>1:
                    raise ValueError("Can't correct for multiple stations..")
                else:
                    anom_height=get_anom_height(self.stationlist[0])
                    tmp=adjust_windspeed(tmp,anom_height,10)
        data['obs']=ma.masked_array(tmp.compressed())
        times=self.ncfile.variables['time']
        tmp = ma.masked_array(num2date(times[:],times.units))
        tmp.mask=mask
        data['time']=ma.masked_array(tmp.compressed())
        for v,v2 in zip(('latitude','longitude'),('lat','lon')):
            ncvar=self.ncfile.variables[v]
            tmp=ma.masked_array(ncvar[:])
            tmp.mask=mask
            #tmp.mask=flag
            data[v2]=ma.masked_array(np.ones(data['obs'].size)*self.ncfile.variables[v][0])
        if data['lon'].size>0:
            if data['lon'][0] < 0:
                print "Correcting longitude"
                data['lon']+=360.
        return data

    def extract_latlon(self,):
        """Extract lat and lon"""
        lat=self.ncfile.variables['latitude'][0]
        lon=self.ncfile.variables['longitude'][0]
        if lon < 0:
            print "Correcting longitude"
            lon+=360.
        return lat,lon


    def close_nc(self,):
        self.ncfile.close()

    def flag_bad(self,var):
        "Returns mask of bad data"
        try:
            mask = self.ncfile.variables[var][:].mask
        except Exception, e:

            pass
        ind = np.where(self.ncfile.variables[var][:]>40)
        mask[ind]=True
        return mask

class sopac(ndbc):
    """docstring for SOPAC"""

    def __init__(self, stationlist, **kwargs):
        """Create initial"""
        self.urll=[]
        for station in stationlist:
            self.urll.append(glob.glob(config.sopacdir+'/SOPAC_'+station+'_*.nc')[0])
        if len(self.urll) == 1:
            print "Loading data from ", self.urll[0]
            self.ncfile = netCDF4.Dataset(self.urll[0])
        else:
            print "    Reading in %s files..." % len(self.urll)
            self.ncfile = netCDF4.MFDataset(self.urll,aggdim='time',exclude=exclude)
        self.statname=getattr(self.ncfile,'location')

    def extract_var(self, var='wave_height',qc=True,**kwargs):
        """Extract data and apply qc flags"""
        data={}
        ncvar=self.ncfile.variables[var]
        tmp=ma.masked_array(ncvar[:])
        mask=tmp.mask
        #flag=self.flag_bad(tmp,thres=30)
        data['obs']=ma.fix_invalid(tmp.compressed())
        times=self.ncfile.variables['time']
        if times.units=='days since 01-Jan-1950':
            units='days since 1950-01-01 00:00:00 UTC'
        tmp = ma.masked_array(num2date(times[:],units))
        tmp.mask=mask
        data['time']=ma.masked_array(tmp.compressed())
        return data

    def extract_latlon(self,):
        """Extract lat and lon"""
        lat=self.ncfile.latitude
        lon=self.ncfile.longitude
        if lon < 0:
            print "Correcting longitude"
            lon+=360.
        return lat,lon

class arena(ndbc):
    """docstring for SOPAC"""

    def __init__(self, stationlist, **kwargs):
        """Create initial"""
        self.urll=[]
        for station in stationlist:
            self.urll.append(glob.glob(config.arenadir+'/'+station+'.nc')[0])
        if len(self.urll) == 1:
            print "Loading data from ", self.urll[0]
            self.ncfile = netCDF4.Dataset(self.urll[0])
        else:
            print "    Reading in %s files..." % len(self.urll)
            self.ncfile = netCDF4.MFDataset(self.urll,aggdim='time',exclude=exclude)
        self.statname=getattr(self.ncfile,'StationName')

    def extract_var(self, var='Hs',qc=True,**kwargs):
        """Extract data and apply qc flags"""
        data={}
        ncvar=self.ncfile.variables[var]
        tmp=ma.masked_array(ncvar[:])
        mask=tmp.mask
        #flag=self.flag_bad(tmp,thres=30)
        data['obs']=ma.fix_invalid(tmp.compressed())
        times=self.ncfile.variables['time']
        tmp = ma.masked_array(num2date(times[:],times.units))
        tmp.mask=mask
        data['time']=ma.masked_array(tmp.compressed())
        return data

    def extract_latlon(self,):
        """Extract lat and lon"""
        lat=self.ncfile.lat
        lon=self.ncfile.lon
        if lon < 0:
            print "Correcting longitude"
            lon+=360.
        return lat,lon

class mhl(ndbc):
    """docstring for mhl"""

    def __init__(self, stationlist, startdate=None, enddate=None, **kwargs):
        """Create initial"""
        self.urll=[]
        for station in stationlist:
            print config.mhldir+'/'+station+'_*.nc'
            self.urll.append(glob.glob(config.mhldir+'/'+station+'_*.nc')[0])
        if len(self.urll) == 1:
            print "Loading data from ", self.urll[0]
            self.ncfile = netCDF4.Dataset(self.urll[0])
        else:
            print "    Reading in %s files..." % len(self.urll)
            self.ncfile = netCDF4.MFDataset(self.urll,aggdim='time',exclude=exclude)

    def extract_var(self, var='VAVH',qc=True,**kwargs):
        """Extract data and apply qc flags"""
        data={}
        ncvar=self.ncfile.variables[var]
        tmp=ma.masked_array(ncvar[:].squeeze())
        mask=tmp.mask
        #flag=self.flag_bad(tmp,thres=30)
        data['obs']=ma.masked_array(tmp.compressed())
        times=self.ncfile.variables['TIME']
        tmp = ma.masked_array(num2date(times[:],times.units))
        tmp.mask=mask
        data['time']=ma.masked_array(tmp.compressed())
        for v,v2 in zip(('LATITUDE','LONGITUDE'),('lat','lon')):
            ncvar=self.ncfile.variables[v]
            tmp=ma.masked_array(ncvar[:])
            tmp.mask=mask
            #tmp.mask=flag
            data[v2]=ma.masked_array(np.ones(data['obs'].size)*self.ncfile.variables[v][0])
        if data['lon'].size>0:
            if data['lon'][0] < 0:
                print "Correcting longitude"
                data['lon']+=360.
        return data

    def extract_latlon(self,):
        """Extract lat and lon"""
        lat=self.ncfile.variables['LATITUDE'][0]
        lon=self.ncfile.variables['LONGITUDE'][0]
        if lon < 0:
            print "Correcting longitude"
            lon+=360.
        return lat,lon

class derm(ndbc):
    """Extracting derm data"""

    def __init__(self, stationlist, startdate=None, enddate=None, **kwargs):
        """Create initial"""
        self.urll=[]
        for station in stationlist:
            self.urll.append(glob.glob(config.dermdir+'/'+station+'_*.csv')[0])
        if len(self.urll) == 1:
            print "Loading data from ", self.urll[0]
            self.df = pd.DataFrame.from_csv(self.urll[0],header=1)
        else:
            print "    Reading in %s files..." % len(self.urll)
            self.ncfile = netCDF4.MFDataset(self.urll,aggdim='time',exclude=exclude)
        with open(self.urll[0]) as tmp:
            header=tmp.readline().split(';')
            self.statname=header[0]
            self.lat=float(header[1].split()[1])
            self.lon=float(header[2].split()[1])

    def extract_var(self, var='Hs',qc=True,**kwargs):
        """Extract data and apply qc flags"""
        data={}
        #tmptime=self.df.index.tz_localize('Australia/Brisbane').tz_convert('UTC')
        #data['time']= np.array(tmptime.to_pydatetime(), dtype=datetime.datetime)+datetime.timedelta(hours=10)
        data['time']= np.array(self.df.index.to_pydatetime(), dtype=datetime.datetime)-datetime.timedelta(hours=10)
        data['obs']=ma.fix_invalid(self.df[var].values)
        ones=np.ones(data['obs'].size)
        data['lat']=ones*self.lat
        data['lon']=ones*self.lon
        if data['lon'].size>0:
            if data['lon'][0] < 0:
                print "Correcting longitude"
                data['lon']+=360.
        return data

class bom(ndbc):
    """Extracting bom data"""

    def __init__(self, stationlist,yearlist, **kwargs):
        """Create initial"""
        self.urll=[]
        if len(stationlist) > 1 or len(yearlist) > 1:
            print "Multiple stations and years not yet implemented..."
            return
        else:
            for station in stationlist:
                for year in yearlist:
                    fpart=config.bomdir+'/'+station+year
                    print fpart
                    if os.path.isfile(fpart+'.txt'):
                        self.urll.append(fpart+'.txt')
                        self.df = pd.read_table(self.urll[0],header=0,skiprows=[1],delim_whitespace=True,parse_dates=[[1, 2]],index_col=0)
                    elif os.path.isfile(fpart+'.xls'):
                        self.urll.append(fpart+'.xls')
                        self.df = pd.read_excel(self.urll[0],0,header=1,skiprows=[0],index_col=0)
                        cols=self.df.columns.values
                        new_cols=cols
                        for ii in np.arange(0,cols.size):
                            new_cols[ii]=cols[ii].split()[0]
                        self.df.columns=new_cols
                    else:
                        print "No matching data for %s %s" %(station,year)
            #if len(self.urll) == 1:
                #print "Loading data from ", self.urll[0]
                #self.df = pd.read_table(self.urll[0],header=0,skiprows=[1],delim_whitespace=True,parse_dates=[[1, 2]],index_col=0)
        self.lat=buoylocs[station][0]
        self.lon=buoylocs[station][1]


    def extract_var(self, var='Hs',qc=True,**kwargs):
        """Extract data and apply qc flags"""
        data={}
        #tmptime=self.df.index.tz_localize('Australia/Brisbane').tz_convert('UTC')
        #data['time']= np.array(tmptime.to_pydatetime(), dtype=datetime.datetime)+datetime.timedelta(hours=10)
        print "NEED TO FIX TIMEZONE"
        data['time']= np.array(self.df.index.to_pydatetime(), dtype=datetime.datetime)-datetime.timedelta(hours=10)
        data['obs']=ma.fix_invalid(self.df[var].values)
        ones=np.ones(data['obs'].size)
        data['lat']=ones*self.lat
        data['lon']=ones*self.lon
        if data['lon'].size>0:
            if data['lon'][0] < 0:
                print "Correcting longitude"
                data['lon']+=360.
        return data


def adjust_windspeed(windspeed, anomheight, refheight=10):
    """
    Adjusts the wind from the anomometer height to the given
    reference height (default 10m) using a logarithmic profile

    See:
    http://www.ndbc.noaa.gov/adjust_wind.shtml
    See also for different values of exponent (0.11 here)
    http://en.wikipedia.org/wiki/Wind_gradient

    windspeed -- measured wind speed
    anomheight -- anomometer height
    refheight -- reference height - desired height of wind speed

    """
    return windspeed * (float(refheight)/float(anomheight))**0.11


def get_anom_height(station):
    """ Look up anemometer height
    use data downloaded from http://www.ndbc.noaa.gov/bmanht.shtml
    """
    stationstring="                                                         \n \
                           National Data Buoy Center                        \n \
                                                                            \n \
                 Heights (in meters) of the sensors at NDBC buoys           \n \
                                                                            \n \
    STATION        SITE        AIR_TEMP     ANEMOMETER     BAROMETER        \n \
     IDENT      ELEVATION     ELEVATION      ELEVATION     ELEVATION        \n \
                                                                            \n \
    41001          0              4               5               0         \n \
    41002          0              4               5               0         \n \
    41004          0              4               5               0         \n \
    41008          0              4               5               0         \n \
    41009          0              4               5               0         \n \
    41010          0              4               5               0         \n \
    41012          0              4               5               0         \n \
    41013          0              4               5               0         \n \
    41025          0              4               5               0         \n \
    41036          0              4               5               0         \n \
    41040          0              4               5               0         \n \
    41041          0              4               5               0         \n \
    41043          0              4               5               0         \n \
    41044          0              4               5               0         \n \
    41046          0              4               5               0         \n \
    41047          0              4               5               0         \n \
    41048          0              4               5               0         \n \
    41049          0              4               5               0         \n \
    42001          0              4               5               0         \n \
    42002          0              10              10              0         \n \
    42003          0              4               5               0         \n \
    42012          0              4               5               0         \n \
    42019          0              4               5               0         \n \
    42020          0              4               5               0         \n \
    42035          0              4               5               0         \n \
    42036          0              4               5               0         \n \
    42039          0              4               5               0         \n \
    42040          0              10              10              0         \n \
    42055          0              4               5               0         \n \
    42056          0              4               5               0         \n \
    42057          0              4               5               0         \n \
    42058          0              4               5               0         \n \
    42059          0              4               5               0         \n \
    42060          0              4               5               0         \n \
    44005          0              4               5               0         \n \
    44007          0              4               5               0         \n \
    44008          0              4               5               0         \n \
    44009          0              4               5               0         \n \
    44011          0              4               5               0         \n \
    44013          0              4               5               0         \n \
    44014          0              4               5               0         \n \
    44017          0              4               5               0         \n \
    44018          0              4               5               0         \n \
    44020          0              4               5               0         \n \
    44025          0              4               5               0         \n \
    44027          0              4               5               0         \n \
    44065          0              4               5               0         \n \
    44066          0              4               5               0         \n \
    45001          183            4               5               183       \n \
    45002          176.4          4               5               176.4     \n \
    45003          177            3.2             3.2             177       \n \
    45004          183            4               5               183       \n \
    45005          173.9          4               5               173.9     \n \
    45006          183            4               5               183       \n \
    45007          176.4          4               5               176.4     \n \
    45008          177            4               5               177       \n \
    45012          74.7           4               5               74.7      \n \
    46001          0              4               5               0         \n \
    46002          0              4               5               0         \n \
    46005          0              4               5               0         \n \
    46006          0              4               5               0         \n \
    46011          0              4               5               0         \n \
    46012          0              4               5               0         \n \
    46013          0              4               5               0         \n \
    46014          0              4               5               0         \n \
    46015          0              4               5               0         \n \
    46022          0              4               5               0         \n \
    46025          0              4               5               0         \n \
    46026          0              4               5               0         \n \
    46027          0              4               5               0         \n \
    46028          0              4               5               0         \n \
    46029          0              4               5               0         \n \
    46035          0              10              10              0         \n \
    46041          0              4               5               0         \n \
    46042          0              4               5               0         \n \
    46047          0              4               5               0         \n \
    46050          0              4               5               0         \n \
    46053          0              4               5               0         \n \
    46054          0              4               5               0         \n \
    46059          0              4               5               0         \n \
    46060          0              4               5               0         \n \
    46061          0              4               5               0         \n \
    46066          0              4               5               0         \n \
    46069          0              4               5               0         \n \
    46070          0              4               5               0         \n \
    46071          0              4               5               0         \n \
    46072          0              4               5               0         \n \
    46073          0              10              10              0         \n \
    46075          0              4               5               0         \n \
    46076          0              4               5               0         \n \
    46077          0              4               5               0         \n \
    46078          0              4               5               0         \n \
    46080          0              4               5               0         \n \
    46081          0              4               5               0         \n \
    46082          0              4               5               0         \n \
    46083          0              4               5               0         \n \
    46084          0              4               5               0         \n \
    46085          0              4               5               0         \n \
    46086          0              4               5               0         \n \
    46087          0              4               5               0         \n \
    46088          0              4               5               0         \n \
    46089          0              4               5               0         \n \
    51000          0              4               5               0         \n \
    51001          0              4               5               0         \n \
    51002          0              4               5               0         \n \
    51003          0              4               5               0         \n \
    51004          0              4               5               0         \n \
    51100          0              4               5               0         \n \
    51101          0              4               5               0         \n \
    # Added by manually looking up                                          \n \
    42007          0              4               5               0         \n \
    44004          0              4               5               0         \n \
    "
    dt = np.dtype([('station','S6'),
        ('site_el', float),
        ('temp_el', float),
        ('anom_el', float),
        ('barom_el', float),
        ])
    statlist = StringIO(stationstring)
    statdat=np.loadtxt(statlist,skiprows=8,dtype=dt)
    index=[statdat['station'] == station][0]
    if not index.any():
        raise ValueError('Buoy station %s height unknown!' %station)
    return statdat['anom_el'][index][0]

def plot_ts(data, obslabel='Obs', fig=None, subplot='111', ):
    """
    plots obs vs time
    """
    if fig == None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot)
    obsdat, timedat = data['obs'], data['time']
    obsline = plt.plot(timedat, obsdat, label=obslabel)
    ax.xaxis.set_major_formatter(DateFormatter('%d %b %Y '))
    fig.autofmt_xdate()
    xticks = ax.get_xticklabels()
    ax.grid()
    return

def test_ndbc(var='wind_spd'):
    stationlist='51002',
    #stationlist='52201',
    yearlist='2013',
    tmp=ndbc(stationlist,yearlist)
    fig=plt.figure()
    data=tmp.extract_var(var='wind_spd',correctwndsp=False)
    plot_ts(data,fig=fig)
    data=tmp.extract_var(var='wind_spd',correctwndsp=True)
    plot_ts(data,fig=fig)
    return tmp

def test_sopac():
    #stationlist='44011',
    stationlist='33',
    tmp=sopac(stationlist)
    data=tmp.extract_var('Hs')
    plot_ts(data)
    return tmp

def test_mhl():
    #stationlist='44011',
    stationlist='55014',
    tmp=mhl(stationlist)
    data=tmp.extract_var('VAVH')
    plot_ts(data)
    return tmp

def test_derm():
    #stationlist='44011',
    stationlist='bri31',
    tmp=derm(stationlist)
    data=tmp.extract_var('Hs')
    plot_ts(data)
    return tmp

def test_bom():
    #stationlist='44011',
    stationlist='cdec',
    yearlist='2008',
    tmp=bom(stationlist,yearlist)
    data=tmp.extract_var('Hs')
    plot_ts(data)
    return tmp

def test_arena():
    #stationlist='44011',
    stationlist='55026',
    tmp=arena(stationlist)
    data=tmp.extract_var('Hs')
    plot_ts(data)
    return tmp

class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.buoy = '51000'

    def test_adjust_wnd(self):
        wndsp=np.arange(0,10)
        # Do nothing if reference height is the same as anomometer height
        self.assertTrue(np.allclose(adjust_windspeed(wndsp,10,10),wndsp))

    def test_anemom_height(self):
        self.assertEqual(get_anom_height('46073'), 10.0)
        self.assertEqual(get_anom_height('51001'), 5.0)
        self.assertRaises(ValueError,get_anom_height,'5101')


if __name__ == '__main__':
    #unittest.main()
    #tmp=test_bom()
    tmp=test_arena()
    #tmp=test_ndbc()
    plt.show()
