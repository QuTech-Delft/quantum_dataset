# -*- coding: utf-8 -*-
""" Module to load and use the Delft Quantum DataSet

@author:    Pieter Eendebak <pieter.eendebak@tno.nl>

"""

#%% Import the packages needed

import os
import distutils.version

import numpy as np
import matplotlib.pyplot as plt

from MarkupPy import markup
from MarkupPy.markup import oneliner as oneliner
import imageio

import qtt
import qtt.utilities.json_serializer

   
class QuantumDataset():
    
    def __init__(self, datadir, tags=None):
        """ Create object to load and store datasets
        
        Args:
            datadir (str): directory with stored results
            tags (None or list): list with possible tags
        
        """
        self._minimal_version = '0.1.2'
        self._test_datadir = datadir
        self._datafile_extensions = ['.json']
        if tags is None:
            tags=os.listdir(datadir)
            tags = [tag for tag in tags if '.' not in tag]
        self.tags=tags
        for subdir in tags:
            sdir=os.path.join(self._test_datadir, subdir)
            qtt.utilities.tools.mkdirc(sdir)
               
    def _check_data(self):
        """ Check whether the required data is present """
        try:
            with open(os.path.join(self._test_datadir, 'quantumdataset.txt'), 'rt') as fid:
                version = fid.readline().strip()                
        except Exception as ex:
            raise Exception('could not find data for QuantumDataset at location %s' % self._test_datadir) from ex
        if not distutils.version.StrictVersion(self._version) <= distutils.version.StrictVersion(version):
            raise Exception('version of data %s is older than version required %s' % (version, self._minimal_version) )
        
    def generate_save_function(self, tag):
        """ Generate a function to save data for a particular tag """
        def save_function(dataset, overwrite = False):
            self.save_dataset(dataset, tag, subtag = None, overwrite = overwrite)
        save_function.__doc__ = 'save dataset under tag %s' % tag
        save_function.__name__ = 'save_%s' % tag
        save_function.__qualname__=save_function.__name__
        
        setattr(self, 'save_%s' % tag, save_function)
        
   
    def show_data(self):
        """ List all data in dataset """
        for subdir in self.tags:
            sdir=os.path.join(self._test_datadir, subdir)
            qtt.utilities.tools.mkdirc(sdir)
            ll=self.list_subtags(subdir) # os.listdir(sdir)
            print('tag %s: %d results'  % (subdir, len(ll)))

    def list_subtags(self, tag):
        sdir=os.path.join(self._test_datadir, tag)
        ll=qtt.gui.dataviewer.DataViewer.find_datafiles(datadir=sdir, extensions=self._datafile_extensions)
        subtags=[os.path.relpath(path, start=sdir) for path in ll]
        return subtags
    
    def results_old(self, tag):
        sdir=os.path.join(self._test_datadir, tag)
        ll=qtt.gui.dataviewer.DataViewer.find_datafiles(datadir=sdir, extensions=self._datafile_extensions)

        return ll
    
    def plot_dataset(self, dataset, fig=100):
        """ Plot a dataset into a matplotlib figure window """
        qtt.data.plot_dataset(dataset, fig=fig)

    def _figure2image(self, fig):
        """ Convert matplotlib figure window to an RGB image """
        Fig=plt.figure(fig)
        plt.draw(); plt.pause(1e-3)
        data = np.fromstring(Fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(Fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def show(self, tag, fig=100):

        import matplotlib.pyplot as plt
        
        import qtt.gui.dataviewer
        import numpy as np
        sdir=os.path.join(self._test_datadir, tag)
        datafiles=qtt.gui.dataviewer.DataViewer.find_datafiles(datadir=sdir, )
        print('tag %s: %d result(s)'  % (tag, len(datafiles)))
        
        nx=ny=int(np.sqrt(len(datafiles))+1)
        plt.close(fig)
        Fig=plt.figure(fig); plt.clf()
        tmpfig=fig+1
        plt.close(tmpfig)
        Fig=plt.figure(tmpfig); plt.clf()
        for jj, l in enumerate(datafiles):
            print('tag %s: plot %d' % (tag, jj))
            try:
                ds=qtt.data.load_dataset(l)
                idx=(jj)%(nx*ny)+1
                self.plot_dataset(ds, fig=tmpfig)
                   
                data = np.fromstring(Fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data = data.reshape(Fig.canvas.get_width_height()[::-1] + (3,))
    
                plt.figure(fig)
                plt.subplot(nx,ny, idx)
                plt.imshow(data); plt.axis('image' ); plt.axis('off'); plt.draw()
            except Exception as ex:
                print('failed to plot %s'  % l)
        
        
    def generate_results_page(self, tag, htmldir, filename, plot_function = None, verbose=1):
        """ Generate a result page for a particular tag """
        
        if verbose:
            print('generate_results_page: tag %s' % tag)
            
        if plot_function is None:
            plot_function = self.plot_dataset
            
        page=markup.page()
        page.init(title="Quantum Dataset: tag %s" % tag,
                      #css=('../oastyle.css'),
                      lang='en', # htmlattrs=dict({'xmlns': 'http://www.w3.org/1999/xhtml', 'xml:lang': 'en'}),
                      header="<!-- Start of page -->",
                      bodyattrs=dict({'style': 'padding-left: 3px;'}),
                      doctype='<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
                      metainfo=({'text/html': 'charset=utf-8', 'keywords': 'quantum dataset',
                                 'robots': 'index, follow', 'description': 'quantum dataset'}),
                      footer="<!-- End of page -->")
    
        page.h1('Quantum Dataset: tag %s' % tag)
        page.h1.close()
        page.p('For more information see https://github.com/QuTech-Delft/qtt')
        page.p.close()
    
        subtags = self.list_subtags(tag)
        
        page.ul()
        
        for ii, dataset_name in enumerate(subtags):
            link=oneliner.a('%s' % dataset_name, href='#dataset%d' % ii) #foo
            page.li(link)        
        page.ul.close()
    
        qtt.utilities.tools.mkdirc(os.path.join(    htmldir, 'images') )
                                            
        for ii, subtag in enumerate(subtags):
            dataset_name=self.generate_filename(tag, subtag)
            imagefile0=os.path.join('images', tag,  'dataset%d.png' % ii)
            imagefile=os.path.join(qtt.utilities.tools.mkdirc(os.path.join(htmldir, 'images', tag )), 'dataset%d.png' % ii)
            if verbose:
                print('generate_results_page %s: %d/%d: dataset_name %s' % (tag, ii, len(subtags), os.path.basename(dataset_name)) )
            dataset=self.load_dataset(filename=dataset_name, output_format='qcodes.DataSet')
            if verbose>=3:
                print(dataset)
            
            plot_function(dataset, fig=123)
            image=self._figure2image(fig=123)[:,:,::-1]
            imageio.imwrite(imagefile, image)
            page.a(name="dataset%d" % ii)
            page.h3('Dataset: %s' % dataset_name)
            page.a.close()
            page.img(src=imagefile0)
            
        if filename is not None:
            with(open(filename, 'wt')) as fid:
                fid.write(str(page))
            
        return page
    
    def generate_overview_page(self, htmldir, plot_functions):
        for tag in self.list_tags():    
            filename=os.path.join(htmldir, 'qdataset-%s.html' % tag)
            
            plot_function = plot_functions.get(tag, None)
            page=self.generate_results_page( tag, htmldir, filename, plot_function = plot_function)    
    
        page= self._generate_main_page( htmldir)
        
    def _generate_main_page(self, htmldir):
        """ Generate overview page with results """
        page=markup.page()
        page.init(title="Quantum Dataset",
                      lang='en',
                      header="<!-- Start of page -->",
                      bodyattrs=dict({'style': 'padding-left: 3px;'}),
                      doctype='<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
                      metainfo=({'text/html': 'charset=utf-8', 'keywords': 'quantum dataset',
                                 'robots': 'index, follow', 'description': 'quantum dataset'}),
                      footer="<!-- End of page -->")
    
        page.h1('Quantum Dataset')
    
        tags=sorted(self.list_tags()  )
        page.ol()
        
        for ii, tag in enumerate(tags):
            link='qdataset-%s.html' % tag
            link=oneliner.a('%s' % tag, href=link) 
            subtags=self.list_subtags(tag)
            page.li(link + ': %d datasets' % len(subtags))        
        page.ol.close()
    
            
        if htmldir is not None:
            filename=os.path.join(htmldir, 'index.html')
            with(open(filename, 'wt')) as fid:
                fid.write(str(page))
            
        return page
            
    def create_tag(self, tag):
        qtt.utilities.tools.mkdirc(os.path.join(self._test_datadir, tag))
        
    def generate_filename(self, tag, subtag):
        filename= os.path.join(self._test_datadir, tag, subtag)
        if not filename.endswith('.json'):
            filename+= '.json'
        return filename

    def load_dataset(self, tag=None, subtag=None, filename=None, output_format = None):
         if isinstance(subtag, int):
             subtag = self.list_subtags(tag)[subtag]
         if filename is None:
             filename = self.generate_filename(tag, subtag)
         dataset_dictionary = qtt.utilities.json_serializer.load_json(filename)
         if output_format=='qcodes.DataSet' or output_format is None:
             dataset = qtt.data.dictionary_to_dataset(dataset_dictionary)
         elif output_format=='dict':
             dataset = dataset_dictionary
         else:
             raise Exception('output_format %s not valid' % (output_format,))
         return dataset 
     
    def save_dataset(self, dataset, tag, subtag = None, overwrite= False):
        """ Save dataset to disk """
        if qcodes is not None:
            if isinstance(dataset, qcodes.data.data_set.DataSet):
                dataset = qtt.data.dataset_to_dictionary(dataset)
        if not isinstance(dataset, dict):
            raise Exception('cannot store dataset of type %s' % type(dataset))
            
        # store
        if subtag is None:
            subtag = dataset['extra']['location'].replace('/', '_')
            subtag = subtag.replace('\\', '_')
            subtag = '_'.join(subtag.split(':'))
        filename = self.generate_filename(tag, subtag)
        qtt.utilities.tools.mkdirc(os.path.split(filename)[0])

        if not overwrite:        
            if os.path.exists(filename):
                raise Exception('filename %s already exists' % filename)
        qtt.utilities.json_serializer.save_json(dataset, filename)
        return filename
    
    def list_tags(self):
        return self.tags
    
       
