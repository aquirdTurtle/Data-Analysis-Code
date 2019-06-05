import h5py as h5
from colorama import Fore, Style
from numpy import array as arr
import numpy as np
import Miscellaneous as misc
dataAddress = None


def annotate(fileID=None):
    title = input("Run Title: ")
    hashNum = int(input("Title-Level: "))
    titleStr = ''.join('#' for _ in range(hashNum)) + ' ' + title
    rationale = input("Experiment Rationale:")
    result = input("Experiment Result:")
    with ExpFile() as f:
        f.open_hdf5(fileID, openFlag='a')
        
        if 'Experiment_Rationale' in f.f['Miscellaneous'].keys():
            del f.f['Miscellaneous']['Experiment_Rationale']
        dset = f.f['Miscellaneous'].create_dataset("Experiment_Rationale", shape=(1,), dtype="S"+str(len(rationale))) 
        dset[0] = np.string_(rationale)
        
        if 'Experiment_Result' in f.f['Miscellaneous'].keys():
            del f.f['Miscellaneous']['Experiment_Result']
        dset2 = f.f['Miscellaneous'].create_dataset("Experiment_Result", shape=(1,), dtype="S"+str(len(result))) 
        dset2[0] = np.string_(result)
        
        if 'Experiment_Title' in f.f['Miscellaneous'].keys():
            del f.f['Miscellaneous']['Experiment_Title']
        dset3 = f.f['Miscellaneous'].create_dataset("Experiment_Title", shape=(1,), dtype="S"+str(len(titleStr))) 
        dset3[0] = np.string_(titleStr)
        

def checkAnnotation(fileNum, force=True, quiet=False):
    try:
        with ExpFile(fileNum) as f:
            if (   'Experiment_Rationale' not in f.f['Miscellaneous']
                or 'Experiment_Result'    not in f.f['Miscellaneous']
                or 'Experiment_Title'     not in f.f['Miscellaneous']):
                #pass
                if force:
                    raise RuntimeError('HDF5 File number ' + str(fileNum) + ' Has not been annotated. Please call exp.annotate() to annotate the file.')
                else:
                    print('HDF5 File number ' + str(fileNum) + ' Has not been annotated. Please call exp.annotate() to annotate the file.')
                return False
    except OSError:
        # failed to open file probably, nothing to annotate.
        return False
    except KeyError:
        # file failed to open, probably a special run
        return False
    return True

def getAnnotation(fileNum):
    with ExpFile(fileNum) as f:
        f_misc = f.f['Miscellaneous']
        if (   'Experiment_Rationale' not in f_misc
            or 'Experiment_Result'    not in f_misc
            or 'Experiment_Title'     not in f_misc):
            raise RuntimeError('HDF5 File number ' + str(fileNum) + ' Has not been annotated. Please call exp.annotate() to annotate the file.')
        return (f_misc['Experiment_Title'][0].decode("utf-8"), 
                f_misc['Experiment_Rationale'][0].decode("utf-8"), 
                f_misc['Experiment_Result'][0].decode("utf-8"))

def setPath(day, month, year, repoAddress="J:\\Data repository\\New Data Repository"):
    """
    This function sets the location of where all of the data files are stored. It is occasionally called more
    than once in a notebook if the user needs to work past midnight.

    :param day: A number string, e.g. '11'.
    :param month: The name of a month, e.g. 'November' (must match file path capitalization).
    :param year: A number string, e.g. '2017'.
    :return:
    """
    global dataAddress
    dataAddress = repoAddress + "\\" + year + "\\" + month + "\\" + month + " " + day + "\\Raw Data\\"
    return dataAddress




# Exp is short for experiment here.
class ExpFile:
    """
    a wrapper around an hdf5 file for easier handling and management.
    """
    def __init__(self, file_id=None, old=False):
        """
        if you give the constructor a file_id, it will automatically fill the relevant member variables.
        """
        # copy the current value of the address
        self.f = None
        self.key_name = None
        self.key = None 
        self.pics = None
        self.reps = None
        self.exp_start_time = None
        self.exp_start_date = None
        self.exp_stop_time = None
        self.exp_stop_date = None
        self.data_addr = dataAddress
        if file_id is not None:
            self.f = self.open_hdf5(fileID=file_id)
            if old:
                self.key_name, self.key = self.__get_old_key()
            else:
                self.key_name, self.key = self.get_key()
            self.pics = self.get_pics()
            self.reps = self.f['Master-Parameters']['Repetitions'][0]
            self.exp_start_date, self.exp_start_time, self.exp_stop_date, self.exp_stop_time = self.get_experiment_time_and_date()
    
    
    def __enter__(self):
        return self

    
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.f.close()
        except AttributeError:
            return
            
    
    def open_hdf5(self, fileID=None, useBase=False, openFlag='r'):
        
        if type(fileID) == int:
            path = self.data_addr + "data_" + str(fileID) + ".h5"
        elif useBase:
            # assume a file address itself
            path = self.data_addr + fileID + ".h5"
        else:
            path = fileID
        file = h5.File(path, openFlag)
        self.f = file
        return file
    
    def get_reps(self):
        self.reps = self.f['Master-Parameters']['Repetitions'][0]
        return self.reps

    
    def get_key(self):
        """
        :param file:
        :return:
        """
        keyNames = []
        keyValues = []
        foundOne = False
        nokeyreturn = 'No-Variation', arr([1])
        try:
            for var in self.f['Master-Parameters']['Seq #1 Variables']:
                if not self.f['Master-Parameters']['Seq #1 Variables'][var].attrs['Constant']:
                    foundOne = True
                    keyNames.append(var)
                    keyValues.append(arr(self.f['Master-Parameters']['Seq #1 Variables'][var]))
            if foundOne:
                if len(keyNames) > 1:
                    return keyNames, arr(misc.transpose(arr(keyValues)))
                else:
                    return keyNames[0], arr(keyValues[0])
            else:
                return nokeyreturn
        except KeyError:
            return nokeyreturn
    
    def get_old_key(self):
        """
        :param file:
        :return:
        """
        keyNames = []
        keyValues = []
        foundOne = False
        for var in self.f['Master-Parameters']['Variables']:
            if not self.f['Master-Parameters']['Variables'][var].attrs['Constant']:
                foundOne = True
                keyNames.append(var)
                keyValues.append(arr(self.f['Master-Parameters']['Variables'][var]))
        if foundOne:
            if len(keyNames) > 1:
                return keyNames, arr(transpose(arr(keyValues)))
            else:
                return keyNames[0], arr(keyValues[0])
        else:
            return 'No-Variation', arr([1])
        
    def get_pics(self):
        p_t = arr(self.f['Andor']['Pictures'])
        pics = p_t.reshape((p_t.shape[0], p_t.shape[2], p_t.shape[1]))
        return pics
    
    
    def get_basler_pics(self):
        p_t = arr(self.f['Basler']['Pictures'])
        pics = p_t.reshape((p_t.shape[0], p_t.shape[2], p_t.shape[1]))
        return pics
        
    def get_avg_pic(self):
        pics = self.get_pics()
        avg_pic = np.zeros(pics[0].shape)
        for p in pics:
            avg_pic += p
        avg_pic /= len(pics)
        return avg_pic

    def get_avg_basler_pic(self):
        pics = self.get_basler_pics()
        avg_pic = np.zeros(pics[0].shape)
        for p in pics:
            avg_pic += p
        avg_pic /= len(pics)
        return avg_pic
    
    
    def print_all(self):
        self.__print_hdf5_obj(self.f,'')
    
    def print_all_groups(self):
        self.__print_groups(self.f,'')

        
    def print_parameters(self):
        self.__print_hdf5_obj(self.f['Master-Parameters']['Seq #1 Variables'],'')
        
    def __print_groups(self, obj, prefix):
        """
        Used recursively to print the structure of the file.
        obj can be a single file or a group or dataset within.
        """
        for o in obj:
            if o == 'Functions':
                print(prefix, o)
                self.print_functions(prefix=prefix+'\t')
            elif o == 'Master-Script' or o == "Seq. 1 NIAWG-Script":
                print(prefix,o)
            elif type(obj[o]) == h5._hl.group.Group:
                print(prefix, o)
                self.__print_groups(obj[o], prefix + '\t')
            elif type(obj[o]) == h5._hl.dataset.Dataset:
                print(prefix, o)
            #else:
            #    raise TypeError('???')
        
    def __print_hdf5_obj(self, obj, prefix):
        """
        Used recursively in other print functions.
        obj can be a single file or a group or dataset within.
        """
        for o in obj:
            if o == 'Functions':
                print(prefix, o)
                self.print_functions(prefix=prefix+'\t')
            elif o == 'Master-Script' or o == "Seq. 1 NIAWG-Script":
                print(prefix,o)
                self.print_script(obj[o])
            elif type(obj[o]) == h5._hl.group.Group:
                print(prefix, o)
                self.__print_hdf5_obj(obj[o], prefix + '\t')
            elif type(obj[o]) == h5._hl.dataset.Dataset:
                print(prefix, o, ':',end='')
                self.__print_ds(obj[o],prefix+'\t')
            else:
                raise TypeError('???')
    
    def print_functions(self, brief=True, prefix='', which=None):
        """
        print the list of all functions which were created at the time of the experiment.
        if not brief, print the contents of every function.
        """
        for func in self.f['Master-Parameters']['Functions']:
            if which is not None:
                if func != which:
                    print(func)
                    continue
            print(prefix,'-',func,end='')
            if not brief:
                print(': \n---------------------------------------')
                # I think it's a bug that this is nested like this.
                for x in self.f['Master-Parameters']['Functions'][func]:
                    for y in self.f['Master-Parameters']['Functions'][func][x]:
                        # print(Style.DIM, y.decode('utf-8'), end='') for some reason the 
                        # DIM isn't working at the moment on the data analysis comp...
                        print(y.decode('utf-8'), end='')
                print('\n---------------------------------------\ncount=')
            print('')

    def print_master_script(self):
        # A shortcut
        self.print_script(self.f['Master-Parameters']['Master-Script'])

    def print_niawg_script(self):
        # A shortcut
        self.print_script(self.f['NIAWG']['Seq. 1 NIAWG-Script'])

        
    def print_script(self, script):
        """
        special formatting used for printing long scripts which are stored as normal numpy bytes.
        """
        print(Fore.GREEN,'\n--------------------------------------------')
        for x in script:
            print(x.decode('UTF-8'),end='')
        print('\n--------------------------------------------\n\n', Style.RESET_ALL)
            
    def __print_ds(self, ds, prefix):
        """
        Print dataset
        """
        if type(ds) != h5._hl.dataset.Dataset:
            raise TypeError('Tried to print non dataset as dataset.')
        else:
            if len(ds) > 0:
                if type(ds[0]) == np.bytes_:
                    print(' "',end='')
                    for x in ds:
                        print(x.decode('UTF-8'),end='')
                    print(' "',end='')
                elif type(ds[0]) in [np.uint8, np.uint16, np.uint32, np.uint64, 
                                     np.int8, np.int16, np.int32, np.int64, 
                                     np.float32, np.float64]:
                    for x in ds:
                        print(x,end=' ')
                else:
                    print(' type:', type(ds[0]), ds[0])
            print('')
            
    def get_pic_info(self):
        infoStr = 'Number of Pictures: ' + str(self.pics.shape[0]) + '; '
        infoStr += 'Picture Dimensions: ' + str(self.pics.shape[1]) + ' x ' + str(self.pics.shape[2]) + '\n'
        return infoStr
    
    def get_basic_info(self):
        """
        Some quick easy to read summary info
        """
        infoStr = self.get_pic_info()
        infoStr += 'Variations: ' + str(len(self.key)) + ';\t'
        infoStr += 'Repetitions: ' + str(self.reps) + '\n'
        infoStr += 'Experiment started at (H:M:S) ' + str(self.exp_start_time) + ' on (Y-M-D) ' + str(self.exp_start_date) + ', '
        infoStr += 'And ended at ' + str(self.exp_stop_time) + ' on ' + str(self.exp_stop_date) + '\n'
        if 'Experiment_Rationale' in self.f['Miscellaneous'].keys():
            infoStr += 'Experiment Rationale: ' + str(self.f['Miscellaneous']['Experiment_Rationale'][0].decode("utf-8")) + '\n'
        else:
            infoStr += 'Experiment Rationale: HDF5 NOT ANNOTATED: please call exp.Annotate() to annotate this file.\n'
        if 'Experiment_Result' in self.f['Miscellaneous'].keys():
            infoStr += 'Experiment Result: ' + str(self.f['Miscellaneous']['Experiment_Result'][0].decode("utf-8")) + '\n'
        else:
            infoStr += 'Experiment Result: HDF5 NOT ANNOTATED: please call exp.Annotate() to annotate this file.\n'
        print(infoStr)
        return infoStr

    def get_experiment_time_and_date(self):
        start_date, stop_date, start_time, stop_time = '','','',''
        try:
            start_date = ''.join([x.decode('UTF-8') for x in self.f['Miscellaneous']['Start-Date']])
        except KeyError:
            pass
        try:
            start_time = ''.join([x.decode('UTF-8') for x in self.f['Miscellaneous']['Start-Time']])
        except KeyError:
            pass
        try:
            stop_date = ''.join([x.decode('UTF-8') for x in self.f['Miscellaneous']['Stop-Date']])
        except KeyError:
            pass
        try:
            stop_time = ''.join([x.decode('UTF-8') for x in self.f['Miscellaneous']['Stop-Time']])
        except KeyError:
            pass
        return start_date, start_time, stop_date, stop_time
        #return "","","",""

    