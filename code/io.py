import os
import sys
import mcmc
import random as rd
import numpy as np
from collections import OrderedDict as od
from datetime import date


def log_parameters(data,command_line):
  log     = open(command_line.folder+'/log.param','w')
  param_file = open(command_line.par,'r')
  log.write("#-----Class {0} (subversion {1})-----\n\n".format(data.version,data.subversion))
  for line in param_file:
    log.write(line)
  param_file.close()
  log.close()

def log_Class_args(data,command_line):
  log     = open(command_line.folder+'/log.param','a')
  log.write('\n\n#-----------Class-arguments---------\n')
  log.write('data.Class_args.update({0})\n'.format(data.Class_args))
  log.close()

def print_parameters(out,param):
  out.write('#  -LogLkl\t')
  for i in range(len(param)):
    out.write('{0}\t'.format(param[i]))
  out.write('\n')

def print_vector(out,N,loglkl,data):
  for j in range(len(out)):
    out[j].write('%d  %.3f\t' % (N,-loglkl))
    for i in range(len(data.vector)):
      out[j].write('%.6f\t' % data.vector[i])
    out[j].write('\n')

def refresh_file(data):
  data.out.close()
  data.out=open(data.out_name,'a')

def create_output_files(command_line,data):
  if command_line.restart is None:
    number = command_line.N
  else:
    command_line.folder = command_line.restart.split('/')[0]
    number = int(command_line.restart.split('/')[-1].split('__')[0].split('_')[1]) + command_line.N

  logname='log.dat'
  data.log=open(command_line.folder+logname,'a')
  print '  Appending to {0}{1}.'.format(command_line.folder,logname)
  # output file
  outname_base='{0}_{1}__'.format(date.today(),number)
  suffix=0
  for files in os.listdir(command_line.folder):
    if files.find(outname_base)!=-1:
      if int(files.split('__')[-1])>suffix:
        suffix=int(files.split('__')[-1])
  suffix+=1
  data.out=open(command_line.folder+outname_base+str(suffix),'w')
  print '  Creating {0}{1}{2}'.format(command_line.folder,outname_base,suffix)
  data.out_name='{0}{1}{2}'.format(command_line.folder,outname_base,suffix)
  # in case of a restart, copying the whole thing in the new file
  if command_line.restart is not None:
    for line in open(command_line.restart,'r'):
      out.write(line)

def clean(folder):
  if os.path.isdir(folder) is False:
    print 'You must provide a valid directory to clean'
    exit()
  print 'Cleaning the following output directory: '+folder
  action=0
  log=open(folder+'/log.dat') # recover all successful chains
  chains=[]
  for line in log:
    chains.append(line.split(':')[0].strip(' '))
  files=[] # list all files in folder, that have no extensions,
  for File in os.listdir(folder):
    if '.' not in File:
      if File not in chains:
	os.remove(folder+'/'+File)
	print ' removing: '+File
	action+=1
  if action==0:
    print '{0} is already perfectly clean'.format(folder)
  else:
    print '{0} now perfectly clean'.format(folder)

def write_log(data,rate,LogLike):
  data.log.write('{0} :\t[ '.format(data.out.name.split('/')[-1]),)
  for i in range(len(data.param_names)):
    data.log.write('{0} '.format(data.param_names[i]),)
  data.log.write(']\tacceptance rate: %.4f,\t Min (-LogLike): %.4f' % (rate,-LogLike)+'\n')
  return 0

class File(file):
  def head(self, lines_2find=1):
    self.seek(0)                            #Rewind file
    return [self.next() for x in xrange(lines_2find)]

  def tail(self, lines_2find=1):  
    self.seek(0, 2)                         #go to end of file
    bytes_in_file = self.tell()             
    lines_found, total_bytes_scanned = 0, 0
    while (lines_2find+1 > lines_found and
       bytes_in_file > total_bytes_scanned): 
      byte_block = min(1024, bytes_in_file-total_bytes_scanned)
      self.seek(-(byte_block+total_bytes_scanned), 2)
      total_bytes_scanned += byte_block
      lines_found += self.read(1024).count('\n')
    self.seek(-total_bytes_scanned, 2)
    line_list = list(self.readlines())
    return line_list[-lines_2find:]
