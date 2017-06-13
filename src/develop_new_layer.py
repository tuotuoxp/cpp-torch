#!/usr/bin/python
import os
dirs=['include/nn','src/nn','src/serializer']
template_layer='LogSoftMax'
new_layer='Tanh'
for dir in dirs:
    files=[f for f in os.listdir(dir) if f.find(template_layer)>=0]
    for file in files:
        with open(os.path.join(dir,file),'r') as f:
            lines=f.readlines()
        idx=file.find(template_layer)
        newfile=file[:idx]+new_layer+file[idx+len(template_layer):]
        with open(os.path.join(dir,newfile),'w') as f:
            for l in lines:
                idx=l.find(template_layer)
                while idx >= 0:
                    l=l[:idx]+new_layer+l[idx+len(template_layer):]
                    idx=l.find(template_layer)
                f.write(l)
files=['include/cpptorch.h','src/reader.h.inl','src/export.cpp','src_cuda/export.cpp']
for file in files:
    with open(file,'r') as f:
        lines=f.readlines()
    idx=file.find(template_layer)
    with open(file,'w') as f:
        for l in lines:
            if l.find(new_layer)<0:
                f.write(l)
            idx=l.find(template_layer)
            if idx>=0:
                while idx >= 0:
                    l=l[:idx]+new_layer+l[idx+len(template_layer):]
                    idx=l.find(template_layer)
                f.write(l)