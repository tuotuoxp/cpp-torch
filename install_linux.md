# Linux Install (legacy)

## Prerequisite
- [OpenBLAS](http://www.openblas.net/)
- or [MKL](https://software.intel.com/en-us/intel-mkl)

Our wrapper is able to run without these prerequisites, but may be very slow.

For GPU version, also install
- [CUDA SDK](https://developer.nvidia.com/cuda-75-downloads-archive)

CUDA 7.5 is testified. Try the latest version on your own risk.

## Install torch core
Next we are going to install torch's kernel libraries: TH, THNN, THC, THCUNN. Make sure all the kernel libraries are under the same folder:
```
/usr/local/cpp-torch/ (you can change it to your own location)
├─ torch7 (TH)
├─ nn (THNN)
├─ cutorch (THC, for GPU version)
└─ cunn (THCUNN, for GPU version)
```

### TH
The following commands install a modified version of torch's TH library.
```
git clone https://github.com/tuotuoxp/torch7.git
cd torch7
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch ../lib/TH
make
make install
cd ../../
```
Code and logic of the original library is intact. We only strip its dependency on torch.

### THNN
The following commands install a modified version of torch's THNN library.
```
git clone https://github.com/tuotuoxp/nn.git
cd nn
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ../lib/THNN
make
make install
cd ../../
```
Same as previous, we only modify the dependency of the library.

### THC
> If only CPU version is required, ignore this step.

The following commands install a modified version of torch's THC library.
```
git clone https://github.com/tuotuoxp/cutorch.git
cd cutorch
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ../lib/THC
make
make install
cd ../../
```
Same as previous, we only modify the dependency of the library.
It takes about half an hour the finish the compilpation. Please kindly ignore the warnings.

### THCUNN
> If only CPU version is required, ignore this step.

The following commands install a modified version of torch's THCUNN library.
```
git clone https://github.com/tuotuoxp/cunn.git
cd cunn
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ../lib/THCUNN
make
make install
cd ../../
```
Same as previous, we only modify the dependency of the library.
Please kindly ignore the warnings.

## Install torch wrapper
The following commands install our C++ wrapper: cpp-torch to replace the lua wrapper in original torch.

Wrapper is under the same directory as core libraries.
```
/usr/local/cpp-torch/ (you can change it to your own location)
├─ torch7 (TH)
├─ nn (THNN)
├─ cutorch (THC, for GPU version)
├─ cunn (THCUNN, for GPU version)
└─ cpp-torch (C++ wrapper)
```

For CPU version:
```
git clone https://github.com/tuotuoxp/cpp-torch
cd cpp-torch
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ..
make
make install
cd ../../
```

For GPU version, set -DBUILD_CUDA=ON:
```
git clone https://github.com/tuotuoxp/cpp-torch
cd cpp-torch
mkdir build
cd build
cmake -DBUILD_CUDA=ON -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ..
make
make install
cd ../../
```

## Test it!
Use the following commands to create a simple example to test your installation.
```
cd cpp-torch/example/basic
cmake -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch .
make
```
Run the generated demo in command line:
```
./cpptorch_demo
```
It should yield the following output:
```
nn.Linear

 12
 13
[torch.FloatTensor of size 2]


 12
 13
[torch.FloatTensor of size 2]
```
