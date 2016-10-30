We support the following platforms:
- [Linux](#Linux)
- [Windows](#Windows)
- [MacOS](#MacOS)

# Linux

## Prerequisite
- [OpenBLAS](http://www.openblas.net/)
- or [MKL](https://software.intel.com/en-us/intel-mkl)

Our wrapper is able to run without these prerequisites, but may be very slow.

For GPU version, also install
- [CUDA SDK](https://developer.nvidia.com/cuda-75-downloads-archive)

CUDA 7.5 is testied. Try the latest version on your own risk.

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
cd ..\..\
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
cd ..\..\
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

For GPU version, set -DWITH_CUDA=1:
```
git clone https://github.com/tuotuoxp/cpp-torch
cd cpp-torch
mkdir build
cd build
cmake -DWITH_CUDA=1 -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ..
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
# Windows
## Prerequisite
- [CMake](https://cmake.org/), remember to add CMake to PATH
- [Visual Studio 2015](https://www.visualstudio.com/downloads/)

For GPU version, also install
- [CUDA SDK](https://developer.nvidia.com/cuda-75-downloads-archive)

CUDA 7.5 is testied. Try the latest version on your own risk.

## Install torch core
Next we are going to install torch's kernel libraries: TH, THNN, THC, THCUNN. Make sure all the kernel libraries are under the same folder:
```
D:\cpp-torch\ (you can change it to your own place)
├─ torch7 (TH)
├─ nn (THNN)
├─ cutorch (THC, for GPU version)
└─ cunn (THCUNN, for GPU version)
```
All the following commands are run using `VS2015 x64 Native Tools Command Prompt`.

### TH
The following commands install a modified version of torch's TH library.
```
git clone https://github.com/tuotuoxp/torch7.git
cd torch7
mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../install ../lib/TH
nmake
nmake install
cd ..\..\
```
Code and logic of the original library is intact. We only strip its dependency on torch.

### THNN
The following commands install a modified version of torch's THNN library.
```
git clone https://github.com/tuotuoxp/nn.git
cd nn
mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../install -DCMAKE_PREFIX_PATH=../../install ../lib/THNN
nmake
nmake install
cd ..\..\
```
Same as previous, we only modify the dependency of the library.

### THC
> If only CPU version is required, ignore this step.

```
git clone https://github.com/tuotuoxp/cutorch.git
cd cutorch
mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../install ..lib/THC
nmake install
cd ..\..\
```
Same as previous, we only modify the dependency of the library. It takes about half an hour the finish the compilpation. Please kindly ignore the warnings.

### THCUNN
> If only CPU version is required, ignore this step.

```
git clone https://github.com/tuotuoxp/cunn.git
cd cunn
mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../install ..lib/THCUNN
nmake install
cd ..\..\
```
Same as previous, we only modify the dependency of the library. Please kindly ignore the warnings.

## Install torch wrapper
The following commands install our C++ wrapper: cpp-torch to replace the lua wrapper in original torch.

Wrapper is under the same directory as core libraries.
```
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
cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=../../install -DCMAKE_PREFIX_PATH=../../install ..
nmake
nmake install
cd ..\..\
```

For GPU version, set -DWITH_CUDA=1:
```
git clone https://github.com/tuotuoxp/cpp-torch
cd cpp-torch
mkdir build
cd build
cmake -DWITH_CUDA=1 -DCMAKE_INSTALL_PREFIX=/usr/local/cpp-torch -DCMAKE_PREFIX_PATH=/usr/local/cpp-torch ..
make
make install
cd ../../
```

## Test it!
Use the following commands to create a simple VS 2015 solution to test your installation.
```
cd cpp-torch/example/basic
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_PREFIX_PATH=../../../install .
```

Copy TH.dll, THNN.dll and cpptorch.dll from `cpp-torch/install/bin/` to project's binary folder `cpp-torch/example/basic/Debug/` and `cpp-torch/example/basic/Release/`.

Open solution with Visual Studio 2015, run cpptorch_demo project. It should yield the following output:
```
nn.Linear

 12
 13
[torch.FloatTensor of size 2]


 12
 13
[torch.FloatTensor of size 2]
```

# MacOS

## Prerequisite
- [XCode](http://developer.apple.com/xcode/)

## Install
Same as Linux CPU version.

## Test it!
Use the following commands to create a simple XCode solution to test your installation.
```
cd cpp-torch/example/basic
cmake -G "Xcode" -DCMAKE_PREFIX_PATH=../../../install .
```

Open solution with XCode, build and run cpptorch_demo project. It should yield the following output:
```
nn.Linear

 12
 13
[torch.FloatTensor of size 2]


 12
 13
[torch.FloatTensor of size 2]
```
