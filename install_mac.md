# MacOS Install

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
