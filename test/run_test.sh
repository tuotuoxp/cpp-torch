#!/bin/sh


# $1 - the bin directory of torch
# $2 - the root directory of cmake build-tree
# $3 - the lua testcase

input_t7=/tmp/input.t7
output_t7=/tmp/output.t7
net_t7=/tmp/net.t7


# generate t7 files for comparison
$1/luajit "$3" $input_t7 $output_t7 $net_t7
if [ $? != 0 ]
then
    echo Revoke luajit failed
    exit 1
fi


# run test
$2/test/cpptorch_tester -i $input_t7 -o $output_t7 -n $net_t7
if [ $? != 0 ]
then
    echo Run test failed
    exit 2
fi


# cleanup
rm -f $input_t7 $output_t7 $net_t7
exit 0

