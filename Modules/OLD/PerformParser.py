#!/usr/bin/env python
# coding: utf-8

# Perform (e.g. Gabe) data parser
#
#   fp      fall15  - original crap version
#
#


###
# preliminaries
###

from __future__ import division

import numpy as np
import pandas as pd
from numpy import array

###
# helper functions
###


def dictFileToDataFrame(filename):
    '''a simple reader for the Perform dict-log-dump'''
    leek = []

    # remap nan's (which are np.nans which are different from NaNa)
    nan = float('NaN')  # NOQA
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    with open(filename) as f:
        for line in f:
            leek.append(eval(line))

    df = pd.DataFrame(leek)

    # we'll want to do some fun stuff with this
    return df


def toListToArray(l):
    '''a weird piece of glue we needed to get Pandas to properly interpret
    our lists'''
    return np.array(list(l))


def toArray(df, key):
    '''glue to take a dataframe and a key and convert it into an nparray'''
    return toListToArray(df[key].values)


###
# meat
###


def readPerformDictFlat(filename):
    '''Reads a performlab file. This makes a simple, flat version.
    E.g. makes a simple column index instead of a MultiIndex.
    Parses the XYZ stuff, and handles special case of matrices.'''

    # read
    df = dictFileToDataFrame(filename)

    # parse and create new columns
    for name in df.columns.values:
        s = name.split('_')
        if len(s) is 2:
            # otherwise, its already a scalar
            (base, desc) = s

            arr = toArray(df, name)

            if desc == '4x4':
                # for special case of '4x4'
                for index in range(16):
                    df[base + '_' + str(index)] = arr[:, index]
            elif desc == '3x3':
                for index in range(9):
                    df[base + '_' + str(index)] = arr[:, index]
            else:
                # use the letters themselves
                for index in range(len(desc)):
                    df[base + '_' + desc[index]] = arr[:, index]

            df.drop(name, axis=1, inplace=True)

    return df


def readPerformDict(filename):
    '''Reads a performlab file version w/ complicated multiindices.
    Parses the XYZ stuff, and handles special case of matrices.'''

    # read
    df = dictFileToDataFrame(filename)

    groupnames = []
    partnames = []
    datar = []

    # parse and create new columns
    for name in df.columns.values:
        s = name.split('_')
        if len(s) is 2:
            # otherwise, its already a scalar
            (base, desc) = s

            arr = toArray(df, name)

            if desc == '4x4':
                # for special case of '4x4'
                for index in range(16):
                    # df[base+'_'+str(index)] = arr[:,index]
                    groupnames.append(base)
                    partnames.append(str(index))
                    datar.append(arr[:, index])

            elif desc == '3x3':
                for index in range(9):
                    # df[base+'_'+str(index)] = arr[:,index]
                    groupnames.append(base)
                    partnames.append(str(index))
                    datar.append(arr[:, index])

            else:
                # use the letters themselves
                for index in range(len(desc)):
                    # df[base+'_'+desc[index]] = arr[:,index]
                    groupnames.append(base)
                    partnames.append(desc[index])
                    datar.append(arr[:, index])

            # df.drop(name, axis=1, inplace=True)
        else:
            # it's a scalar
            groupnames.append(s[0])
            partnames.append('')
            datar.append(df[name].values)

    dd = np.array(datar)
    return pd.DataFrame(dd.T, columns=[groupnames, partnames], index=np.arange(dd.shape[1]))


###
# test
###


if __name__ == "__main__":
    # main(sys.argv)
    df = readPerformDict("Data/exp/exp_data-2018-3-6-15-31.dict")
    
