#!/usr/bin/env python
# -*- coding:utf8 -*-

from layers import *
from deepid_class import *

import cPickle
import gzip
import os
import sys
import time
import numpy
import theano

class DeepIDGenerator:
    def __init__(self, exist_params):
        self.rng = numpy.random.RandomState(1234)
        self.exist_params = exist_params

    def layer_params(self, nkerns, batch_size):
        src_channel = 3
        self.layer1_image_shape  = (batch_size, src_channel, 55, 47)
        self.layer1_filter_shape = (nkerns[0],  src_channel, 4, 4)
        self.layer2_image_shape  = (batch_size, nkerns[0], 26, 22)
        self.layer2_filter_shape = (nkerns[1], nkerns[0], 3, 3)
        self.layer3_image_shape  = (batch_size, nkerns[1], 12, 10)
        self.layer3_filter_shape = (nkerns[2], nkerns[1], 3, 3)
        self.layer4_image_shape  = (batch_size, nkerns[2], 5, 4)
        self.layer4_filter_shape = (nkerns[3], nkerns[2], 2, 2)
        self.result_image_shape  = (batch_size, nkerns[3], 4, 3)

    def build_layer_architecture(self, n_hidden, acti_func=relu):
        '''
        simple means the deepid layer input is only the layer4 output.
        layer1: convpool layer
        layer2: convpool layer
        layer3: convpool layer
        layer4: conv layer
        deepid: hidden layer
        '''
        x = T.matrix('x')
        
        layer1_input = x.reshape(self.layer1_image_shape)
        self.layer1 = LeNetConvPoolLayer(self.rng,
                input        = layer1_input,
                image_shape  = self.layer1_image_shape,
                filter_shape = self.layer1_filter_shape,
                poolsize     = (2,2),
                W = self.exist_params[5][0],
                b = self.exist_params[5][1],
                activation   = acti_func)

        self.layer2 = LeNetConvPoolLayer(self.rng,
                input        = self.layer1.output,
                image_shape  = self.layer2_image_shape,
                filter_shape = self.layer2_filter_shape,
                poolsize     = (2,2),
                W = self.exist_params[4][0],
                b = self.exist_params[4][1],
                activation   = acti_func)

        self.layer3 = LeNetConvPoolLayer(self.rng,
                input        = self.layer2.output,
                image_shape  = self.layer3_image_shape,
                filter_shape = self.layer3_filter_shape,
                poolsize     = (2,2),
                W = self.exist_params[3][0],
                b = self.exist_params[3][1],
                activation   = acti_func)

        self.layer4 = LeNetConvLayer(self.rng,
                input        = self.layer3.output,
                image_shape  = self.layer4_image_shape,
                filter_shape = self.layer4_filter_shape,
                W = self.exist_params[2][0],
                b = self.exist_params[2][1],
                activation   = acti_func)

        layer3_output_flatten = self.layer3.output.flatten(2)
        layer4_output_flatten = self.layer4.output.flatten(2)
        deepid_input = T.concatenate([layer3_output_flatten, layer4_output_flatten], axis=1)

        self.deepid_layer = HiddenLayer(self.rng,
                input = deepid_input,
                n_in  = numpy.prod( self.result_image_shape[1:] ) + numpy.prod( self.layer4_image_shape[1:] ),
                n_out = n_hidden,
                W = self.exist_params[1][0],
                b = self.exist_params[1][1],
                activation = acti_func)
    
        self.generator = theano.function(inputs=[x],
                outputs=self.deepid_layer.output)

    def generate_deepid(self, x):
        deepid_data = self.generator(x)
        return deepid_data

def deepid_generating(dataset_folder, params_file, nkerns, n_hidden, acti_func=relu):    
    pd_helper = ParamDumpHelper(params_file)
    exist_params = pd_helper.get_params_from_file()
    if len(exist_params) != 0:
        exist_params = exist_params[-1]
    else:
        print 'error, no trained params'
        return

    x, y = load_data_xy(dataset_folder)
    deepid = DeepIDGenerator(exist_params)
    deepid.layer_params(nkerns, x.shape[0])
    deepid.build_layer_architecture(n_hidden, acti_func)
    new_x = deepid.generate_deepid(x)
    return (new_x, y), y

def load_data_xy(dataset_path):
    f = open(dataset_path, 'rb')
    x, y = pickle.load(f)
    f.close()
    return x,y

def outsource(vector_file, param_file):

    nkerns  = [20,40,60,80]
    n_hidden = 160
    return deepid_generating(vector_file, param_file, nkerns, n_hidden, acti_func=relu)