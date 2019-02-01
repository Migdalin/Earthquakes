



from collections import namedtuple
import tensorflow as tf
import numpy as np


ConvArgs = namedtuple('ConvArgs',
                      'layerInput, numFilters, filterSize, stride, init, namePrefix')
ConvArgs.__new__.__defaults__ = (None,) * len(ConvArgs._fields)

def BuildConv2D(convArgs):
    channelAxis = 3
    filterShape = [convArgs.filterSize, 
                   convArgs.filterSize, 
                   convArgs.layerInput.get_shape()[channelAxis], 
                   convArgs.numFilters]

    filters = tf.get_variable(shape=filterShape, 
                              dtype=tf.float32,
                              initializer=convArgs.init,
                              name=convArgs.namePrefix + 'filters')
    
    conv = tf.nn.conv2d(input=convArgs.layerInput,
                        filter=filters, 
                        strides=[1,convArgs.stride,convArgs.stride,1], 
                        padding='VALID')

    activated = tf.nn.relu(conv)
    return activated

tf.reset_default_graph()

with tf.Session() as sess:
    data = tf.placeholder(dtype=tf.float32, 
                          shape=(None,150000,1), 
                          name='data')

    dataShape = tf.shape(data)


    numRows = int(150000 / 8)
    data2D = tf.reshape(data, (tf.shape(data)[0], numRows, 8, 1))

    kernelInit = tf.orthogonal_initializer()
    conv_1 = BuildConv2D(
            ConvArgs(layerInput = data2D, 
                     numFilters = 8,
                     filterSize = 8,
                     stride = 4,
                     init = kernelInit,
                     namePrefix='c1'))

    c1_flattened = tf.layers.Flatten()(conv_1)
    c1_shape = tf.shape(c1_flattened)
    numRows = tf.cast(c1_shape[1]/8, tf.int32)
    conv1_reshaped = tf.reshape(
            c1_flattened, 
            (c1_shape[0], numRows, 8, 1))

    conv_2 = BuildConv2D(
            ConvArgs(layerInput = conv1_reshaped, 
                     numFilters = 16,
                     filterSize = 8,
                     stride = 4,
                     init = kernelInit,
                     namePrefix='c2'))

    c2_flattened = tf.layers.Flatten()(conv_2)
    c2_shape = tf.shape(c2_flattened)
    numRows = tf.cast(c2_shape[1]/8, tf.int32)
    conv2_reshaped = tf.reshape(
            c2_flattened, 
            (c2_shape[0], numRows, 8, 1))

    conv_3 = BuildConv2D(
            ConvArgs(layerInput = conv2_reshaped, 
                     numFilters = 16,
                     filterSize = 8,
                     stride = 4,
                     init = kernelInit,
                     namePrefix='c3'))

    c3_flattened = tf.layers.Flatten()(conv_3)
    c3_shape = tf.shape(c3_flattened)
    numRows = tf.cast(c3_shape[1]/8, tf.int32)
    conv3_reshaped = tf.reshape(
            c3_flattened, 
            (c3_shape[0], numRows, 8, 1))

    conv_4 = BuildConv2D(
            ConvArgs(layerInput = conv3_reshaped, 
                     numFilters = 16,
                     filterSize = 8,
                     stride = 4,
                     init = kernelInit,
                     namePrefix='c4'))
    c4_flattened = tf.layers.Flatten()(conv_4)
    c4_shape = tf.shape(c4_flattened)
    c4_reshaped = tf.reshape(c4_flattened, (c4_shape[0], c4_shape[1], 1))
    
    lstmCell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=16)

    rawOutputs, final_state = tf.nn.dynamic_rnn(
            lstmCell, 
            c4_reshaped, 
            dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    resultA, resultB = sess.run([rawOutputs, final_state], feed_dict={data:  np.random.randint(-5, 12, size=(2,150000,1))})
    #result = sess.run(c4_flattened, feed_dict={data:  np.random.randint(-5, 12, size=(2,150000,1))})
    print(resultA.shape)
    print(resultB.shape)
    #print(result[:, 0:20])
    
    
