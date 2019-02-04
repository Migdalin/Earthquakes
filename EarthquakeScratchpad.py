



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
showError = True

with tf.Session() as sess:
    data = tf.placeholder(dtype=tf.float32, 
                          shape=(None,150000,1), 
                          name='data')

    dataShape = tf.shape(data)

    numCols = 80
    numRows = int(150000 / numCols)
    data2D = tf.reshape(data, (tf.shape(data)[0], numRows, numCols, 1))

    kernelInit = tf.orthogonal_initializer()
    conv1 = BuildConv2D(
            ConvArgs(layerInput = data2D, 
                     numFilters = 8,
                     filterSize = 8,
                     stride = 4,
                     init = kernelInit,
                     namePrefix='c1'))

    conv2 = BuildConv2D(
            ConvArgs(layerInput = conv1, 
                     numFilters = 16,
                     filterSize = 8,
                     stride = 4,
                     init = kernelInit,
                     namePrefix='c2'))

    conv3 = BuildConv2D(
            ConvArgs(layerInput = conv2, 
                     numFilters = 16,
                     filterSize = 3,
                     stride = 3,
                     init = kernelInit,
                     namePrefix='c3'))

    c3_flattened = tf.layers.Flatten()(conv3)
 
    if(showError == True):
        #        lstmCell = tf.contrib.rnn.LSTMBlockCell(num_units=16)
        #    
        #        rawOutputs, final_state = tf.nn.dynamic_rnn(
        #                lstmCell, 
        #                c3_flattened, 
        #                dtype=tf.float32)
        fc_1 = tf.layers.Dense(units = 64, 
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=kernelInit,
                               name='fc_1')(c3_flattened)
        output = tf.layers.Dense(units=1,
                                 kernel_initializer=kernelInit,
                                 name='output')(fc_1)

    sess.run(tf.global_variables_initializer())
    result = sess.run(c3_flattened, feed_dict={data:  np.random.randint(-5, 12, size=(2,150000,1))})
    print(result.shape)
    print(result[:, 0:20])
    
    if(showError == True):
        #resultA, resultB = sess.run([rawOutputs, final_state], feed_dict={data:  np.random.randint(-5, 12, size=(2,150000,1))})
        #print(resultA.shape)
        #print(resultB.shape)
        result = sess.run(output, feed_dict={data:  np.random.randint(-5, 12, size=(2,150000,1))})
        print(result.shape)
