
from collections import namedtuple
from time import time
import tensorflow as tf

from Globals import RESHAPED_DIMS, SAMPLE_LENGTH

ModelInfo = namedtuple('ModelInfo', 
                       'data, labels, loss, optimizer, output')
ModelInfo.__new__.__defaults__ = (None,) * len(ModelInfo._fields)

ConvArgs = namedtuple('ConvArgs',
                      'layerInput, numFilters, filterSize, stride, init, namePrefix')
ConvArgs.__new__.__defaults__ = (None,) * len(ConvArgs._fields)

class ConvLearner:
    def __init__(self):
        tf.reset_default_graph()
        self.session = tf.Session()
        self.learning_rate = 1e-5
        self.total_step_count = 0
        self.model = self.BuildModel()
        self.saver = tf.train.Saver()
        self.InitStatsWriter()
        self.session.run(tf.global_variables_initializer())

    def InitStatsWriter(self):        
        self.statsWriter = tf.summary.FileWriter(f"tensorboard/{int(time())}")
        tf.summary.scalar("Loss", self.model.loss)
        self.writeStatsOp = tf.summary.merge_all()
        self.SaveWeightsFilename = "./checkpoint/ConvLearnerCheckpoint.ckpt"

    def WriteStats(self, feedDict):
        summary = self.session.run(self.writeStatsOp, feed_dict=feedDict)
        self.statsWriter.add_summary(summary, self.total_step_count)
        
    def BuildConv2D(self, convArgs):
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

    def BuildModel(self):
        kernelInit = tf.orthogonal_initializer()
    
        data = tf.placeholder(dtype=tf.float32, 
                              shape=(None,) + RESHAPED_DIMS, 
                              name='data')

        numRows = int(SAMPLE_LENGTH / 8)
        data2D = tf.reshape(data, (tf.shape(data)[0], numRows, 8, 1))
    
        conv_1 = self.BuildConv2D(
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
    
        conv_2 = self.BuildConv2D(
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
                conv_2, 
                (c2_shape[0], numRows, 8, 1))
    
        conv_3 = self.BuildConv2D(
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
                conv_3, 
                (c3_shape[0], numRows, 8, 1))
    
        conv_4 = self.BuildConv2D(
                ConvArgs(layerInput = conv3_reshaped, 
                         numFilters = 16,
                         filterSize = 8,
                         stride = 4,
                         init = kernelInit,
                         namePrefix='c4'))

        flattened = tf.layers.Flatten()(conv_4)
        fc_1 = tf.layers.Dense(units = 64, 
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=kernelInit,
                               name='fc_1')(flattened)
        
        output = tf.layers.Dense(units=1,
                                 kernel_initializer=kernelInit,
                                 name='output')(fc_1)

        labels = tf.placeholder(dtype=tf.float32, 
                               shape=(None,1),
                               name="labels")

        loss = tf.reduce_mean(tf.multiply(tf.abs(tf.subtract(labels, output)), 4e2))
        optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate).minimize(loss)

        result = ModelInfo(data=data,
                           labels=labels,
                           loss=loss,
                           optimizer=optimizer,
                           output=output)
        return result        

    def Fit(self, iBatch, iLabels):
        feed_dict={self.model.data: iBatch,
                   self.model.labels: iLabels}
        self.session.run(self.model.optimizer, feed_dict)
    
        self.total_step_count += 1
        if((self.total_step_count % 10) == 0):
            self.WriteStats(feed_dict)

    def Predict(self, iData):
        feedDict={self.model.data: iData}
        result = self.session.run(self.model.output, feed_dict=feedDict)
        return result.flatten()

    def Save(self):
        self.saver.save(self.session, self.SaveWeightsFilename)

    def Load(self):
        if(tf.train.checkpoint_exists(self.SaveWeightsFilename)):
            self.saver.restore(self.session, self.SaveWeightsFilename)
        