import numpy as np 
import ModelUtil as mu
import tensorflow as tf
import tensorflow.compat.v1 as tf1

co = mu.CmdOptions(['', 'Tst', '500', '2', '1', '0', 'A'])
md = mu.ModelBuilder()

#D = md.log.LoadTable0()
D = np.load('sphere.npy')

N, yDim = D.shape[0],  D.shape[1]
L, repDim  = 100, 5
r0, decay = 0.0001, 0.8
layers = [24, 12, 6]

md.InitModel(0, yDim)
R =  tf.Variable(np.zeros([L, repDim], dtype=np.float32))
md.inputHod = tf1.placeholder(tf.int32, shape=[None], name='InputHolder')
md.top = tf.gather(R, md.inputHod)
md.AddLayers(layers)
md.AddLayers(yDim, activation=tf.nn.tanh)
md.top *= 50.0

md.batchSize = 1
md.SetLearningRate(r0)
md.SetLearningDecay(decay)
md.cost = md.SquaredCost(md.top, md.Label())
md.SetAdamOptimizer(co.epochs, N)
md.InitAllVariables()

for md.lastEpoch in range(1, co.epochs+1):
    epErr = 0.0
    for row in range(N):
        inTensor = np.random.randint(0, L, size=(md.batchSize))
        _, err = md.sess.run([md.trainTarget, md.cost], {md.inputHod:inTensor, md.outputHod:D[row:row+1,:]})
        epErr += err
    epErr /= N
    print('%d: %.1f'%(md.lastEpoch, epErr))

md.log.ShowMatrix(md.sess.run(R))
