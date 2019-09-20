import numpy as np 
import ModelUtil as mu
import tensorflow as tf
import tensorflow.compat.v1 as tf1

co = mu.CmdOptions()
md = mu.ModelBuilder(job=co.job)
#D = np.load('sphere.npy')
D = np.load('sphere2.npy')
N, yDim = D.shape[0],  D.shape[1]
md.r0, md.decay, md.batchSize, layers, L, repDim = 0.001, 0.99, 100*co.jj, 3*[24], 200, 8

md.InitModel(0, yDim)
R = tf.Variable(np.random.uniform(0, 0.1, [L, repDim]).astype(np.float32))
md.inputHod = tf1.placeholder(tf.int32, shape=[None], name='InputHolder')
md.top = tf.gather(R, md.inputHod)
md.AddLayers(layers)
md.AddLayers(yDim, activation=tf.nn.sigmoid)
md.AddScalingTo(D)

batchDist = tf.reduce_sum(tf.square(md.top - md.Label()), axis=1)
md.cost = batchDist[tf.argmin(batchDist)]
md.SetAdamOptimizer(co.epochs, N)
md.InitAllVariables()

for md.lastEpoch in range(1, co.epochs+1):
    md.lastError = 0.0
    for row in range(N):
        inTensor = np.random.randint(0, L, size=(md.batchSize))
        _, err = md.sess.run([md.trainTarget, md.cost], {md.inputHod:inTensor, md.outputHod:D[row:row+1,:]})
        md.lastError += err
    md.lastError /= N
    print('%d: %.1f'%(md.lastEpoch, md.lastError))
    md.log.ReportCost(md.lastEpoch, md.lastError, md.job)

md.log.ShowMatrix(md.sess.run(R), view=8)
md.log.RunScript('pp.Reset().Start().Show3DView(); pp.Close()')

