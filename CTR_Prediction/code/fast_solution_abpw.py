'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>
Modified by : Abhishek Thakur <abhishek4@gmail.com>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from math import log, exp, sqrt


# TL; DR
# the main learning process start at line 122


# parameters #################################################################

#train = 'train.csv'  # path to training file
#test = 'test.csv'  # path to testing file
train = '../data/train'  # path to training file
test = '../data/test'  # path to testing file

D = 2 ** 24  # number of weights use for each model, we have 32 of them
alpha = .1   # learning rate for sgd optimization


# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, traindata=False):
    for t, line in enumerate(open(path)):
        # initialize our generator
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
            x = [0] * 27
            continue
        # parse x
        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            elif traindata and m == 1:
                y = [float(feat)]
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
                if traindata:
                    x[m] = abs(hash(str(m) + '_' + feat)) % D
                else:
                    x[m+1] = abs(hash(str(m+1) + '_' + feat)) % D

        yield (ID, x, y) if traindata else (ID, x)

# B. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     bounded logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

#def logloss_mini(p, Y):
#    p = max(min(p, 1. - 10e-15), 10e-15)
#    return -log(p) if y == 1. else -log(1. - p)


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def predict(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 40.), -40.)))  # bounded sigmoid


def predict_mini(X, w):
    minip = []
    wTx = 0.
    for x in X:
        for i in x:  # do wTx
            wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
        p_i = 1. / (1. + exp(-max(min(wTx, 40.), -40.)))  # bounded sigmoid
        minip.append(p_i)
    return minip


# D. Update given model
# INPUT:
# alpha: learning rate
#     w: weights
#     n: sum of previous absolute gradients for a given feature
#        this is used for adaptive learning rate
#     x: feature, a list of indices
#     p: prediction of our model
#     y: answer
# MODIFIES:
#     w: weights
#     n: sum of past absolute gradients
def update(alpha, w, n, x, p, y):
    for i in x:
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
        n[i] += abs(p - y)
        w[i] -= (p - y) * 1. * alpha / sqrt(n[i])

def update_mini(alpha, w, n, X, P, Y):
    for x,[y],p  in zip(X,Y,P):
        for i in x:
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
            n[i] += abs(p - y)
            w[i] -= ((p - y) * 1. * alpha / sqrt(n[i]))/10.


# training and testing #######################################################
start = datetime.now()

K = [0]

w = [[0.] * D]
n = [[0.] * D]

loss = 0.

tt = 1

X = []
Y = []
bsize = 10
mcount = 0
for ID, x, y in data(train, traindata = True):

    if mcount < bsize:
        X.append(x)
        Y.append(y)
        mcount += 1
    else:
        k = 0
#        print "X ", X
#        print "Y ", Y
#        print "w ", w[k]
#        print "Y[0], ", Y[0]
        P = predict_mini(X, w[k])
        update_mini(alpha, w[k], n[k], X, P, Y)
        for y,p in zip(Y,P):
            loss += logloss(p, y[k])  # for progressive validation
        mcount = 0
        X = []
        Y = []
#        break
#        print "w2 ", w[k]

    # print out progress, so that we know everything is working
    if tt % 1000000 == 0:
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), tt, (loss * 1./tt)))
#        print "k : ", K
#        print "x : ", x
#        print "y : ", y
    tt += 1
        
    # get predictions and train on all labels
#    for k in K:



#for ID, x, y in data(train, traindata = True):

    # get predictions and train on all labels
#    for k in K:
#    k = 0
#    p = predict(x, w[k])
#    update(alpha, w[k], n[k], x, p, y[k])
#    loss += logloss(p, y[k])  # for progressive validation

    # print out progress, so that we know everything is working
#    if tt % 1000000 == 0:
#        print('%s\tencountered: %d\tcurrent logloss: %f' % (
#                datetime.now(), tt, (loss * 1./tt)))
#        print "k : ", K
#        print "x : ", x
#        print "y : ", y
#    tt += 1

with open('submission.csv', 'w') as outfile:
    outfile.write('id,click\n')
    for ID, x in data(test):
        for k in K:
            p = predict(x, w[k])
            outfile.write('%s,%s\n' % (ID, str(p)))

print('Done, elapsed time: %s' % str(datetime.now() - start))
