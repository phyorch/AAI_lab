import numpy as np

def Euclidian(traindata, testdata, cls_idx_train, k=100):
    pred = []
    testdata = np.reshape(testdata,(testdata.shape[0], testdata.shape[1]*testdata.shape[2]*testdata.shape[3]))
    traindata = np.reshape(traindata,(traindata.shape[0], traindata.shape[1] * traindata.shape[2] * traindata.shape[3]))
    for td in testdata:
        dis = np.linalg.norm(td - traindata, axis=1)
        idx = np.argsort(dis)[0:100]
        info = np.zeros((1,10))
        for i in idx:
            info[0][cls_idx_train[i]] += 1
        pred.append(np.argsort(info)[0][-1])
    pred = np.array(pred)
    return pred

