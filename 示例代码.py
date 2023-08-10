import test_code.test as t
import numpy as np

if __name__ == '__main__':
    x, y, z = np.load('XX.npy'), np.load('YY.npy'), np.load('ZZ.npy')
    z = np.transpose(z)
    # 训练神经网络模型
    _,model,best_epoch,best_loss,z = t.train(x, y, z,0,1)
    print(best_epoch,best_loss)
    print(model)
    print(z)
