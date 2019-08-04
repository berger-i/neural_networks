from mnist import MNIST
mndata = MNIST(path='./MNIST', mode='rounded_binarized', return_type='numpy', gz=False)
images, labels = mndata.load_training()
images = images[:10]
labels = labels[:10]
pass
print (1)
