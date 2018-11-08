from mnist import MNIST

mndataSet = '.\Dataset'
mndata = MNIST()
#mndata.gz =True
mndata = MNIST(mndataSet)

images,labels = mndata.load_training()

#print(mndata.display(images[1]))

#%%