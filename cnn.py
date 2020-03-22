import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        #filters is a 3d array with dim (num_filters, 3,3)
        #and we divide by 9 to reduce variance of initialization
        self.filters = np.random.randn(num_filters,3,3)/9.

    def iterate_regions(self,image):
        #generates all possible 3x3 image regions using a valid padding scheme
        # - image is a 2d numpy array
        h,w = image.shape
        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3),j:(j+3)]
                yield img_region, i,j

    def forward(self, input):
        #performs a forward pass of the conv layer using the given input
        #returns a 3d numpy array with dims (h,w,num_filters)
        self.last_input = input
        h,w = input.shape
        output = np.zeros((h-2,w-2,self.num_filters))
        for im_region, i,j in self.iterate_regions(input):
            output[i,j] = np.sum(im_region * self.filters, axis=(1,2))
        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i,j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i,j,f] * im_region
        #that is ALL the gradient code. This seems rather simple?
        #update filters:
        self.filters -= learn_rate * d_L_d_filters
        return None

class MaxPool2:

    def iterate_regions(self,image):
        #generates non overlapping 2x2 image regions to pool over
        h,w,_ = image.shape
        new_h = h//2
        new_w = w//2
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2 +2),(j*2):(j*2 +2)]
                yield im_region, i,j

    def forward(self, input):
        #performs a forward pass of the maxpool layer using the given input
        #input is a 3d numpy array with dims (h/2,w/2 num_filtesr)
        h,w,num_filters = input.shape
        self.last_input = input
        output = np.zeros((h//2,w//2,num_filters))
        for im_region, i,j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region, axis=(0,1))
        return output

    def backprop(self, d_L_d_out):
        #performs a backward pass for the maxpool layer
        #layers that were in the max get a gradient of 1, all else 0
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i,j in self.iterate_regions(self.last_input):
            h,w,f = im_region.shape
            amax = np.amax(im_region,axis=(0,1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        #if this pixel was the max value, copy the grad to it
                        if im_region[i2,j2,j2] == amax[f2]:
                            d_L_d_input[i*2+i2, j*2+j2, f2] = d_L_d_out[i,j,f2]
        return d_L_d_input

class Softmax:
    #this is essentially a FC layer with a softmax activation function
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len,nodes)/input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        #this is the non-numerically stable way of doing it
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out,learn_rate):
        # we know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            #e^totals
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S**2)
            d_out_d_t[i] = t_exp[i] * (S-t_exp[i])/(S**2)

            #gradients of totals against weights/biases /inputs
            d_t_d_w = self.last_input =
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis] #wtf does this do?
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)



if __name__ == '__main__':
    import mnist
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    conv = Conv3x3(8)
    pool = MaxPool2()

    output = conv.forward(train_images[0])
    output = pool.forward(output)
    out = softmax.forward(out)
    print(output.shape)

    #calculate cross entropy loss
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    print(out, loss, acc)
