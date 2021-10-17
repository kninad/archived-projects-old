import numpy as np
from pathlib import Path
# filename = str(Path(__file__).parent / 'model_file.xyz')

# import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
TODO:
- experiment with different noise levels + archs
- introduce model regularization -- activity regularizer
- convergence iters -- num epochs
- analyze loss value -- bcs vs mse + visual results
- conv layes for better layers + sparisty constraint
'''


def util_add_noise(clean_data, noise_type='gauss'):
    ''' Function to create synthetic training data for the denoising AE training
        by adding noise to the clean data.
    
    Args:
        clean_data: The clean data array to which noise has to be added.
        noise: Type of noise to be added, Default is 'gauss' i.e gaussian noise.
               Other acceptable values is 'saltpepper'
    
    Returns:
        n_data: The noisy data array to be used in training.    
    '''
    
    n_data = np.copy(clean_data)
    if noise_type == 'gauss':
        nfactor = 0.40 # max 0.60
        mean = 0.0
        std = 1.0  
        noise_matrix = np.random.normal(loc=mean, scale=std, size=n_data.shape)
        n_data += nfactor * noise_matrix
        n_data = np.clip(n_data, 0, 1)
    elif noise_type == 'saltpepper':
        prob = 0.20  # max 0.25
        rand_matrix = np.random.random(size=n_data.shape)
        n_data[rand_matrix < prob] = 0.0
        n_data[rand_matrix > (1-prob)] = 1.0
    else:
        print("Invalid noise string specified")
        return None
        
    return n_data


def util_imgshow(imgvector):
    import matplotlib.pyplot as plt
    img = imgvector.reshape(28,28)
    plt.figure()
    plt.imshow(img)
    plt.show()
    return


def util_visual(noisy, denoised, value):    
    # from matplotlib import pyplot as plt    
    # noisy_img = noisy[idx].reshape(28,28)
    # denos_img = denoised[idx].reshape(28,28)

    # plt.imshow(noisy_img)
    # fname_noisy = str(idx) + noise_type + '-sample-noisyimg-' + str(value)
    # plt.savefig(fname_noisy)
    # plt.imshow(denos_img)
    # fname_den = str(idx) + noise_type + '-sample-denoised-' + str(value)
    # plt.savefig(fname_den)

    import matplotlib.pyplot as plt    
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(denoised[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # plt.show()  
    fname = 'sample-' + str(value)
    plt.savefig(fname)
    return


def util_l1_norm(xvec):
    xvec = xvec.view(-1)
    return torch.abs(xvec).sum()


def util_mse(pred, target):
    return np.square(pred - target).mean()



class Denoiser:

    class DaeNet(nn.Module):

        def __init__(self, d_in, h1):        
            # super(CustomNet, self).__init__()
            nn.Module.__init__(self)    
            self.enc1 = nn.Linear(d_in, h1)
            self.dec1 = nn.Linear(h1, d_in)

        def forward(self, x):           
            oe1 = nn.functional.relu(self.enc1(x))            
            out_dec = nn.functional.sigmoid(self.dec1(oe1))
            
            encoding = oe1
            return out_dec, encoding


    def __init__(self):

        self.epochs = 15 #num_epochs
        self.lrate = 1e-3
        self.noise = 'gauss' #noise_type
        self.loss_type = 'bce' #loss_func
        self.lamb = 1e-6    # for sparsity constraint on activations
        self.modfile = 'model-initial.pt' #fname

        self.d_in = 784
        self.h1 = 64 #h1
        
        self.Net = self.DaeNet(self.d_in, self.h1)
                
        if self.loss_type == 'mse':            
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'bce':            
            self.criterion = nn.BCELoss()            
        else:
            print("Incorrect loss! Going with BCE.") 
            self.criterion = nn.BCELoss()
            self.loss_type = 'bce'
       

    def fit(self, clean_data):
        """Fit the denoising model using clean data

        Arguments:
            clean_data: (numpy ndarray, shape = (n_samples, n_features))
                Uncorrupted data for training.
        """
        
        num_epochs = self.epochs
        noisy_data = util_add_noise(clean_data, noise_type=self.noise)    

        batch_size = 128
        learning_rate = self.lrate
        optimizer = torch.optim.Adam(self.Net.parameters(), lr=learning_rate)

        print('Begin training using loss function: ' + str(self.loss_type))        
        loss_epochs = []
        
        N, _ = clean_data.shape
        upper_lim = int(np.ceil(N/batch_size))

        for t in range(num_epochs):                       
            loss_val = []
            for b in range(upper_lim):                   
                
                low = b * batch_size
                high = (b+1) * batch_size
                if low > N-1:
                    break
                if high > N:
                    high = N
                
                # print(low, high)

                batch_clean = clean_data[low: high]
                batch_clean = torch.FloatTensor(batch_clean)  

                batch_noisy = noisy_data[low: high]
                batch_noisy = torch.FloatTensor(batch_noisy)               
     
                output_dec, encoding = self.Net(batch_noisy)
                
                data_loss = self.criterion(output_dec, batch_clean) 
                regu_loss = self.lamb * util_l1_norm(encoding) 
                loss = data_loss + regu_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val.append(loss.item())    
            
            loss_epochs.append(np.average(loss_val))
            print("Loss at epoch ", t+1, np.average(loss_val))
            
    
        print('\nSaving the trained model')
        self.save_model()
        return


    def denoise(self, noisy_data):
        """Denoise the noisy data

        Arguments:
            noisy_data: (numpy ndarray, shape = (n_samples, n_features))
                Corrupted data.

        Returns:
            denoised_data: (numpy ndarray, shape = (n_samples, n_features))
                Denoised data with the same size of noisy_data.
                NOTE: do NOT perform in-place operation to overwrite the
                content of noisy_data.
        """
        dummy_input = noisy_data.copy()
        tensor_in = torch.from_numpy(dummy_input)
        tensor_out, _ = self.Net(tensor_in)        
        denoised_data = tensor_out.detach().numpy()       
        return denoised_data


    def save_model(self):
        """Save the model to disk

        Arguments: none
        """
        fname = self.modfile
        from pathlib import Path
        filename = str(Path(__file__).parent / fname)         
        torch.save(self.Net.state_dict(), filename)
        return


    def load_model(self):
        """Load the saved model from disk

        Arguments: none
        """
        fname = self.modfile
        from pathlib import Path
        filename = str(Path(__file__).parent / fname)
        
        self.Net = self.DaeNet(self.d_in, self.h1)
        self.Net.load_state_dict(torch.load(filename))
        self.Net.eval()
        return



# clean_data = np.load('../data/dataA.npz')['train_image']
# noisy_data = np.load('../data/dataB.npz')['test_image']

# model = Denoiser()
# # model.fit(clean_data)
# model.load_model()
# denoised_data = model.denoise(noisy_data)


def main():
    clean_data = np.load('../data/dataA.npz')['train_image']
    noisy_data = np.load('../data/dataB.npz')['test_image']

    model = Denoiser()
    # model.fit(clean_data)
    model.load_model()
    denoised_data = model.denoise(noisy_data)


if __name__ == '__main__':
    main()
