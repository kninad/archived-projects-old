import numpy as np
from pathlib import Path
# filename = str(Path(__file__).parent / 'model_file.xyz')

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Lower number of filters to 16 and 32
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

        def __init__(self):        
            # super(CustomNet, self).__init__()
            nn.Module.__init__(self)    
            self.encoder = nn.Sequential(
                            nn.Conv2d(1,16,3,padding=1),   # batch x 16 x 28 x 28
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.Conv2d(16,16,3,padding=1),   # batch x 16 x 28 x 28
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.Conv2d(16,32,3,padding=1),  # batch x 32 x 28 x 28
                            nn.ReLU(),
                            nn.BatchNorm2d(32),
                            nn.Conv2d(32,32,3,padding=1),  # batch x 32 x 28 x 28
                            nn.ReLU(),
                            nn.BatchNorm2d(32),
                            nn.MaxPool2d(2,2)   # batch x 64 x 14 x 14
            )            
            self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(32,16,3,1,1),
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.ConvTranspose2d(16,16,3,1,1),
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.ConvTranspose2d(16,1,3,2,1,1),
                            nn.Sigmoid()
            )
      

        def forward(self, x):           
            batch_size = list(x.size())[0]
            imgs = x.view(batch_size, 1, 28, 28)                   
            encoding = self.encoder(imgs)
            out = self.decoder(encoding) 
            # print(out.shape)
            out_dec = out.view(batch_size, 784)
            return out_dec, encoding


    def __init__(self):

        self.epochs = 35 #num_epochs
        self.lrate = 1e-3
        self.noise = 'gauss' #noise_type
        self.loss_type = 'bce' #loss_func
        self.lamb = 1e-6    # for sparsity constraint on activations
        self.modfile = 'model-best-accuracy.pt' #fname
        
        self.Net = self.DaeNet()
                
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
            print("Epoch ", t+1, ": loss = ", np.average(loss_val))
            
        self.save_model()
        print('\nSaved the trained model')
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
        denoised_data = np.zeros_like(dummy_input)
        # ~ tensor_in = torch.from_numpy(dummy_input)
        # ~ tensor_out, _ = self.Net(tensor_in)        
        batch_size = 100
        N, _ = dummy_input.shape
        upper_lim = int(np.ceil(N/batch_size))
        
        for b in range(upper_lim):                                   
            low = b * batch_size
            high = (b+1) * batch_size
            if low > N-1:
                break
            if high > N:
                high = N

            # ~ batch_clean = clean_data[low: high]
            # ~ batch_clean = torch.FloatTensor(batch_clean)  
            batch_in = dummy_input[low:high]
            batch_in = torch.FloatTensor(batch_in)

            output_dec, _ = self.Net(batch_in)
            out_np = output_dec.detach().numpy()
            
            denoised_data[low:high] = out_np               
        
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
        
        self.Net = self.DaeNet()
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
