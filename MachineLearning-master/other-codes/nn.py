import numpy as np
import gzip
import pickle

import torch
import torch.nn as nn


class NN():
    """A network architecture of simultaneous localization and
       classification of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """

    class CustomNet(nn.Module):

        def __init__(self, d_in, h1, h2, h3_1, h3_2, d_out1, d_out2):
            # super(CustomNet, self).__init__()
            nn.Module.__init__(self)    
            self.linear1 = nn.Linear(d_in, h1)
            self.linear2 = nn.Linear(h1, h2)

            self.linear3_1 = nn.Linear(h2, h3_1)
            self.linear4_1 = nn.Linear(h3_1, d_out1)

            self.linear3_2 = nn.Linear(h2, h3_2)
            self.linear4_2 = nn.Linear(h3_2, d_out2)
            

        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary (differentiable) operations on Tensors.
            """
            lay1 = self.linear1(x)
            lay1 = nn.functional.relu(lay1)

            lay2 = self.linear2(lay1)
            lay2 = nn.functional.relu(lay2)
            
            lay3_1 = self.linear3_1(lay2)
            lay3_1 = nn.functional.relu(lay3_1)

            ## CHECK HERE TOO!!!
            out_1 = self.linear4_1(lay3_1)
            out_1 = out_1.view(-1, )    # reshape it to a 1d-array
            
            # taken care by BCEWithLogitsLoss
            # out_1 = nn.functional.softmax(out_1, dim=0) 
            
            lay3_2 = self.linear3_2(lay2)
            lay3_2 = nn.functional.relu(lay3_2)
            
            out_2 = self.linear4_2(lay3_2)
            
            return out_1, out_2


    def __init__(self, alpha=.5, epochs=5):
        # super(NN, self).__init__()

        self.alpha = alpha
        self.epochs = epochs

        self.d_in = 3600
        self.h1 = 256
        self.h2 = 64
        self.h3_1 = 32
        self.h3_2 = 32
        self.d_out1 = 1
        self.d_out2 = 2

        self.Net = self.CustomNet(self.d_in, self.h1, self.h2, 
                                    self.h3_1, self.h3_2,
                                    self.d_out1, self.d_out2)

        self.criterion_1 = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
        self.criterion_2 = nn.MSELoss(reduction='elementwise_mean')
        
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None
        self.w5 = None
        self.w6 = None

        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.b4 = None
        self.b5 = None
        self.b6 = None


    def objective(self, X, y_class, y_loc):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the objects.

        Returns:
            Composite objective function value.
        """
        wlist = self.get_model_params()
        
        with torch.no_grad():
            self.Net.linear1.weight = torch.nn.Parameter(torch.from_numpy(wlist[0].T))
            self.Net.linear1.bias = torch.nn.Parameter(torch.from_numpy(wlist[1]))

            self.Net.linear2.weight = torch.nn.Parameter(torch.from_numpy(wlist[2].T))
            self.Net.linear2.bias = torch.nn.Parameter(torch.from_numpy(wlist[3]))

            self.Net.linear3_1.weight = torch.nn.Parameter(torch.from_numpy(wlist[4].T))
            self.Net.linear3_1.bias = torch.nn.Parameter(torch.from_numpy(wlist[5]))

            self.Net.linear3_2.weight = torch.nn.Parameter(torch.from_numpy(wlist[6].T))
            self.Net.linear3_2.bias = torch.nn.Parameter(torch.from_numpy(wlist[7]))

            self.Net.linear4_1.weight = torch.nn.Parameter(torch.from_numpy(wlist[8].T))
            self.Net.linear4_1.bias = torch.nn.Parameter(torch.from_numpy(wlist[9]))

            self.Net.linear4_2.weight = torch.nn.Parameter(torch.from_numpy(wlist[10].T))
            self.Net.linear4_2.bias = torch.nn.Parameter(torch.from_numpy(wlist[11]))

        alpha = self.alpha
        tensor_X = torch.from_numpy(X)
        t_y_class = torch.from_numpy(y_class)
        t_y_class = t_y_class.float()
        t_y_loc = torch.from_numpy(y_loc)

        print("computing outputs:")
        out_1, out_2 = self.Net(tensor_X)
        loss_1 = self.criterion_1(out_1, t_y_class)
        loss_2 = self.criterion_2(out_2, t_y_loc)

        l1 = loss_1.detach().numpy()
        l2 = loss_2.detach().numpy()
        loss = alpha * l1 +  (1 - alpha) * l2        
       
        return loss


    def predict(self, X):
        """Predict class labels and object locations for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Input matrix where each row is a feature vector.

        Returns:
            y_class (numpy ndarray, shape = (samples,)):
                predicted labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                The predicted (vertical, horizontal) locations of the
                objects.
        """
        wlist = self.get_model_params()
        
        with torch.no_grad():
            self.Net.linear1.weight = torch.nn.Parameter(torch.from_numpy(wlist[0].T))
            self.Net.linear1.bias = torch.nn.Parameter(torch.from_numpy(wlist[1]))

            self.Net.linear2.weight = torch.nn.Parameter(torch.from_numpy(wlist[2].T))
            self.Net.linear2.bias = torch.nn.Parameter(torch.from_numpy(wlist[3]))

            self.Net.linear3_1.weight = torch.nn.Parameter(torch.from_numpy(wlist[4].T))
            self.Net.linear3_1.bias = torch.nn.Parameter(torch.from_numpy(wlist[5]))

            self.Net.linear3_2.weight = torch.nn.Parameter(torch.from_numpy(wlist[6].T))
            self.Net.linear3_2.bias = torch.nn.Parameter(torch.from_numpy(wlist[7]))

            self.Net.linear4_1.weight = torch.nn.Parameter(torch.from_numpy(wlist[8].T))
            self.Net.linear4_1.bias = torch.nn.Parameter(torch.from_numpy(wlist[9]))

            self.Net.linear4_2.weight = torch.nn.Parameter(torch.from_numpy(wlist[10].T))
            self.Net.linear4_2.bias = torch.nn.Parameter(torch.from_numpy(wlist[11]))


        tensor_X = torch.from_numpy(X)
        out_1, out_2 = self.Net(tensor_X)
        
        ## Apply sigmoid to out_1 here before predicting
        out_1 = torch.sigmoid(out_1)
        out_1[out_1 >= 0.5] = 1
        out_1[out_1 < 0.5] = 0
        out_1 = out_1.long()

        y_predict = out_1.detach().numpy()
        offset_predict = out_2.detach().numpy()
        return y_predict, offset_predict


    def fit(self, X, y_class, y_loc):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the
                objects.
        """
        epochs = self.epochs
        alpha = self.alpha

        tensor_X = torch.from_numpy(X)
        t_y_class = torch.from_numpy(y_class)
        t_y_class = t_y_class.float()
        t_y_loc = torch.from_numpy(y_loc)
        
        learning_rate = 1e-3
        optimizer = torch.optim.Adagrad(self.Net.parameters(), lr=learning_rate)

        for i in range(epochs):                    
            out_1, out_2 = self.Net(tensor_X)
            loss_1 = self.criterion_1(out_1, t_y_class)
            loss_2 = self.criterion_2(out_2, t_y_loc)
            loss = alpha * loss_1 +  (1 - alpha) * loss_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        params_dict = self.Net.state_dict()
        w1 = params_dict['linear1.weight'].detach().numpy()
        w1 = w1.T
        b1 = params_dict['linear1.bias'].detach().numpy()

        w2 = params_dict['linear2.weight'].detach().numpy()
        w2 = w2.T
        b2 = params_dict['linear2.bias'].detach().numpy()

        w3 = params_dict['linear3_1.weight'].detach().numpy()
        w3 = w3.T 
        b3 = params_dict['linear3_1.bias'].detach().numpy()

        w4 = params_dict['linear3_2.weight'].detach().numpy()
        w4 = w4.T 
        b4 = params_dict['linear3_2.bias'].detach().numpy()

        w5 = params_dict['linear4_1.weight'].detach().numpy()
        w5 = w5.T
        b5 = params_dict['linear4_1.bias'].detach().numpy()

        w6 = params_dict['linear4_2.weight'].detach().numpy()
        w6 = w6.T 
        b6 = params_dict['linear4_2.bias'].detach().numpy()

        self.set_model_params(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6)

        return


    def get_model_params(self):
        """Get the model parameters.

        Returns:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and biases for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and biases for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w4 (numpy ndarray, shape = (64, 32)):
            b4 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w5 (numpy ndarray, shape = (32, 1)):
            b5 (float):
                weights and biases for FC(32, 1) for the logit for
                class probability output

            w6 (numpy ndarray, shape = (32, 2)):
            b6 (numpy ndarray, shape = (2,)):
                weights and biases for FC(32, 2) for location outputs
        """
        w1 = self.w1
        b1 = self.b1
        w2 = self.w2
        b2 = self.b2
        w3 = self.w3
        b3 = self.b3
        w4 = self.w4
        b4 = self.b4
        w5 = self.w5
        b5 = self.b5
        w6 = self.w6
        b6 = self.b6

        return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6


    def set_model_params(self, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):
        """Set the model parameters.

        Arguments:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and biases for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and biases for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w4 (numpy ndarray, shape = (64, 32)):
            b4 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w5 (numpy ndarray, shape = (32, 1)):
            b5 (float):
                weights and biases for FC(32, 1) for the logit for
                class probability output

            w6 (numpy ndarray, shape = (32, 2)):
            b6 (numpy ndarray, shape = (2,)):
                weights and biases for FC(32, 2) for location outputs
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6

        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.b6 = b6

        return


def main():
    np.random.seed(0)

    def classification_score(y_pred, test_y):      
        N = test_y.shape[0]
        count = sum(y_pred == test_y)
        accuracy = count/N
        return accuracy
    
    def mse_error(y_pred, test_y):
        err = np.square(y_pred - test_y).mean()
        return err


    with gzip.open('../data/nn_data.pkl.gz', 'rb') as f:
        (train_X, train_y_class, train_y_loc,
            test_X, test_y_class, test_y_loc) = pickle.load(f)
        a = 0
        while a <= 1:
            model = NN(alpha=a, epochs=100)
            model.fit(train_X, train_y_class, train_y_loc)
            
            ytr_class_predict, ytr_loc_predict = model.predict(train_X)
            tr_cl_acc = classification_score(ytr_class_predict, train_y_class)
            tr_ms_err = mse_error(ytr_loc_predict, train_y_loc)

            print("\nAlpha: ", a)
            print("Train classi acc:", tr_cl_acc)
            print("Train mse err", tr_ms_err)

            yts_class_predict, yts_loc_predict = model.predict(test_X)
            ts_cl_acc = classification_score(yts_class_predict, test_y_class)
            ts_ms_err = mse_error(yts_loc_predict, test_y_loc)

            print("Test classi acc:", ts_cl_acc)
            print("Test mse err", ts_ms_err)
            a += 0.1

        # print("Performing experiments with alpha")
        # vals = np.linspace(0.1, 1, 11)
        # cls_list = []
        # mse_list = []

        # for a in vals:
        #     print("\nAlpha: ", a)

        #     model = NN(alpha=a, epochs=100)
        #     model.fit(train_X, train_y_class, train_y_loc)
            
        #     yts_class_predict, yts_loc_predict = model.predict(test_X)
        #     ts_cl_acc = classification_score(yts_class_predict, test_y_class)
        #     ts_ms_err = mse_error(yts_loc_predict, test_y_loc)
            
        #     cls_list.append(ts_cl_acc)
        #     mse_list.append(ts_ms_err)           
            
        #     print("Test classi acc:", ts_cl_acc)
        #     print("Test mse err", ts_ms_err)

if __name__ == '__main__':
    main()


# np.random.seed(0)
# with gzip.open('../data/nn_data.pkl.gz', 'rb') as f:
#     (train_X, train_y_class, train_y_loc,
#         test_X, test_y_class, test_y_loc) = pickle.load(f)

#     model = NN(epochs=10)
    
#     ten_X = torch.from_numpy(train_X)
#     o = model.Net(ten_X)
#     print (o[0].shape, train_y_class.shape)

#     l = model.objective(train_X, train_y_class, train_y_loc)
#     print(l)

#     wtlist = model.get_model_params()

#     model.set_model_params(wtlist[0], wtlist[1], wtlist[2],
#                             wtlist[3], wtlist[4], wtlist[5],
#                             wtlist[6], wtlist[7], wtlist[8],
#                             wtlist[9], wtlist[10], wtlist[11])                            

#     for name, param in model.Net.named_parameters():
#         print (name, type(param), param.shape)
    
#     print('\n')
#     params = model.Net.state_dict()
#     for key in params:
#         print(key)


