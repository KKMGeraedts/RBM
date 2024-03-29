import numpy as np

class RBM_numpy():

    def __init__(self, nv, nh):
        self.nv = nv
        self.nh = nh
        
        # Initial weights
        self.W = np.random.normal(0, 0.05, nv + nh + nv * nh)

        # Default learning parameters
        self.lr = 0.01
        self.batch_size = 50
        self.gamma = 1e-5
        self.regul = "L1"

    def _sigma(self, x):
        return 1 / (1 + np.exp(-x))

    def _positive_statistics(self, v_data):
        n = self.nv + self.nh
        
        # Seperate parameter vector into weights and biases
        h_bias = self.W[self.nv:n].reshape(-1, 1)
        w = self.W[n:].reshape(self.nv, self.nh)
        
        # Data operators
        vh_data = np.zeros(len(self.W))
        
        # Add <v> to data operators
        vh_data[:self.nv] = np.mean(v_data, axis=1)
        
        # Compute P(h=1|v)
        ph1v = self._sigma(h_bias + w.T @ v_data) # Shape = (nh, N_samples)

        # Add <h> to data operators (Using P(h=1|v) instead of sampling)
        vh_data[self.nv:n] = np.mean(ph1v, axis=1)
        
        # Compute <vh>
        vh = (v_data @ ph1v.T) / v_data.shape[1] # Shape = (nh, nv)
        vh_data[n:] = vh.reshape(-1)
        
        return vh_data

    def _negative_statistics(self, v_data, k=5):
        n = self.nv + self.nh
        N_samples = v_data.shape[1]
        
        # Seperate parameter vector into weights and biases
        v_bias = self.W[:self.nv].reshape(-1, 1)
        h_bias = self.W[self.nv:n].reshape(-1, 1)
        w = self.W[n:].reshape(self.nv, self.nh)
        
        # Reconstruction operators
        vh_recon = np.zeros(len(self.W))
        
        v_recon = v_data
            
        # Send v_data through the network k-times
        for i in range(k):
            # Forward pass (P(h=1|v))
            ph1v = self._sigma(h_bias + w.T @ v_recon) # Shape = (nh, N_samples)
            h =  ph1v > np.random.rand(self.nh, N_samples) # Shape = (nh, N_samples)
            
            # Backward pass (P(v=1|h))
            pv1h = self._sigma(v_bias + w @ h) # Shape = (nv, N_samples)
            v_recon = pv1h > np.random.rand(self.nv, N_samples) # For the visible units its fine to use the probabilities instead of sampling
        
        # Last forward pass
        ph1v = self._sigma(h_bias + w.T @ v_recon)
        h_recon = ph1v # For the last update we use the probabilities also for the hidden layer
        
        # Add <v_recon> to reconstruction operators
        vh_recon[:self.nv] = np.mean(v_recon, axis=1)
        
        # Add <h_recon> to reconstruction operators
        vh_recon[self.nv:n] = np.mean(h_recon, axis=1)
        
        # Compute <vh>_recon
        vh = (v_recon @ h_recon.T) / N_samples # Shape = (nh, nv)
        
        # Add <vh> to data operators
        vh_recon[n:] = vh.reshape(-1)
        
        return vh_recon
    
    def train(self, v_data, epochs=5000):
        n = self.nv + self.nh
        # Largest axis of 'v_data' is chosen to be the datapoints
        if v_data.shape[0] > v_data.shape[1]:
            N = v_data.shape[0]
            v_data = v_data.T
        else:
            N = v_data.shape[1]

        # Things to keep track of
        reconstruction_errors = []
        da_list = []
        db_list = []
        dw_list = []

        # Start training
        for epoch in range(epochs):

            # Keep track of the weight updates per epoch
            da = 0
            db = 0
            dw = 0

            # Mini-batches
            for _ in range((N // self.batch_size) + 1):

                # Select batch
                batch_idx = np.random.choice(N, self.batch_size, replace=True)
                v_batch = v_data[:, batch_idx]

                # Compute statistics
                vh_data = self._positive_statistics(v_batch)
                vh_recon = self._negative_statistics(v_batch)

                # Weight update
                w_update = (vh_data - vh_recon) * self.lr 

                # Compute reconstruction error
                reconstruction_error = np.sum((vh_recon - vh_data) ** 2)
                reconstruction_errors.append(reconstruction_error)

                # Regularization term
                if self.regul == "L1":
                    w_update += self.gamma * np.sign(self.W)
                elif self.regul == "L2":
                    w_update += self.gamma * self.W

                # Update weight
                self.W += w_update

                # Store updates
                da += np.linalg.norm(self.W[:self.nv])
                db += np.linalg.norm(self.W[self.nv:n])
                dw += np.linalg.norm(self.W[n:].reshape(-1))

            da_list.append(da)
            db_list.append(db)
            dw_list.append(dw)

            # Print progress
            if (epoch+1) % (epochs // 20) == 0:
                print('Epoch [{}/{}], Reconstructions errors: {}'.format(epoch+1, epochs, reconstruction_error))

        return reconstruction_errors, da_list, db_list, dw_list