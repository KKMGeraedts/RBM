import torch
import numpy as np
import matplotlib.pyplot as plt

class RBM(torch.nn.Module):
    
    def __init__(self, nv=None, nh=None, basis="01"):
        
        super(RBM, self).__init__()
        
        self.nv = nv
        self.nh = nh
        self.basis = basis

        self.w = torch.nn.Parameter(torch.randn(nv, nh) * 0.01)
        self.a = torch.nn.Parameter(torch.zeros(nv))
        self.b = torch.nn.Parameter(torch.zeros(nh))

        # Toggles for monitoring certains things during learning
        self.monitor_overfitting = False
        self.track_spectra = False
        self.v_validation = torch.tensor([])

        # For monitoring learning
        self.f_trains = []
        self.f_valids = []
        self.f_diffs = []

        # Eigenvalue spectra at each epoch
        self.h_spectra = None
        self.v_recon_spectra = None
        self.h_recon_spectra = None

        # Weights and costs at each epoch
        self.W_epochs = None
        self.updates = None
        self.weights = None
        self.costs = None
        self.reconstructions = None

        self.device = None

    def save(self, f):
        """
        Store the attributes of the current RBM class at f.
        """
        # Check
        if self.h_spectra is None or self.v_recon_spectra is None or self.h_recon_spectra is None:
            print(f"This model does not seem to be trained. The arrays containing eigenvalue spectra" \
                  + "are empty. No model was saved.")
            return

        # Create dictionary to save
        attributes_to_save = {
            "nv": self.nv,
            "nh": self.nh,
            "basis": self.basis,
            "v_validation": self.v_validation,
            "f_trains": self.f_trains,
            "f_valids": self.f_valids,
            "f_diffs": self.f_diffs,
            "h_spectra": self.h_spectra,
            "v_recon_spectra": self.v_recon_spectra,
            "h_recon_spectra": self.h_recon_spectra,
            "updates": self.updates,
            "weights": self.weights,
            "costs": self.costs,
            "reconstructions": self.reconstructions,
            "W_epochs": self.W_epochs,
            "device": self.device,
            "state_dict": self.state_dict()
        }

        # Save model
        torch.save(attributes_to_save, f"{f}.pth")

    def load(self, loaded_data):
        """
        Set attributes of self to values stored in loaded_data dictionary.
        """
        # Set loaded attributes to self
        for attr, value in loaded_data.items():
            if attr == "state_dict":
                self.load_state_dict(value)
            else:
                setattr(self, attr, value)

    def _bernoulli(self, pvh, v, dw_norm_list):
        if torch.any(pvh > 1) or torch.any(pvh < 0) or torch.any(torch.isnan(pvh)) or torch.any(torch.isinf(pvh)):
            args1 = pvh[pvh > 1]
            args_negative = pvh[pvh < 0]
            args_nan = torch.isnan(pvh)
            args_inf = torch.isinf(pvh)
            
            print("Values > 1:", args1)
            print("Values < 0:", args_negative)
            print("NaN values:", args_nan)
            print("Inf values:", args_inf)
            print(pvh)
            plt.imshow(v[0].reshape(100, 100))
            plt.show()

            for i in range(self.nh):
                plt.imshow(self.w.detach().cpu().numpy()[:, i].reshape(100, 100))
                plt.title(f"i = {i}")
                plt.show()


            plt.plot(range(len(dw_norm_list)), dw_norm_list)
            plt.xlabel("epoch_i*batch_j")
            plt.ylabel("L2 norm of w.grad")
            plt.show()
            exit()
        else:
            v = (pvh >= torch.rand(size=(pvh.shape[0], pvh.shape[1])).to(self.device)).float()
            return v

    def sample_h(self, v, dw_norm_list):
        
        if self.basis == "11":
            phv = torch.sigmoid(2 * torch.matmul(v, self.w) + 2 * self.b)
            h = self._bernoulli(phv, v, dw_norm_list)
            h[h == 0] = -1
        else:
            phv = torch.sigmoid(torch.matmul(v, self.w) + self.b)
            h = self._bernoulli(phv, v, dw_norm_list)
        
        return h, phv
    
    def sample_v(self, h, dw_norm_list):
        
        if self.basis == "11":
            pvh = torch.sigmoid(2 * torch.matmul(h, self.w.t()) + 2 * self.a)
            v = self._bernoulli(pvh, h, dw_norm_list)
            v[v == 0] = -1

        else:
            pvh = torch.sigmoid(torch.matmul(h, self.w.t()) + self.a)
            v = self._bernoulli(pvh, h, dw_norm_list)

        return v, pvh
    
    def forward(self, v, k, dw_norm_list):
        h, phv = self.sample_h(v, dw_norm_list)
        for i in range(k):
            v, pvh = self.sample_v(phv, dw_norm_list)
            h, phv = self.sample_h(v, dw_norm_list)
        v, pvh = self.sample_v(phv, dw_norm_list)
        return v
    
    def free_energy(self, v):
        
        vt = torch.matmul(v, self.a)

        if self.basis == "11":
            exp_term = torch.matmul(v, self.w) - self.b

            # NOTE: We seperate the cases where "exp_term" is large and small. When this term is large log(e^x + e^-x) is simply
            # x since e^-x is close to zero. This prevents any overflows from happening. For instance when x=100 we have 
            # log(e^100 + e^-100) ~ log(e^100) ~ 100 but computing the exponential (e^100~10^43) will give overflow errors.
            exp_term_small = exp_term.clone()
            exp_term_large = exp_term.clone()
            mask_small = torch.logical_and(exp_term > -10, exp_term < 10)
            exp_term_small[~mask_small] = torch.nan
            exp_term_large[mask_small] = 0

            ht_small = torch.log(torch.exp(-exp_term_small) + torch.exp(exp_term_small))
            ht_small[ht_small.isnan()] = 0
            ht = torch.sum(exp_term_large + ht_small, dim=1)

            if torch.any(ht.isinf()) or torch.any(ht.isnan()):
                for i in range(self.nh):
                    plt.imshow(self.w.cpu().detach().numpy()[:, i].reshape(100, 100))
                    plt.title(f"Weight of receptive field {i}")
                    plt.show()
                print(f"w: {self.w}")
                print(f"b: {self.b}")
                print(f"exp term small: {exp_term_small}")
                print(f"exp term large: {exp_term_large}")
                print(f"ht_small: {ht_small}")
                print(f"ht: {ht}")

        else:
            ht = torch.sum(torch.log(1 + torch.exp(torch.matmul(v, self.w) + self.b)), dim = 1)
        return -(vt + ht)
    
    def torch_to_w(self):
        a = (self.a).cpu().detach().numpy()
        b = (self.b).cpu().detach().numpy()
        w = (self.w).cpu().detach().numpy()
        
        W = list(a) + list(b) + list(w.reshape(self.nv * self.nh))
        
        return np.array(W)
    
    def _monitor_learning(self, v_test, device):
        if self.monitor_overfitting == True:
            # Need to have a validation set to monitor overfitting
            if self.v_validation.shape[0] == 0:
                print("No validation dataset given. Validation dataset is stored at RBM.v_validation.")
                return
            
            # Compute free energy of train and validation set
            f_train = self.free_energy(v_test.to(device))
            f_valid = self.free_energy(self.v_validation.to(device))

            # Difference between free energies gives an indication of the overfitting
            f_diff = f_valid - f_train

            # Store in RBMs properties
            self.f_trains.append(torch.mean(f_train).cpu().detach().numpy())
            self.f_valids.append(torch.mean(f_valid).cpu().detach().numpy())
            self.f_diffs.append(torch.mean(f_diff).cpu().detach().numpy())

    def _set_visible_biases(self, train_loader, device):
        """
        It is often helpful to set the initial values for the biases to log(p_i / (1-p_i)),
        where p_i is the fraction of datapoints in which v_i = 1. Otherwise most of early
        learning will be spent on trying to set the probability of visible unit i being on to p_i.
        """
        dataset = torch.clone(train_loader.dataset)

        # Set all non-one values to zero (otherwise it does not work for 11 basis)
        dataset[torch.where(dataset != 1)] = 0

        # P(v_i = 1) in the dataset
        p_i = torch.mean(dataset, dim=0)

        # Set visible biases
        self.a = torch.nn.Parameter(p_i)

        # Send to the correct device again
        self = self.to(device)

    def _eigenvalue_spectra(self, train_loader, k, device):
        """
        Compute the eigenvalue spectra of the original dataset, the hidden activations given the 
        dataset and the reconstruction spectra for both the visible and hidden sites.

        Parameters:
            train_loader - training data. Either stored as an torch.tensor, numpy array or DataLoader

        Return: 
            h_spectrum - eigenvalue spectra of the hidden units given the data
            v_recon_spectrum - eigenvalue spectra of the reconstructed visible units
            h_recon_spectrum - eigenvalue spectra of the reconstructed hidden units
        """
        if type(train_loader) == torch.utils.data.dataloader.DataLoader:
            dataset = train_loader.dataset.to(device)
        else:
            dataset = train_loader.to(device)

        # Largest axis is assumed to be N_samples. Dataset.shape = (N_samples, nv)
        if dataset.shape[0] < dataset.shape[1]:
            dataset = dataset.t()

        ### Compute a bunch of eigenvalue spectra ###

        # h data
        _, phv = self.sample_h(dataset)
        h_spectrum = self._eigenvalue_spectrum(phv.t())

        # v recon
        v_recon = self.forward(dataset, k) #NOTE: Perhaps I should not sample here
        v_recon_spectrum = self._eigenvalue_spectrum(v_recon.t()) 

        # h recon
        _, phv_recon = self.sample_h(v_recon)
        h_recon_spectrum = self._eigenvalue_spectrum(phv_recon.t())

        # Return spectra
        return h_spectrum, v_recon_spectrum, h_recon_spectrum

    def _eigenvalue_spectrum(self, x):

        try: 
            # Compute correlation
            corr =  torch.corrcoef(x)
            corr = corr + (torch.randn(corr.shape) * 1e-6).to(self.device)

            # Eigen decomposition
            # print(f"condition number: {torch.linalg.cond(corr)}")
            eigvals, _ = torch.linalg.eigh(corr)

            # Check for degeneracy
            # is_degenerate = torch.sum(torch.isclose(eigvals[:, None], eigvals))
            # print("Are there repeated eigenvalues?", (is_degenerate - eigvals.shape[0]) / 2)

        except torch.linalg.LinAlgError as e:
            # Add some noise to the dataset
            x = x + (torch.randn(x.shape) * 1e-6).to(self.device)

            # Compute correlation
            corr =  torch.corrcoef(x)
            corr = corr + (torch.randn(corr.shape) * 1e-6).to(self.device)

            # Eigen decomposition
            eigvals, _ = torch.linalg.eigh(corr)

            # Check for degeneracy
            # is_degenerate = torch.sum(torch.isclose(eigvals[:, None], eigvals))
            # print("Are there repeated eigenvalues?", (is_degenerate - eigvals.shape[0]) / 2)

            # Notify user on the error
            print(f"An error occurred: {e}")

        return eigvals

    def train(self, train_loader, optimizer, k, epochs, device, n_logs=20):
        # Prevent division by zero later
        if n_logs > epochs:
            n_logs = epochs

        self.device = device

        ### Things to keep track of ###

        # Cost and reconstruction error
        cost_list = np.empty(epochs)
        recon_list = np.empty(epochs)

        # Weight updates
        da_norm_list = []
        db_norm_list = []
        dw_norm_list = []

        # Weights
        a_norm_list = []
        b_norm_list = []
        w_norm_list = []

        # Eigenvalue spectra
        self.h_spectra = np.empty(shape=(epochs, self.nh))
        self.v_recon_spectra = np.empty(shape=(epochs, self.nv))
        self.h_recon_spectra = np.empty(shape=(epochs, self.nh))

        # Keep track of the weights after each epoch
        self.W_epochs = np.zeros(shape=(epochs, self.nv * self.nh + self.nv + self.nh))

        for epoch in range(epochs):
            # Inits
            epoch_cost = 0.
            error_epoch = 0.
            da_epoch = 0.
            db_epoch = 0.
            dw_epoch = 0.

            # Check P(h=1|batch)
            check_batch = 1

            # Store eigenvalue spectra
            if self.track_spectra == True:
                spectra = self._eigenvalue_spectra(train_loader, k, device)
                self.h_spectra[epoch] = spectra[0].cpu().detach().numpy()
                self.v_recon_spectra[epoch] = spectra[1].cpu().detach().numpy()
                self.h_recon_spectra[epoch] = spectra[2].cpu().detach().numpy()

            for i, batch in enumerate(train_loader):
                # Check whether the data contains labels
                if isinstance(batch, list):
                    data = batch[0]
                else:
                    data = batch
                
                batch = data.view(-1, self.nv).to(device)
            
                # Constrastive Divergence
                v = self.forward(batch, k, dw_norm_list).to(device)
                cost = torch.mean(self.free_energy(batch)) - torch.mean(self.free_energy(v))
                cost = cost.to(device)

                # Reconstruction error
                error = torch.sum((batch - v) ** 2)
                
                # Gradient step
                epoch_cost += cost.item()
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Add batch gradient to epoch gradient
                da_norm_list.append(np.linalg.norm(self.a.grad.cpu().detach().numpy()))
                db_norm_list.append(np.linalg.norm(self.b.grad.cpu().detach().numpy()))
                dw_norm_list.append(np.linalg.norm(self.w.grad.cpu().detach().numpy()))

                # Store weights
                a_norm_list.append(np.linalg.norm(self.a.cpu().detach().numpy()))
                b_norm_list.append(np.linalg.norm(self.b.cpu().detach().numpy()))
                w_norm_list.append(np.linalg.norm(self.w.cpu().detach().numpy()))

                # for i in range(self.nh):
                #     print(np.linalg.norm(self.w.cpu().detach().numpy()[:, i]), end=", ")
                #     print(np.linalg.norm(self.w.grad.cpu().detach().numpy()[:, i]), end=", ")
                
                # print(f"b: {np.linalg.norm(self.b.cpu().detach().numpy())}", end=", ")
                # print(f"b.grad: {np.linalg.norm(self.b.grad.cpu().detach().numpy())}", end=", ")
                # print(f"a: {np.linalg.norm(self.a.cpu().detach().numpy())}", end=", ")
                # print(f"a.grad: {np.linalg.norm(self.a.grad.cpu().detach().numpy())}", end=", ")
                # print()

                # Store the weights after 10 batches. This was done to see how the receptive fields
                # change during early learning
                # if epoch == 0 and (i+1) % 10 == 0:
                #     # Store weights
                #     self.W_epochs.append(self.torch_to_w())

                # Add batch error to epoch error
                error_epoch = error.item() + error_epoch

                # Check P(h=1|batch)
                if check_batch == 0:
                    _, ph1v = self.sample_h(batch)
                    x = range(ph1v.shape[1])

                    for i, p in enumerate(ph1v):
                        p = p.cpu().detach().numpy()
                        pmin = min(p)
                        pmax = max(p)
                        im = plt.imshow(p.reshape(8, 8), cmap='gray', vmin=pmin, vmax=pmax)
                        plt.colorbar(im)
                        plt.ylabel("P(h=1|batch)")
                        plt.title(f"{i}-th datapoint in batch")
                        plt.show()

                    # Only check a single batch per epoch
                    check_batch = 1

            # Store cost and recon
            cost_list[epoch] = epoch_cost
            recon_list[epoch] = error_epoch

            # Store weights
            self.W_epochs[epoch] = self.torch_to_w()

            # Print progress
            if (epoch+1) % (round(epochs//n_logs)) == 0:
                print('Epoch [{}/{}], cost: {:.4f}'.format(epoch+1, epochs, epoch_cost))

            # Monitor learning
            self._monitor_learning(train_loader.dataset[:1000], device)

        # Combine
        update_norms = (da_norm_list, db_norm_list, dw_norm_list)
        weight_norms = (a_norm_list, b_norm_list, w_norm_list)

        # Store in models attributes
        self.updates = update_norms
        self.weights = weight_norms
        self.costs = cost_list
        self.reconstructions = recon_list
    