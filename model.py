import torch
from torch import nn

"""
Defaut VAE model with 2 hidden layers of 256 neurons
"""
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim = 256, latent_dim = 32):
        super(VAE, self).__init__()
        #encoder
        self.img_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)
        
        #decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_img = nn.Linear(hidden_dim, input_dim)

        #relu activation
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        #sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        hidden = self.relu(self.img_to_hidden(x))
        hidden = self.relu(self.hidden_to_hidden(hidden))
        mu = self.hidden_to_mu(hidden)
        log_var = self.hidden_to_logvar(hidden)
        return mu, log_var
    
    def decoder(self, z):
        hidden = self.relu(self.latent_to_hidden(z))
        hidden = self.relu(self.hidden_to_hidden(hidden))
        # use sigmoid activation to squash the output pixels to [0, 1]
        img = self.sigmoid(self.hidden_to_img(hidden))
        return img
    
    def reparametrization(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(sigma)
        return mu + sigma*epsilon

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z_reparam = self.reparametrization(mu, log_var)

        x_recon = self.decoder(z_reparam)
        return x_recon, mu, log_var


"""
VAE model supporting expanding method
"""
class VAE_expanding(nn.Module):
    def __init__(self, input_size, device):
        super(VAE_expanding, self).__init__()

        # Check if the input size is a tuple of 2 elements
        assert isinstance(input_size, tuple) and len(input_size) == 2
        self.input_size = input_size

        self.device = device
        
        #Initialize 2 parts of the model: encoder and decoder
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        #Define activation function:
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    """
    Function construct() to build the model
    
    Inputs: - input_size: size of the input image
            - encoder_config: list of hidden layers in the encoder (including the latent code layer)
            - decoder_config: list of hidden layers in the decoder - to be verified if needed or not, can be inversed of encoder_config
            - bool_convolution: if True, the model will be built with convolutional layers, otherwise, fully connected layers
    """
    def construct(self, encoder_config=[], decoder_config=[], bool_convolution = False):
        if bool_convolution == False:
            assert len(encoder_config) != 0 and len(decoder_config) != 0   

            #Build the encoder
            encoderLayers = []
            in_features = self.input_size[0]*self.input_size[1]

            # Add hidden layers to the encoder part and adding the ReLU activation function
            for iLayer in range(0,len(encoder_config)):
                
                if iLayer < len(decoder_config)-1:
                    out_features = encoder_config[iLayer]

                    # Add a linear fully connected layer and initialize the weights with He initialization
                    fc = nn.Linear(in_features, out_features, bias=True)
                    nn.init.kaiming_normal_(fc.weight, nonlinearity='relu')
                    encoderLayers.append(fc)

                    # Add the ReLU activation function
                    encoderLayers.append(self.relu)
                    in_features = out_features
                
                # At the last layer of the encoder, we need to output the mean and logvar of the latent code
                else:
                    out_features = encoder_config[iLayer]

                    '''
                    For the 2 last layers (latent space), it is useful to assign the transformation to the class attributes
                    for future extraction
                    '''
                    self.hidden_to_mu = nn.Linear(in_features, out_features, bias=True)
                    self.hidden_to_logvar = nn.Linear(in_features, out_features, bias=True)

                    # Initialize the weights with He initialization
                    nn.init.kaiming_normal_(self.hidden_to_mu.weight, nonlinearity='relu')
                    nn.init.kaiming_normal_(self.hidden_to_logvar.weight, nonlinearity='relu')
                    
                    '''
                    # !!! PROBLEM: the following code will add the last 2 layers to the encoder, but it will not be able to extract the mean and logvar since its belongs to a Sequential
                    # Adding the last 2 layers to the encoder
                    encoderLayers.append(self.hidden_to_mu)
                    encoderLayers.append(self.hidden_to_logvar)

                    encoderLayers.append(self.relu)'''

                    in_features = out_features
                
            self.encoder = nn.Sequential(*encoderLayers).to(device=self.device)

            decoderLayers = []
            in_features = encoder_config[-1]

            for iLayer in range(0,len(decoder_config)):
                out_features = decoder_config[iLayer]

                # Add a linear fully connected layer and initialize the weights with He initialization
                fc = nn.Linear(in_features, out_features, bias=True)
                nn.init.kaiming_normal_(fc.weight, nonlinearity='relu')
                decoderLayers.append(fc)

                # Add the activation function: ReLU for hidden layers and Sigmoid for the output layer
                if iLayer == len(decoder_config) - 1:
                    decoderLayers.append(self.sigmoid)
                else:
                    decoderLayers.append(self.relu)
                in_features = out_features

            self.decoder = nn.Sequential(*decoderLayers).to(device=self.device)
        else:
            print("Not implemented yet")

    def reparametrization(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(sigma).to(self.device)
        return mu + sigma*epsilon

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        #print("start encode")
        # Pass the images into the encoder part (not include the latent layer)
        encoder_output = self.encoder(x)
        encoder_output = encoder_output.to(self.device)
        #print("done 1st encode")

        # Extract the mean and logvar of the latent code
        self.hidden_to_mu = self.hidden_to_mu.to(self.device)
        self.hidden_to_logvar = self.hidden_to_logvar.to(self.device)
        mu = self.relu(self.hidden_to_mu(encoder_output))
        log_var = self.relu(self.hidden_to_logvar(encoder_output))
        #print("done 2nd encode")

        # Reparametrization trick
        z = self.reparametrization(mu, log_var)
        
        # Pass the latent code into the decoder part
        x_recon = self.decoder(z)

        return x_recon, mu, log_var
    '''
    Function expand_layer to add more neurons to a defined layer
    '''
    def expand_layer(self, layer_index, nb_neuron_increase):
        layer = self.encoder[layer_index]
        old_weight = layer.weight
        old_bias = layer.bias.data if layer.bias is not None else None
        old_out_features = layer.out_features
        new_out_features = old_out_features + nb_neuron_increase

        # Create a new layer with the new number of neurons
        new_layer = nn.Linear(layer.in_features, new_out_features, bias=(old_bias is not None)).to(self.device)
        
        # Copy the old weights and biases to the new layer
        with torch.no_grad():
            new_layer.weight[:old_out_features] = old_weight
            if old_out_features < new_out_features:
                nn.init.kaiming_normal_(new_layer.weight[old_out_features:], mode='fan_in', nonlinearity='relu')
        
        if old_bias is not None:
            with torch.no_grad():
                nn.init.zeros_(new_layer.bias)
                new_layer.bias[:old_out_features] = old_bias

        # Replace the current layer with the new created layer
        self.encoder[layer_index] = new_layer.to(self.device)

        # Adjust the subsequent layer's in_features and weight matrix if it exists
        if layer_index + 2 < len(self.encoder) and isinstance(self.encoder[layer_index + 2], nn.Linear):
            old_next_layer = self.encoder[layer_index + 2]
            new_next_layer = nn.Linear(new_out_features, old_next_layer.out_features, bias=(old_next_layer.bias is not None)).to(self.device)
            with torch.no_grad():
                new_next_layer.weight[:, :old_out_features] = old_next_layer.weight.data[:, :old_out_features]
                if old_out_features < new_out_features:
                    nn.init.zeros_(new_next_layer.weight[:, old_out_features:])

            self.encoder[layer_index + 2] = new_next_layer.to(self.device)
            
            print('Layer ', layer_index, ': ', self.encoder[layer_index], 'Layer ', layer_index + 2, ': ', self.encoder[layer_index + 2])
        
        # Adjust the latent layer's in_features and weight matrix if it exists
        elif layer_index + 2 >= len(self.encoder):
            new_hidden_to_mu = nn.Linear(new_out_features, self.hidden_to_mu.out_features, bias=True).to(self.device)
            new_hidden_to_logvar = nn.Linear(new_out_features, self.hidden_to_logvar.out_features, bias=True).to(self.device)
            with torch.no_grad():
                new_hidden_to_mu.weight[:, :old_out_features] = self.hidden_to_mu.weight.data[:, :old_out_features]
                new_hidden_to_logvar.weight[:, :old_out_features] = self.hidden_to_logvar.weight.data[:, :old_out_features]
                if old_out_features < new_out_features:
                    nn.init.zeros_(new_hidden_to_mu.weight[:, old_out_features:])
                    nn.init.zeros_(new_hidden_to_logvar.weight[:, old_out_features:])
                
            self.hidden_to_mu = new_hidden_to_mu.to(self.device)
            self.hidden_to_logvar = new_hidden_to_logvar.to(self.device)

            print('Layer ', layer_index, ': ', self.encoder[layer_index], 'Layer mu&logvar: ', self.hidden_to_mu)


if __name__ == "__main__":

    encoder_config = [256, 128, 32]
    decoder_config = [128, 256, 784]
    model = VAE_expanding((28, 28), "cpu")
    model.construct(encoder_config, decoder_config, False)

    # Expand the first hidden layer in the encoder by 64 neurons
    model.expand_layer(0, 64)

    # Print the structure of the encoder and decoder
    print("Encoder structure:")
    print(model.encoder)

    print("\nDecoder structure:")
    print(model.decoder)

    x = torch.randn(64, 784)
    x_recon, mu, log_var = model(x)

    print(x.shape)
    print(x_recon.shape)
    print(mu.shape)
    print(log_var.shape)
