from collections import OrderedDict
import torch
import torch.nn as nn


def grow_classifier(device, classifier, class_increment, weight_initializer):
    """
    Function to grow the units of a classifier an initializing only the newly added units while retaining old knowledge.

    Parameters:
        device (str): Name of device to use.
        classifier (torch.nn.module): Trained classifier portion of the model.
        class_increment (int): Number of classes/units to add.
        weight_initializer (WeightInit): Weight initializer class instance defining initialization schemes/functions.
    """

    # add the corresponding amount of features and resize the weights
    new_in_features = classifier[-1].in_features
    new_out_features = classifier[-1].out_features + class_increment
    bias_flag = False

    tmp_weights = classifier[-1].weight.data.clone()
    if not isinstance(classifier[-1].bias, type(None)):
        tmp_bias = classifier[-1].bias.data.clone()
        bias_flag = True

    classifier[-1] = nn.Linear(new_in_features, new_out_features, bias=bias_flag)
    classifier[-1].to(device)

    # initialize the correctly shaped layer.
    weight_initializer.layer_init(classifier[-1])

    # copy back the temporarily saved parameters for the slice of previously trained classes.
    classifier[-1].weight.data[0:-class_increment, :] = tmp_weights
    if not isinstance(classifier[-1].bias, type(None)):
        classifier[-1].bias.data[0:-class_increment] = tmp_bias


def get_feat_size(block, spatial_size, ncolors=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.

    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    """

    x = torch.randn(2, ncolors, spatial_size, spatial_size)
    out = block(x)
    num_feat = out.size(1)
    spatial_dim_x = out.size(2)
    spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y


class SingleConvLayer(nn.Module):
    """
    Convenience function defining a single block consisting of a convolution or transposed convolution followed by
    batch normalization and a rectified linear unit activation function.
    """
    def __init__(self, l, fan_in, fan_out, kernel_size=3, padding=1, stride=1, batch_norm=1e-5, is_transposed=False):
        super(SingleConvLayer, self).__init__()

        if is_transposed:
            self.layer = nn.Sequential(OrderedDict([
                ('transposed_conv' + str(l), nn.ConvTranspose2d(fan_in, fan_out, kernel_size=kernel_size,
                                                                padding=padding, stride=stride, bias=False))
            ]))
        else:
            self.layer = nn.Sequential(OrderedDict([
                ('conv' + str(l), nn.Conv2d(fan_in, fan_out, kernel_size=kernel_size, padding=padding, stride=stride,
                                            bias=False))
            ]))
        if batch_norm > 0.0:
            self.layer.add_module('bn' + str(l), nn.BatchNorm2d(fan_out, eps=batch_norm))

        self.layer.add_module('act' + str(l), nn.ReLU())

    def forward(self, x):
        x = self.layer(x)
        return x


class SingleLinearLayer(nn.Module):
    """
    Convenience function defining a single block consisting of a fully connected (linear) layer followed by
    batch normalization and a rectified linear unit activation function.
    """
    def __init__(self, l, fan_in, fan_out, batch_norm=1e-5):
        super(SingleLinearLayer, self).__init__()

        self.fclayer = nn.Sequential(OrderedDict([
            ('fc' + str(l), nn.Linear(fan_in, fan_out, bias=False)),
        ]))

        if batch_norm > 0.0:
            self.fclayer.add_module('bn' + str(l), nn.BatchNorm1d(fan_out, eps=batch_norm))

        self.fclayer.add_module('act' + str(l), nn.ReLU())

    def forward(self, x):
        x = self.fclayer(x)
        return x


class MLP(nn.Module):
    """
    MLP design with two hidden layers and 400 hidden units each in the encoder according to
    ﻿Measuring Catastrophic Forgetting in Neural Networks: https://arxiv.org/abs/1708.02072
    Extended to the variational setting and our unified model.
    """

    def __init__(self, device, num_classes, num_colors, args):
        super(MLP, self).__init__()

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.device = device
        self.out_channels = args.out_channels

        self.seen_tasks = []

        self.num_samples = args.var_samples
        self.latent_dim = args.var_latent_dim

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_layer1', SingleLinearLayer(1, self.num_colors * (self.patch_size ** 2), 400,
                                                 batch_norm=self.batch_norm)),
            ('encoder_layer2', SingleLinearLayer(2, 400, 400, batch_norm=self.batch_norm))
        ]))

        self.latent_mu = nn.Linear(400, self.latent_dim, bias=False)
        self.latent_std = nn.Linear(400, self.latent_dim, bias=False)

        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_layer0', SingleLinearLayer(0, self.latent_dim, 400, batch_norm=self.batch_norm)),
            ('decoder_layer1', SingleLinearLayer(1, 400, 400, batch_norm=self.batch_norm)),
            ('decoder_layer2', nn.Linear(400, self.out_channels * (self.patch_size ** 2), bias=False))
        ]))

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        z_mean = self.latent_mu(x)
        z_std = self.latent_std(x)
        return z_mean, z_std

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        x = self.decoder(z)
        x = x.view(-1, self.out_channels, self.patch_size, self.patch_size)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        x = x.view(-1, self.out_channels, self.patch_size, self.patch_size)
        return x

    def forward(self, x):
        z_mean, z_std = self.encode(x)
        output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
                                     self.patch_size).to(self.device)
        classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
        for i in range(self.num_samples):
            z = self.reparameterize(z_mean, z_std)
            output_samples[i] = self.decode(z)
            classification_samples[i] = self.classifier(z)
        return classification_samples, output_samples, z_mean, z_std

class GM:

    def __init__(self, args):
        self.num_epochs = args.epochs
        self.cuda = args.cuda
        self.verbose = args.verbose

        self.batch_size = args.batch_size
        self.batch_size_val = args.batch_size_val
        self.learning_rate = args.learning_rate
        self.decay_epoch = args.decay_epoch
        self.lr_decay = args.lr_decay
        self.w_cat = args.w_categ
        self.w_gauss = args.w_gauss    
        self.w_rec = args.w_rec
        self.rec_type = args.rec_type 

        self.num_classes = args.num_classes
        self.gaussian_size = args.gaussian_size
        self.input_size = args.input_size

        # gumbel
        self.init_temp = args.init_temp
        self.decay_temp = args.decay_temp
        self.hard_gumbel = args.hard_gumbel
        self.min_temp = args.min_temp
        self.decay_temp_rate = args.decay_temp_rate
        self.gumbel_temp = self.init_temp

        self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
        self.losses = LossFunctions()
        self.metrics = Metrics()

        if self.cuda:
            self.network = self.network.cuda() 
  

    def unlabeled_loss(self, data, out_net):

        # obtain network variables
        z, data_recon = out_net['gaussian'], out_net['x_rec'] 
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']

        # reconstruction loss
        loss_rec = self.losses.reconstruction_loss(data, data_recon, self.rec_type)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # total loss
        loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat

        # obtain predictions
        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = {'total': loss_total, 
                    'predicted_labels': predicted_labels,
                    'reconstruction': loss_rec,
                    'gaussian': loss_gauss,
                    'categorical': loss_cat}
        return loss_dic
    
    
    def train_epoch(self, optimizer, data_loader):

        self.network.train()
        total_loss = 0.
        recon_loss = 0.
        cat_loss = 0.
        gauss_loss = 0.

        accuracy = 0.
        nmi = 0.
        num_batches = 0.

        true_labels_list = []
        predicted_labels_list = []

        # iterate over the dataset
        for (data, labels) in data_loader:
            if self.cuda == 1:
                data = data.cuda()

                optimizer.zero_grad()

                # flatten data
                data = data.view(data.size(0), -1)

                  # forward call
                out_net = self.network(data, self.gumbel_temp, self.hard_gumbel) 
                unlab_loss_dic = self.unlabeled_loss(data, out_net) 
                total = unlab_loss_dic['total']

                  # accumulate values
                total_loss += total.item()
                recon_loss += unlab_loss_dic['reconstruction'].item()
                gauss_loss += unlab_loss_dic['gaussian'].item()
                cat_loss += unlab_loss_dic['categorical'].item()

                  # perform backpropagation
                total.backward()
                optimizer.step()  

                  # save predicted and true labels
                predicted = unlab_loss_dic['predicted_labels']
                true_labels_list.append(labels)
                predicted_labels_list.append(predicted)   

            num_batches += 1. 

        # average per batch
        total_loss /= num_batches
        recon_loss /= num_batches
        gauss_loss /= num_batches
        cat_loss /= num_batches

        # concat all true and predicted labels
        true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

        # compute metrics
        accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
        nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

        return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi


    def test(self, data_loader, return_loss=False):

        self.network.eval()
        total_loss = 0.
        recon_loss = 0.
        cat_loss = 0.
        gauss_loss = 0.

        accuracy = 0.
        nmi = 0.
        num_batches = 0.

        true_labels_list = []
        predicted_labels_list = []

        with torch.no_grad():
            for data, labels in data_loader:
                if self.cuda == 1:
                      data = data.cuda()

            # flatten data
            data = data.view(data.size(0), -1)

            # forward call
            out_net = self.network(data, self.gumbel_temp, self.hard_gumbel) 
            unlab_loss_dic = self.unlabeled_loss(data, out_net)  

            # accumulate values
            total_loss += unlab_loss_dic['total'].item()
            recon_loss += unlab_loss_dic['reconstruction'].item()
            gauss_loss += unlab_loss_dic['gaussian'].item()
            cat_loss += unlab_loss_dic['categorical'].item()

            # save predicted and true labels
            predicted = unlab_loss_dic['predicted_labels']
            true_labels_list.append(labels)
            predicted_labels_list.append(predicted)   

            num_batches += 1. 

        # average per batch
        if return_loss:
            total_loss /= num_batches
            recon_loss /= num_batches
            gauss_loss /= num_batches
            cat_loss /= num_batches

        # concat all true and predicted labels
        true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

        # compute metrics
        accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
        nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

        if return_loss:
            return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi
        else:
            return accuracy, nmi


    def train(self, train_loader, val_loader):

        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        train_history_acc, val_history_acc = [], []
        train_history_nmi, val_history_nmi = [], []

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_rec, train_gauss, train_cat, train_acc, train_nmi = self.train_epoch(optimizer, train_loader)
            val_loss, val_rec, val_gauss, val_cat, val_acc, val_nmi = self.test(val_loader, True)

      # if verbose then print specific information about training
        if self.verbose == 1:
            print("(Epoch %d / %d)" % (epoch, self.num_epochs) )
            print("Train - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
                  (train_rec, train_gauss, train_cat))
            print("Valid - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
                  (val_rec, val_gauss, val_cat))
            print("Accuracy=Train: %.5lf; Val: %.5lf    Total Loss=Train: %.5lf; Val: %.5lf" % \
                  (train_acc, val_acc,  train_loss, val_loss))
        else:
            print('(Epoch %d / %d) Train_Loss: %.3lf; Val_Loss: %.3lf   Train_ACC: %.3lf; Val_ACC: %.3lf ' % \
                  (epoch, self.num_epochs, train_loss, val_loss, train_acc, val_acc))

      # decay gumbel temperature
        if self.decay_temp == 1:
            self.gumbel_temp = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)
        if self.verbose == 1:
            print("Gumbel Temperature: %.3lf" % self.gumbel_temp)

        train_history_acc.append(train_acc)
        val_history_acc.append(val_acc)
        train_history_nmi.append(train_nmi)
        val_history_nmi.append(val_nmi)
        return {'train_history_nmi' : train_history_nmi, 'val_history_nmi': val_history_nmi,
                'train_history_acc': train_history_acc, 'val_history_acc': val_history_acc}
  

    def latent_features(self, data_loader, return_labels=False):

        self.network.eval()
        N = len(data_loader.dataset)
        features = np.zeros((N, self.gaussian_size))
        if return_labels:
            true_labels = np.zeros(N, dtype=np.int64)
        start_ind = 0
        with torch.no_grad():
            for (data, labels) in data_loader:
                if self.cuda == 1:
                    data = data.cuda()
                    # flatten data
                    data = data.view(data.size(0), -1)  
                    out = self.network.inference(data, self.gumbel_temp, self.hard_gumbel)
                    latent_feat = out['mean']
                    end_ind = min(start_ind + data.size(0), N+1)

                # return true labels
                if return_labels:
                    true_labels[start_ind:end_ind] = labels.cpu().numpy()
                features[start_ind:end_ind] = latent_feat.cpu().detach().numpy()  
                start_ind += data.size(0)
        if return_labels:
            return features, true_labels
        return features


    def reconstruct_data(self, data_loader, sample_size=-1):

        self.network.eval()

        # sample random data from loader
        indices = np.random.randint(0, len(data_loader.dataset), size=sample_size)
        test_random_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=sample_size, sampler=SubsetRandomSampler(indices))

        # obtain values
        it = iter(test_random_loader)
        test_batch_data, _ = it.next()
        original = test_batch_data.data.numpy()
        if self.cuda:
            test_batch_data = test_batch_data.cuda()  

        # obtain reconstructed data  
        out = self.network(test_batch_data, self.gumbel_temp, self.hard_gumbel) 
        reconstructed = out['x_rec']
        return original, reconstructed.data.cpu().numpy()


    def plot_latent_space(self, data_loader, save=False):

        # obtain the latent features
        features = self.latent_features(data_loader)

        # plot only the first 2 dimensions
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
                edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
        plt.colorbar()
        if(save):
            fig.savefig('latent_space.png')
        return fig
  
  
    def random_generation(self, num_elements=1):

        # categories for each element
        arr = np.array([])
        for i in range(self.num_classes):
            arr = np.hstack([arr,np.ones(num_elements) * i] )
        indices = arr.astype(int).tolist()

        categorical = F.one_hot(torch.tensor(indices), self.num_classes).float()

        if self.cuda:
            categorical = categorical.cuda()

        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        # generate new samples with the given gaussian
        generated = self.network.generative.pxz(gaussian)

        return generated.cpu().detach().numpy() 
class WRNBasicBlock(nn.Module):
    """
    Convolutional or transposed convolutional block consisting of multiple 3x3 convolutions with short-cuts,
    ReLU activation functions and batch normalization.
    """
    def __init__(self, in_planes, out_planes, stride, batchnorm=1e-5, is_transposed=False):
        super(WRNBasicBlock, self).__init__()

        if is_transposed:
            self.layer1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                             output_padding=int(stride > 1), bias=False)
        else:
            self.layer1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.useShortcut = ((in_planes == out_planes) and (stride == 1))
        if not self.useShortcut:
            if is_transposed:
                self.shortcut = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                                                   output_padding=int(1 and stride == 2), bias=False)
            else:
                self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        if not self.useShortcut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.layer1(out if self.useShortcut else x)))
        out = self.conv2(out)

        return torch.add(x if self.useShortcut else self.shortcut(x), out)


class WRNNetworkBlock(nn.Module):
    """
    Convolutional or transposed convolutional block
    """
    def __init__(self, nb_layers, in_planes, out_planes, block_type, batchnorm=1e-5, stride=1,
                 is_transposed=False):
        super(WRNNetworkBlock, self).__init__()

        if is_transposed:
            self.block = nn.Sequential(OrderedDict([
                ('convT_block' + str(layer + 1), block_type(layer == 0 and in_planes or out_planes, out_planes,
                                                             layer == 0 and stride or 1, batchnorm=batchnorm,
                                                             is_transposed=(layer == 0)))
                for layer in range(nb_layers)
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('conv_block' + str(layer + 1), block_type((layer == 0 and in_planes) or out_planes, out_planes,
                                                           (layer == 0 and stride) or 1, batchnorm=batchnorm))
                for layer in range(nb_layers)
            ]))

    def forward(self, x):
        x = self.block(x)
        return x


class WRN(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting and to our unified model.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(WRN, self).__init__()

        self.widen_factor = args.wrn_widen_factor
        self.depth = args.wrn_depth

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.device = device
        self.out_channels = args.out_channels
        self.double_blocks = args.double_wrn_blocks

        self.seen_tasks = []

        self.num_samples = args.var_samples
        self.latent_dim = args.var_latent_dim

        self.nChannels = [args.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,
                          64 * self.widen_factor, 64 * self.widen_factor, 64 * self.widen_factor,
                          64 * self.widen_factor]

        if self.double_blocks:
            assert ((self.depth - 2) % 12 == 0)
            self.num_block_layers = int((self.depth - 2) / 12)
            self.encoder = nn.Sequential(OrderedDict([
                ('encoder_conv1',
                 nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
                ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block4', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[4],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block5', WRNNetworkBlock(self.num_block_layers, self.nChannels[4], self.nChannels[5],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block6', WRNNetworkBlock(self.num_block_layers, self.nChannels[5], self.nChannels[6],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_bn1', nn.BatchNorm2d(self.nChannels[6], eps=self.batch_norm)),
                ('encoder_act1', nn.ReLU(inplace=True))
            ]))
        else:
            assert ((self.depth - 2) % 6 == 0)
            self.num_block_layers = int((self.depth - 2) / 6)

            self.encoder = nn.Sequential(OrderedDict([
                ('encoder_conv1',
                 nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
                ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm)),
                ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
                ('encoder_act1', nn.ReLU(inplace=True))
            ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,
                                                                                          self.num_colors)
        self.latent_mu = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels,
                                   self.latent_dim, bias=False)
        self.latent_std = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                                    self.latent_dim, bias=False)

        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

        self.latent_decoder = nn.Linear(self.latent_dim, self.enc_spatial_dim_x * self.enc_spatial_dim_y *
                                        self.enc_channels, bias=False)

        if self.double_blocks:
            self.decoder = nn.Sequential(OrderedDict([
                ('decoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[6], self.nChannels[5],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample1', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[5], self.nChannels[4],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample2', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[4], self.nChannels[3],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample3', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block4', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample4', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block5', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample5', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block6', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[0],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_bn1', nn.BatchNorm2d(self.nChannels[0], eps=self.batch_norm)),
                ('decoder_act1', nn.ReLU(inplace=True)),
                ('decoder_upsample6', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_conv1', nn.Conv2d(self.nChannels[0], self.out_channels, kernel_size=3, stride=1, padding=1,
                                            bias=False))
            ]))
        else:
            self.decoder = nn.Sequential(OrderedDict([
                ('decoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample1', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample2', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[0],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_bn1', nn.BatchNorm2d(self.nChannels[0], eps=self.batch_norm)),
                ('decoder_act1', nn.ReLU(inplace=True)),
                ('decoder_conv1', nn.Conv2d(self.nChannels[0], self.out_channels, kernel_size=3, stride=1, padding=1,
                                            bias=False))
            ]))

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z_mean = self.latent_mu(x)
        z_std = self.latent_std(x)
        return z_mean, z_std

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        z = self.latent_decoder(z)
        z = z.view(z.size(0), self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        z_mean, z_std = self.encode(x)
        output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
                                     self.patch_size).to(self.device)
        classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
        for i in range(self.num_samples):
            z = self.reparameterize(z_mean, z_std)
            output_samples[i] = self.decode(z)
            classification_samples[i] = self.classifier(z)
        return classification_samples, output_samples, z_mean, z_std
class Critic(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            channel_size, channel_size*2,
            kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            channel_size*2, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            channel_size*4, channel_size*8,
            kernel_size=4, stride=1, padding=1,
        )
        self.fc = nn.Linear((image_size//8)**2 * channel_size*4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, (self.image_size//8)**2 * self.channel_size*4)
        return self.fc(x)


class Generator(nn.Module):
    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.fc = nn.Linear(z_size, (image_size//8)**2 * channel_size*8)
        self.bn0 = nn.BatchNorm2d(channel_size*8)
        self.bn1 = nn.BatchNorm2d(channel_size*4)
        self.deconv1 = nn.ConvTranspose2d(
            channel_size*8, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channel_size*2)
        self.deconv2 = nn.ConvTranspose2d(
            channel_size*4, channel_size*2,
            kernel_size=4, stride=2, padding=1,
        )
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv3 = nn.ConvTranspose2d(
            channel_size*2, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            channel_size, image_channel_size,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        g = F.relu(self.bn0(self.fc(z).view(
            z.size(0),
            self.channel_size*8,
            self.image_size//8,
            self.image_size//8,
        )))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return F.sigmoid(g)
    
class GenerativeMixin(object):
    """Mixin which defines a sampling iterface for a generative model."""
    def sample(self, size):
        raise NotImplementedError    
    
class Scholar(GenerativeMixin, nn.Module):
    """Scholar for Deep Generative Replay"""
    def __init__(self, label, generator, solver):
        super().__init__()
        self.label = label
        self.generator = generator
        self.solver = solver

    def train_with_replay(
            self, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32,
            generator_iterations=2000,
            generator_training_callbacks=None,
            solver_iterations=1000,
            solver_training_callbacks=None,
            collate_fn=None):
        # scholar and previous datasets cannot be given at the same time.
        mutex_condition_infringed = all([
            scholar is not None,
            bool(previous_datasets)
        ])
        assert not mutex_condition_infringed, (
            'scholar and previous datasets cannot be given at the same time'
        )

        # train the generator of the scholar.
        self._train_batch_trainable_with_replay(
            self.generator, dataset, scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=generator_iterations,
            training_callbacks=generator_training_callbacks,
            collate_fn=collate_fn,
        )

        # train the solver of the scholar.
        self._train_batch_trainable_with_replay(
            self.solver, dataset, scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=solver_iterations,
            training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

    @property
    def name(self):
        return self.label

    def sample(self, size):
        x = self.generator.sample(size)
        y = self.solver.solve(x)
        return x.data, y.data

    def _train_batch_trainable_with_replay(
            self, trainable, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32, iterations=1000,
            training_callbacks=None, collate_fn=None):
        # do not train the model when given non-positive iterations.
        if iterations <= 0:
            return

        # create data loaders.
        data_loader = iter(utils.get_data_loader(
            dataset, batch_size, cuda=self._is_on_cuda(),
            collate_fn=collate_fn,
        ))
        data_loader_previous = iter(utils.get_data_loader(
            ConcatDataset(previous_datasets), batch_size,
            cuda=self._is_on_cuda(), collate_fn=collate_fn,
        )) if previous_datasets else None

        # define a tqdm progress bar.
        progress = tqdm(range(1, iterations+1))

        for batch_index in progress:
            # decide from where to sample the training data.
            from_scholar = scholar is not None
            from_previous_datasets = bool(previous_datasets)
            cuda = self._is_on_cuda()

            # sample the real training data.
            x, y = next(data_loader)
            x = Variable(x).cuda() if cuda else Variable(x)
            y = Variable(y).cuda() if cuda else Variable(y)

            # sample the replayed training data.
            if from_previous_datasets:
                x_, y_ = next(data_loader_previous)
            elif from_scholar:
                x_, y_ = scholar.sample(batch_size)
            else:
                x_ = y_ = None

            if x_ is not None and y_ is not None:
                x_ = Variable(x_).cuda() if cuda else Variable(x_)
                y_ = Variable(y_).cuda() if cuda else Variable(y_)

            # train the model with a batch.
            result = trainable.train_a_batch(
                x, y, x_=x_, y_=y_,
                importance_of_new_task=importance_of_new_task
            )

            # fire the callbacks on each iteration.
            for callback in (training_callbacks or []):
                callback(trainable, progress, batch_index, result)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
