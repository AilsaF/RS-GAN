import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable  # needed for define Variable
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import shutil
import os
import time
import matplotlib.ticker as ticker
import matplotlib as mpl
from tempfile import TemporaryFile
outfile = TemporaryFile()
 
plt.switch_backend('agg')

seed_index = 0
torch.manual_seed( seed_index  )
np.random.seed( seed_index )

###################################################################
#########################    Hyper-parameters     #################
###################################################################
DATASET = '1d_cluster'
    # '1dim_uniform' #  '1Gaussian' # '4box'  # ' 8gaussians, 25gaussians, swissroll
numSamples = 100    # default: 100
numDimX = 1   # Model dimensionality
numDimZ = 1 
batchSize = numSamples    # Batch size

discIter = 10   # How many discriminator iterations per generator iteration
genIter = 10  #  default: discIter = 5; genIter = 10
numEpoch = 5000  #  should be multiple of 50 

d_learning_rate = 1e-2  # default 1e-2
g_learning_rate = 1e-2  # default 1e-2
sgd_momentum = 0.9      # default 0.9

disc_width = 10  # width of the discriminator network. Default 10
gen_width = 5  # width of the generator network.  Default 5 

training_method = 'RS'  # 'RS'    # 'JS': JS-GAN original GAN;  'RS':  2: RS-GAN
random_runs_times = 1   # number of random runs of the experiment

visua_train_or_test = 'test'   # 'train' #   # 'train': use train data in the video;  'test': plot test data in the video

# #----------- advanced thoughts to play with -----------
gen_control_init = 0    # 1: control initial weights of generators   else: no control
gen_weight_unif_range = 0.4  # Unif[ -c , c] for generator weights, if gen_control_init == 1
center1 = 0
center2 = 4

#----------- control what to plot -----------
plot_Dfig = 1               # plot the image of D functions over iteration , save lots of png's in a folder 1dclusterresults/
plot_data_and_D_vals = 1    # plot data moving and save in res.png, plot D values and save in 1dim.png. Lots of png's in a folder 1dclusterresults/
plot_figures_loss_distance = 1   # plot D loss, G loss over iterations. Save in one png
print_loss_gap = 2          # print D losses every print_loss_gap iterations (one D loop is one iteration)

#----------- constant for ploting ----------------
xvec = np.arange( center1 - 2 , center2 + 2, 0.1)  # xlim range 
xvec_1d = xvec.astype(np.float32)
xlen = xvec_1d.shape[0]
xvec_2d = np.reshape( xvec_1d, ( xlen, -1)  )
xvec_torch = torch.from_numpy(xvec_2d)

#---------
# plt.rc('legend',fontsize=20) # using a size in points
font_size_def = 20
plt.rc("font", size=font_size_def )

###################################################################
####  Define disciriminator netowrk and generator network 
###################################################################
class Disc(nn.Module):
    def __init__(self, numDimX):
        super(Disc, self).__init__()
        # self.fc1 = nn.Linear(numDimX, 1, bias= True)
        disc_net = nn.Sequential(
            nn.Linear(numDimX, disc_width, bias=True),
            nn.Sigmoid(),
            nn.Linear(disc_width, disc_width, bias=True),
            nn.Sigmoid(),
            nn.Linear(disc_width, 1, bias=True)
        )
        self.disc_net = disc_net

    def forward(self, x, G):
        return self.disc_net(x), self.disc_net(G)

#------------------------------Generater network ------------------------------------
class Gen(nn.Module):
    def __init__(self, numDimZ, numDimX):
        super(Gen, self).__init__()
        gen_net = nn.Sequential(
            nn.Linear(numDimZ, gen_width, bias=True),
            nn.Tanh(),
            nn.Linear(gen_width, gen_width, bias=True),
            nn.Tanh(),
            nn.Linear(gen_width, numDimX, bias=True)
        )
        self.gen_net = gen_net

    def forward(self, z):
        return self.gen_net(z)   

#------------------------------weight initilization------------------------------------
# specify initialization scheme: uniform wegights with bound gen_weight_unif_range
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(- gen_weight_unif_range, gen_weight_unif_range)
        m.bias.data.fill_(0)


###################################################################
####  Define D loss and G loss 
###################################################################
class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
        self.BLL = torch.nn.BCEWithLogitsLoss()   # BLL(x, 1) = log(1 + e^{ x }) 

    def forward(self, logitX, logitG):
        if training_method == 'JS':  # origninal GAN
            return ( self.BLL(logitX, torch.ones_like(logitX)) + self.BLL(logitG, torch.zeros_like(logitG)) )/2
            # logitX = f(x), logitG = f(y), BLL( logitX , 1 ) = log(1 + e^{ - f(x)}), and BLL(logtG, 0) = log(1 + e^{f(y)}), 
            #    Knowledge: BLL( h, 1) = log(1 + exp( - h)), BLL(h, 0) = log(1 + exp(h))
            # Typical GAN fomrulation,  D(x) = 1/(1 + exp( - f(x)) ) = 1 /( 1 + exp( - logitX )), and 1 - D(y) =1 - 1/(1 + exp(- f(y))) = 1/( 1 + exp(f(y)) ).
            # Typical GAN: max log( D(x)) + log( 1 - D(y)) <==> min log(1 + e^{ - f(x)}) + log(1 + e^{f(y)}).
        elif training_method == 'RS':  # RS GAN, no sorting
            return self.BLL( logitX - logitG , torch.ones_like(logitX))  
             # by paper, logitX = f(x), logitG = f(y), BLL( logitX - logitG, 1 ) = log(1 + e^{f(y) - f(x)}), because BLL( h, 1) = log(1 + exp( - h))


#------------------------------G loss -----------------------------------
class GLoss(nn.Module):
    def __init__(self):
        super(GLoss, self).__init__()
        self.BLL = torch.nn.BCEWithLogitsLoss()

    def forward(self, logitX, logitG):
        if training_method == 'JS':  # origninal GAN
            return self.BLL(logitG, torch.ones_like(logitG))  # original GAN
        elif training_method == 'RS': 
            return self.BLL(logitG - logitX, torch.ones_like(logitX))

class WasLoss(nn.Module):  # add square distance
    def __init__(self):
        super(WasLoss, self).__init__()
        self.MSEls = torch.nn.BCEWithLogitsLoss()  # MSELoss()

    def forward(self, true_data, fake_data):
        SLX, _ = torch.sort(true_data, 0)
        SLG, _ = torch.sort(fake_data, 0)
        return self.MSEls(SLG - SLX, torch.ones_like(SLX))  # sorting loss


# could use generator iterator, if use mini-batch new data
###################################################################
# Define the real data distribution
###################################################################

def inf_train_gen():
    real_data = []
    radius = 0.3
    center_vec = [ center1, center2 ]
    for i in range(batchSize):
        point = np.random.rand(1)*radius
        ind = i % 2
        center = center_vec[ind]
        point += center
        real_data.append(point)
    real_data = np.array(real_data, dtype='float32')
    return real_data

class OurDataset(Dataset):
    # Dataset is a class defined in torch; OurDataset is a subclass of Dataset
    def __init__(self, numSamples, numDimX):
        real_data_train = inf_train_gen()  # generate real data
        self.d = torch.Tensor(real_data_train)
        self.numSamples = numSamples
        self.numDimX = numDimX

    def __len__(self):
        return self.numSamples

    def __getitem__(self, idx):
        return self.d[idx, :]

# create the folder to store the pictures
def setDir(filepath):
    '''
    If folder non-exist, then create; if exists, empty it and then create the folder again
    param filepath: file path to be built
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def createDir(filepath):
    '''
    If folder non-exist, then create; if exists, do nothing
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
result_dir = '{}_1dclusterresults_dlr{}_glr{}_diter{}_giter{}_epochnum{}_dwidth{}_gwidth{}_seed{}/'.format(training_method, d_learning_rate, 
    g_learning_rate, discIter, genIter, numEpoch,disc_width, gen_width, seed_index)
setDir(result_dir)                    # store per-iteration information. Need to renew in every run.
createDir(result_dir + 'DLoss_Data_Evolution')    # store D loss figures, and data evolution figures. Keep the folder in different runs. Add new figures to it when adding more runs. 

#########################################################################
###############################    training start now   ###############
########################################################################################

#------------------------------Set up  z samples -----------------------------------
z = 10 * (torch.rand(batchSize, numDimZ) - 0.5)  
z_test = 10 * (torch.rand(batchSize, numDimZ) - 0.5)  # for generating new data

trainData = OurDataset(numSamples, numDimX)
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle= False , num_workers=0, drop_last=True)  # in practice, change to shuffle = True

#------------------------------Start Loop of one random run-----------------------------------
for test_ind in range(random_runs_times):
    ####=========================================================================================
    ####  Setting up basic things: Data, Optimizer, Loss
    ####=========================================================================================
    torch.manual_seed( seed_index )
   
    disc = Disc(numDimX)
    gen = Gen(numDimZ, numDimX)
    if gen_control_init == 1:
        gen.apply(weights_init_uniform)   # set own initial points

    criterionD = DLoss()
    criterionG = GLoss()
    criterionWas = WasLoss()

    dopt = optim.SGD(disc.parameters(), lr=d_learning_rate, momentum=sgd_momentum)

    dopt.zero_grad()
    gopt = optim.SGD(gen.parameters(), lr=g_learning_rate, momentum=sgd_momentum)
    gopt.zero_grad()

    DLossArray = torch.zeros(len(trainLoader)*numEpoch)
    GLossArray = torch.zeros(len(trainLoader)*numEpoch)
    distance_array = torch.zeros(len(trainLoader)*numEpoch)

    epoch_gap = math.ceil( numEpoch/100 )    # Gaps of drawing. Default numEpoch/100, then typically have 100 images. 
    epoch_range_min =  0  
    epoch_range_max =  numEpoch   # default numEpoch

    Loss_record_index = 0   # record where to draw the loss
    gen_data_all = []       # collect all generated data
    start_time = time.time()      # starting timing 


    ####=========================================================================================
    ####  Start real training process
    ####=========================================================================================
    for epoch in range(numEpoch):
        for batch_idx, data in enumerate(trainLoader):            
            ################################################################################
            ##############    draw plots of real and fake data, at current epoch
            ###################################################################
            
            if (epoch % epoch_gap == 0) and ( epoch_range_min <= epoch <= epoch_range_max ) :
                if visua_train_or_test == 'train':
                    xhat = gen(z)
                elif visua_train_or_test == 'test':
                    xhat = gen(z_test)

                #------------------------------Plot real and fake data -----------------------------------
                if plot_data_and_D_vals == 1:
                    if numDimX >= 2:
                        plt.plot(data.detach().numpy()[:, 0], data.detach().numpy()[:, 1], 'bo')
                        plt.plot(xhat.detach().numpy()[:, 0], xhat.detach().numpy()[:, 1], 'rx')
                    else:
                        plt.plot(data.detach().numpy(), [0, ]*batchSize, 'bo')
                        plt.plot(xhat.detach().numpy(), [0, ]*batchSize, 'rx')
                        
                    plt.xlim(-3, 7)
                    epoch_record = epoch/epoch_gap - math.ceil( epoch_range_min/epoch_gap ) 
                    plt.savefig(result_dir +'res_%04d_%04d.png' % (epoch_record, batch_idx))
                    plt.close()

                    #------------------------------Plot output of D values ----------------------------------- 
                    logitX, logitG = disc(data, xhat)   # the output of the discriminator D. 
                    plt.plot(logitX.detach().numpy(), [0, ]*batchSize, 'bx')
                    plt.plot(logitG.detach().numpy(), [0, ]*batchSize, 'ro')
                    plt.xlim(-2, 2)
                    plt.savefig(result_dir +'1dim_proj_%04d_%04d.png' % (epoch_record, batch_idx))
                    plt.close()

            #####----------------------------------------------------------------- 
            ####  Plot the D image over iterations
            #####----------------------------------------------------------------- 

                if plot_Dfig == 1:
                    D_out_torch, D_temp = disc( xvec_torch, xvec_torch )  
                      # we only need to know values of f, but disc is defined to take two inputs, so copy it twice
                    D_output = D_out_torch.detach().numpy()  
                    if training_method == 'JS':  # use JS-GAN
                        D_output = 1 / ( 1 + np.exp( - D_output ) )  # map to [0,1]; ideally 1/2 
                    name_cur = 'D function image at iteration %04d' % epoch 
                    plt.plot(xvec, D_output , '-m' , label=name_cur )
                    plt.title( name_cur )
                    plt.xlim(-3, 7)
                    if training_method == 'JS':
                        plt.ylim(-0.1, 1.1)
                    if training_method == 'RS':
                        plt.ylim(-10, 10)

                    logitG,logitX  = disc( xhat , data  )  
                    gen_value = logitG.detach().numpy()    
                    true_value = logitX.detach().numpy()    
                    if training_method == 'JS':
                        gen_value = 1/(1 + np.exp( - gen_value ))    # map to [0, 1]; ideally 1/(1+exp(f(x))) = 1/2, f(x) = 0
                        true_value = 1/(1 + np.exp( - true_value ))  # map to [0, 1]; ideally 1/2 
                    current_xhat = xhat.detach().numpy()
                    current_data = data.detach().numpy()
                    plt.plot( current_xhat, gen_value  , 'bo', alpha = 0.4, ms = 10, label = 'fake data')   # generated data points
                    plt.plot( current_data, true_value , 'rx', alpha = 0.05, ms = 15, label = 'true data')   # true data points
                    plt.legend()
                    epoch_record = epoch/epoch_gap - math.ceil( epoch_range_min/epoch_gap )
                    plt.savefig(result_dir +'dfig_%04d_%04d.png' % (epoch_record, batch_idx))
                    plt.close()

            xhat = gen(z)    # generate the data by current generator 

            ############################
            # (1) Update D network
            ############################
            dopt.zero_grad()
            for k in range(discIter):
                logitX, logitG = disc(data, xhat.detach())
                loss = criterionD(logitX, logitG)
                loss.backward()
                if epoch % print_loss_gap == 0:
                    if (k+1) % math.ceil(discIter) == 0:
                        DLossArray[ Loss_record_index ] = loss.item() 
                        print("E: %d; B: %d; DLoss: %f" % (epoch, batch_idx, loss.item()))
                dopt.step()
                dopt.zero_grad()

            ############################
            # (2) Update G network
            ############################
            gopt.zero_grad()
            # z = 10*torch.rand(batchSize, numDimZ) - 5
            for k in range(genIter):
                xhat = gen(z)
                logitX, logitG = disc(data, xhat)
                loss = criterionG(logitX, logitG)
                loss.backward()
                if epoch % print_loss_gap == 0:
                    if (k + 1) % math.ceil(genIter) == 0:
                        GLossArray[ Loss_record_index ] = loss.item() 
                        Loss_record_index = Loss_record_index + 1  
                        print("E: %d; B: %d; GLoss: %f" % (epoch, batch_idx, loss.item()))
                        gen_data_all.append( xhat.detach().numpy()  )
                        distance_array[epoch] = criterionWas(xhat, data).item()
                gopt.step()
                gopt.zero_grad()

    ####=========================================================================================
    # Print the number of generated points in each of the two clusters, so as to quickly assess whether success or not \
    if DATASET == '1d_cluster':

        print("center of data are %f and %f  " % ( center1, center2)  )
        xhat = gen(z)
        # xhat = xhat.t()
        count_pos = 0
        count_neg = 0
        countData_pos = 0
        countData_neg = 0
        radius = 0.3
        center1_left = center1 - 1.2 * radius
        center1_right = center1 + 1.2 * radius
        center2_left = center2 - 1.2 * radius
        center2_right = center2 + 1.2 * radius
        for i in range(numSamples):
            if ( xhat[i] > center1_left ) and (xhat[i] < center1_right):
                count_neg += 1
            if (xhat[i] > center2_left) and (xhat[i] < center2_right):
                count_pos += 1
            if (data[i] > center1_left) and (data[i] < center1_right):
                countData_neg += 1
            if (data[i] > center2_left) and (data[i] < center2_right):
                countData_pos += 1
        xhat_trans = xhat.t()
        # print("generated data points are", xhat_trans )
        print("number of points generated in cluster 1 and 2 are respectively %s, %s" % (count_neg, count_pos ))
        print("number of true points in cluster 1 and 2 are respectively %s, %s" % (countData_pos, countData_neg))
 


    end_time = time.time()
    process_time = end_time - start_time
    print("Process time = %f" % process_time)

     ####=========================================================================================
    # Draw one figure of Dloss and Gloss, and one figure of distance
    ###  plot_figures_loss_distance = 0
    if plot_figures_loss_distance == 1:
        ite_vec = np.arange( 0 , Loss_record_index - 1  , 1 )* print_loss_gap
        plt.plot(  ite_vec , DLossArray.numpy()[0: Loss_record_index - 1 ], '-r', label='DLoss', alpha = 0.5 )
        plt.legend()
        plot_x_gap = math.ceil( numEpoch/5 )
        my_x_ticks = np.arange( 0, numEpoch + 1  , plot_x_gap  )
        plt.xticks(my_x_ticks, fontsize= 20 )
        # plt.yticks(fontsize= 20 )
        plt.ylim(0.3, 1.2)
        plt.xlabel(' iteration', fontsize = 20 )
        plt.ylabel(' D loss', fontsize = 20)

        name1 = '_Dloss_Dit%s_Git%s_seed%s_epoch%s_Dw%s_Gw%s_run%s_time%f.pdf' %  ( discIter, genIter , seed_index , numEpoch, disc_width, gen_width, test_ind, process_time )
        fig_name = result_dir +'DLoss_Data_Evolution/' + training_method + name1
        plt.tight_layout()   # resize the figure so that it fits; otherwise the saved figure cuts some texts in the middle
        plt.savefig( fig_name )
        plt.close()

    ####=========================================================================================
    ####################    Draw: How Fake Data Evolve Over Ietartions          #################
    ####=========================================================================================
    ###  plot_gen_data_evolution = 0



    plot_gen_data_evolution = 1
    if plot_gen_data_evolution == 1: 
        fake_all_tensor = np.array ( gen_data_all )   # transfer generated data into numpy form 
        fake_all = fake_all_tensor[:,:,0].T           # transfer generated data into a matrix 
        fake_shape = fake_all.shape
        print(fake_shape)
        data_num = fake_shape[0]
        epoch_record_num = fake_shape[1]
        data_np = data.detach().numpy()       # collect all true data as a vector 
        all_one = np.ones( [ 1, epoch_record_num ] )
        data_matrix = np.matmul(data_np, all_one)       # obtain a matrix, each row is the same true data point
        ite_vec = np.arange ( 0, epoch_record_num , 1 )*print_loss_gap
        Fig_fake = plt.figure(  )
        ax_fake = Fig_fake.add_subplot(1, 1, 1)
        for kk in range( data_num ):
            ax_fake.plot(  ite_vec , fake_all[ kk , : ] , '-b', alpha = 0.5 )
            ax_fake.plot(  ite_vec , data_matrix[ kk , : ] , '-r', alpha = 0.2 )

        plot_x_gap = math.ceil( numEpoch/5 )
        my_x_ticks = np.arange( 0, numEpoch + 1  , plot_x_gap  )
        plt.xticks(my_x_ticks, fontsize= 20 )
        plt.yticks(fontsize= 20 )
        plt.ylim( -2 , 8)  

        plt.xlabel(' iteration' )  #  fontsize = 20
        plt.ylabel(' data position', fontsize = 20)
        name2 = '_FakeData_Dit%s_Git%s_seed%s_epoch%s_Dw%s_Gw%s_run%s_time%f.pdf' %  ( discIter, genIter , seed_index , numEpoch, disc_width, gen_width, test_ind, process_time )
        fig_name = result_dir +'DLoss_Data_Evolution/' + training_method + name2
        plt.tight_layout()   # resize the figure so that it fits; otherwise the saved figure cuts some texts in the middle
        plt.savefig( fig_name )   
        plt.close() 


####=========================================================================================
#  Print hyper-parameters of the current run, to avoid confusion if running multiple experiments
print("discriminator iteration, generator iteration are %s, %s" % (discIter, genIter ) )
print("discriminator lr, generator lr are %f, %f" % ( d_learning_rate, g_learning_rate ) )   
print( "D width, G width are %s, %s  " % ( disc_width , gen_width ) )
if training_method == 'RS':
    print("===========")
    print("RS-GAN training")
    print("===========")
if training_method == 'JS':
    print("------")
    print("JS-GAN training")  
    print("------")


