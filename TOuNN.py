import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
import matplotlib.pyplot as plt
from matplotlib import colors
from FE_Solver import JAXSolver
from network import TopNet
from projections import applyFourierMap, applySymmetry, applyRotationalSymmetry, applyExtrusion
# from jax.experimental import optimizers
from jax.example_libraries import optimizers
from materialCoeffs2 import microStrs
import pickle

# Get arguments
import configparser
configFile = './config.txt'                 # Modified configuration file
config_plot = configparser.ConfigParser()
config_plot.read(configFile)

meshConfig = config_plot['MESH']
nelx_plot = meshConfig.getint('nelx')            # Number of FE elements along X
nely_plot = meshConfig.getint('nely')            # Number of FE elements along Y

saveConfig = config_plot['SAVE']
save_optimized = saveConfig.getboolean('optimized')
save_loss_history = saveConfig.getboolean('loss_history')
path_str = saveConfig.get("folder")

class TOuNN:
    def __init__(self, exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion):
        self.exampleName = exampleName
        self.FE = JAXSolver(mesh, material)
        self.xy = self.FE.mesh.elemCenters
        self.fourierMap = fourierMap

        # Update input dimension based on Fourier mapping
        if fourierMap['isOn']:
            nnSettings['inputDim'] = 2 * fourierMap['numTerms']
        else:
            nnSettings['inputDim'] = self.FE.mesh.ndim

        # Initialize the neural network
        self.topNet = TopNet(nnSettings)
        self.symMap = symMap
        self.mstrData = microStrs
        self.rotationalSymmetry = rotationalSymmetry
        self.extrusion = extrusion

    def optimizeDesign(self, optParams, savedNet):
        convgHistory = {'epoch': [], 'vf': [], 'J': []}
        xyS = applySymmetry(self.xy, self.symMap)
        xyE = applyExtrusion(xyS, self.extrusion)
        xyR = applyRotationalSymmetry(xyE, self.rotationalSymmetry)

        if self.fourierMap['isOn']:
            xyF = applyFourierMap(xyR, self.fourierMap)
        else:
            xyF = xyR
        penal = 1

        # C Matrix
        components = ['00', '11', '22', '01', '02', '12']

        def getCfromCPolynomial(vfracPow, mstr):
            C = {}
            for c in components:
                C[c] = jnp.zeros(self.FE.mesh.numElems)
            for c in components:
                for pw in range(mstr['order'] + 1):
                    C[c] = C[c].at[:].add(mstr[c][str(pw)] * vfracPow[str(pw)])
            return C  # {dict with 6 keys each of size numElems}

        def getCfromEigenPolynomial(vfracPow, mstr):
            lmda = jnp.zeros((self.FE.mesh.numElems, 3, 3))
            for lmdIdx in range(3):  # 3 eigen values
                lamStr = 'lambda' + str(lmdIdx + 1)
                for pw in range(mstr['order'] + 1):
                    lmda = lmda.at[:, lmdIdx, lmdIdx].add(mstr[lamStr][str(pw)] * vfracPow[str(pw)])
            VL = jnp.einsum('ij,ejk->eik', mstr['eVec'], lmda)
            VLVt = jnp.einsum('eij,jk->eik', VL, mstr['eVec'].T)

            C = {'00': VLVt[:, 0, 0], '11': VLVt[:, 1, 1], '22': VLVt[:, 2, 2],
                 '01': VLVt[:, 0, 1], '02': VLVt[:, 0, 2], '12': VLVt[:, 1, 2]}
            return C

        @jit
        def getCMatrix(mstrType, nn_rho):
            vfracPow = {}  # compute the powers once to avoid repeated calc
            for pw in range(self.mstrData['square']['order'] + 1):  # TODO: use the max order of all mstrs
                vfracPow[str(pw)] = nn_rho ** pw
            C = {}
            for c in components:
                C[c] = jnp.zeros(self.FE.mesh.numElems)

            for mstrCtr, mstr in enumerate(self.mstrData):  # mstrsEig # mstrs
                if self.mstrData[mstr]['type'] == 'eig':
                    Cmstr = getCfromEigenPolynomial(vfracPow, self.mstrData[mstr])
                else:
                    Cmstr = getCfromCPolynomial(vfracPow, self.mstrData[mstr])
                mstrPenal = mstrType[:, mstrCtr] ** penal
                for c in components:
                    C[c] = C[c].at[:].add(jnp.einsum('i,i->i', mstrPenal, Cmstr[c]))

            return C

        # Optimizer
        opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
        opt_state = opt_init(self.topNet.wts)
        opt_update = jit(opt_update)

        if savedNet['isAvailable']:
            saved_params = pickle.load(open(savedNet['file'], "rb"))
            opt_state = optimizers.pack_optimizer_state(saved_params)

        # Foward once to get J0-scaling param
        mstrType, density0 = self.topNet.forward(get_params(opt_state), xyF)
        C = getCMatrix(mstrType, 0.01 + density0)
        J0 = self.FE.objectiveHandle(C)

        # Jitting this causes undefined behavior
        def computeLoss(objective, constraints):
            if optParams['lossMethod']['type'] == 'penalty':
                alpha = optParams['lossMethod']['alpha0'] + \
                        self.epoch * optParams['lossMethod']['delAlpha']  # penalty method
                loss = objective
                for c in constraints:
                    loss += alpha * c ** 2
            if optParams['lossMethod']['type'] == 'logBarrier':
                t = optParams['lossMethod']['t0'] * \
                    optParams['lossMethod']['mu'] ** self.epoch
                loss = objective
                for c in constraints:
                    if c < (-1 / t ** 2):
                        psi = -jnp.log(-c) / t
                    else:
                        psi = t * c - jnp.log(1 / t ** 2) / t + 1 / t
                    loss += psi
            return loss

        # Closure function - jitting this causes undefined behavior
        def closure(nnwts):
            mstrType, density = self.topNet.forward(nnwts, xyF)
            volCons = (jnp.mean(density) / optParams['desiredVolumeFraction']) - 1.
            C = getCMatrix(mstrType, 0.01 + density)
            J = self.FE.objectiveHandle(C)
            return computeLoss(J / J0, [volCons])

        # Optimization loop
        for self.epoch in range(optParams['maxEpochs']):
            penal = min(8.0, 1. + self.epoch * 0.02)
            opt_state = opt_update(self.epoch,
                                   optimizers.clip_grads(jax.grad(closure)(get_params(opt_state)), 1.),
                                   opt_state)

            if self.epoch % 1 == 0:
                convgHistory['epoch'].append(self.epoch)
                mstrType, density = self.topNet.forward(get_params(opt_state), xyF)
                C = getCMatrix(mstrType, 0.01 + density)  # getCfromCPolynomial
                J = self.FE.objectiveHandle(C)
                convgHistory['J'].append(J)
                volf = jnp.mean(density)
                convgHistory['vf'].append(volf)

                if self.epoch == 10:
                    J0 = J

                if self.epoch % 10 == 0:
                    status = 'epoch: {:d}, J: {:.2E}, vf: {:.2F}'.format(self.epoch, J, volf)
                    print(status)

                if self.epoch % 10 == 0:
                    self.FE.mesh.plotFieldOnMesh(density, status)

        if savedNet['isDump']:
            trained_params = optimizers.unpack_optimizer_state(opt_state)
            pickle.dump(trained_params, open(savedNet['file'], "wb"))

        # Save the convergence history to a .txt file
        if save_loss_history:
            with open(path_str + "/convergence_history.txt", "w") as f:
                f.write("epoch\tJ\tvf\n")  # header (optional)
                for e, J, vf in zip(convgHistory["epoch"], convgHistory["J"], convgHistory["vf"]):
                    f.write(f"{e}\t{J}\t{vf}\n")

        return convgHistory

    def plotCompositeTopology(self, res):
        xy = self.FE.mesh.generatePoints(res)
        xyS = applySymmetry(xy, self.symMap)
        xyE = applyExtrusion(xyS, self.extrusion)
        xyR = applyRotationalSymmetry(xyE, self.rotationalSymmetry)

        if self.fourierMap['isOn']:
            xyF = applyFourierMap(xyR, self.fourierMap)
        else:
            xyF = xyR
        mstrType, density = self.topNet.forward(self.topNet.wts, xyF)

        # RGB colors to fill the elements
        fillColors = ['white', (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (0, 1, 1),
                      (1, 0, 1), (0.5, 0, 0.5), (1, 0.55, 0), (0, 0.5, 0.5), (0, 0, 0.5), (0, 0.5, 0),
                      (0.5, 0, 0), (0.5, 0.5, 0)]

        # Load in the microstructures images
        # It contains all microstructures images with the thickness varying from 0-1 in steps of 0.01
        # Shape: [microstructure (3), thickness (100), image x, image y]
        microstrImages = np.load('./microStrImages.npy')

        # Number of elements in X
        NX = res * int(np.ceil((self.FE.mesh.bb['xmax'] -
                                self.FE.mesh.bb['xmin']) / self.FE.mesh.elemSize[0]))

        # Number of elements in Y
        NY = res * int(np.ceil((self.FE.mesh.bb['ymax'] -
                                self.FE.mesh.bb['ymin']) / self.FE.mesh.elemSize[1]))

        # Size of each microstructure image in x,y
        nx, ny = microstrImages.shape[2], microstrImages.shape[3]

        # for loop (over the depth/ number of z elements)
        # for i,_ in enumerate(Z)
        # i
        # Create a Microstructure composite image
        compositeImg = np.zeros((NX * nx, NY * ny))

        # Create a color image
        colorImg = np.zeros((NX, NY))

        # Create a density image
        densityImg = np.zeros((NX, NY))

        maxC = 0
        step = 0.01             # step used when gen mstr images!
        cutOff = 0.98           # val above which its a dark square
        res_density = []        # List to store the density results
        res_microstructure = [] # List to store the chosen microstructure
        position_x = []         # List to store the x element position
        position_y = []         # List to store the y element position

        # Iterate over each element
        for elem in range(xy.shape[0]):
            # X row
            cx = int((res * xy[elem, 0]) / self.FE.mesh.elemSize[0])

            # y row
            cy = int((res * xy[elem, 1]) / self.FE.mesh.elemSize[1])

            # Compute the density for this element
            densityImg[cx, cy] = int(100. * density[elem])

            #print("cx: ", cx, "cy: ", cy, "\n", "Element: ", elem, int(100. * density[elem]))
            #  cx: 0 cy: 0
            #  Element:  0 27

            # Create the image with the microstructures
            # If the density is greater than the threshold of 0.98, the microstructure is considered 100% solid
            if density[elem] > cutOff:
                compositeImg[nx * cx:(cx + 1) * nx, ny * cy:(cy + 1) * ny] = np.ones((nx, ny))      # Microstrure is 100% solid, (50x50 ones (img shape))
                colorImg[cx, cy] = 1

                # If not in the threshold, compute which microstructure with which wall thickness (0-1, percentage)
                # Store element, density and microstructure
                res_density.append(100)
                res_microstructure.append(0)
                position_x.append(cx)
                position_y.append(cy)

            else:
                # Compute which thickness to use (0-100 percentage)
                mstrIdx = min(microstrImages.shape[1] - 1, int(density[elem] // step))

                # Get the microstructure type to use for the current element
                mstrTypeIdx = np.argmax(mstrType[elem, :])

                # Get the microstructure image for the current element
                mstrimg = microstrImages[mstrTypeIdx, mstrIdx, :, :].T

                c = np.argmax(mstrType[elem, :]) + 1

                if c > maxC:
                    maxC = c

                compositeImg[nx * cx:(cx + 1) * nx, ny * cy:(cy + 1) * ny] = mstrimg * c
                colorImg[cx, cy] = c

                # If not in the threshold, compute which microstructure with which wall thickness (0-1, percentage)
                # Store element, density and microstructure
                res_density.append(mstrIdx)
                res_microstructure.append(mstrTypeIdx)
                position_x.append(cx)
                position_y.append(cy)

        # Concatenate lists into a new list where each list is a column
        concatenated_list = list(zip(res_density, res_microstructure, position_x, position_y))

        # Save the concatenated list to a .txt file
        with open(path_str + '/density_unity_cell_x_y.txt', 'w') as file:
            file.write("density\tunit_cell_type\tposition_x\tposition_y\n")  # header (optional)
            for row in concatenated_list:
                file.write('\t'.join(map(str, row)) + '\n')

        plt.figure()
        ax = plt.gca()
        plt.imshow(compositeImg.T, cmap=colors.ListedColormap(fillColors[:maxC + 1]),
                   interpolation='none', vmin=0, vmax=maxC, origin='lower')

        minor_ticks = np.arange(0, nelx_plot * 50 + 1, 50)         # 50,15 elements
        minor_ticks_ = np.arange(0, nely_plot * 50 + 1, 50)        # 50,15 elements

        ax.set_xticks(minor_ticks, minor=False)
        ax.set_yticks(minor_ticks_, minor=False)
        ax.set_xticklabels([str(int(t/100 * 2)) if i % 10 == 0 else "" for i, t in enumerate(minor_ticks)])
        ax.set_yticklabels([str(int(t/100 * 2)) if i % 10 == 0 else "" for i, t in enumerate(minor_ticks_)])
        ax.grid(color='k', linestyle='-', linewidth=1, which="both")
        plt.show()

        if save_optimized:
            plt.savefig(path_str + '/optimized_design.pdf', dpi=300)