import numpy as np
import matplotlib.pyplot as plt
from examples import getExampleBC
from Mesher import RectangularGridMesher
from projections import computeFourierMap
from material import Material
from TOuNN import TOuNN
from plotUtil import plotConvergence

import time
import configparser

# Get arguments
configFile = './config.txt'                 # Modified configuration file
config = configparser.ConfigParser()
config.read(configFile)

## Mesh and Boundary Conditions
meshConfig = config['MESH']
ndim = meshConfig.getint('ndim')            # Number of dimensions - default 2
nelx = meshConfig.getint('nelx')            # Number of FE elements along X
nely = meshConfig.getint('nely')            # Number of FE elements along Y
elemSize = np.array(meshConfig['elemSize'].split(',')).astype(float)
exampleName, bcSettings, symMap = getExampleBC(config['PROBLEM'].getint('number'), nelx, nely, elemSize)

## Define grid mesh
mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings)

## Material
materialConfig = config['MATERIAL']
E, nu = materialConfig.getfloat('E'), materialConfig.getfloat('nu')
matProp = {'physics': 'structural', 'Emax': E, 'nu': nu, 'Emin': 1e-3 * E}
material = Material(matProp)

## Neural Network
tounnConfig = config['TOUNN']

# Define NN configurations
nnSettings = {'numLayers': tounnConfig.getint('numLayers'),
              'numNeuronsPerLayer': tounnConfig.getint('hiddenDim'),
              'outputDim': tounnConfig.getint('outputDim')}
  
fourierMap = {'isOn': tounnConfig.getboolean('fourier_isOn'),
              'minRadius': tounnConfig.getfloat('fourier_minRadius'),
              'maxRadius': tounnConfig.getfloat('fourier_maxRadius'),
              'numTerms': tounnConfig.getint('fourier_numTerms')}

fourierMap['map'] = computeFourierMap(mesh, fourierMap)

# Optimization params
lossConfig = config['LOSS']
lossMethod = {'type': 'penalty', 'alpha0': lossConfig.getfloat('alpha0'),
              'delAlpha': lossConfig.getfloat('delAlpha')}

optConfig = config['OPTIMIZATION']
optParams = {'maxEpochs': optConfig.getint('numEpochs'),
             'lossMethod': lossMethod,
             'learningRate': optConfig.getfloat('lr'),
             'desiredVolumeFraction': optConfig.getfloat('desiredVolumeFraction'),
             'gradclip': {'isOn': optConfig.getboolean('gradClip_isOn'),
                         'thresh': optConfig.getfloat('gradClip_clipNorm')}}

## Other projection settings
rotationalSymmetry = {'isOn': False, 'sectorAngleDeg': 90,
      'centerCoordn': np.array([20, 10])}
extrusion = {'X': {'isOn': False, 'delta': 1.},
                      'Y': {'isOn': False, 'delta': 1.}}

## Run optimization
plt.close('all')
savedNet = {'isAvailable': False,
            'file': './netWeights.pkl',
            'isDump': False}

start = time.perf_counter()

# Instantiate the NN
tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion)

# Optimize the design
convgHistory = tounn.optimizeDesign(optParams, savedNet)

# Print calculation time
print(f'Time taken (sec): {time.perf_counter() - start:.2F}')

# Plot the convergence history
plotConvergence(convgHistory)

# Plot the results with the microstructures
tounn.plotCompositeTopology(1)

# Show everything to the screen
plt.show(block=True)
