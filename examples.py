import numpy as np

## Examples of pre-defined shapes
def getExampleBC(example, nelx, nely, elemSize):

    ## Tip cantilever
    if example == 1:
        exampleName = 'TipCantilever'
        bcSettings = {'fixedNodes': np.arange(0, 2 * (nely + 1), 1),
                      'forceMagnitude': -1.,
                      'forceNodes': 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1,
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': False, 'midPt': 0.5 * nelx * elemSize[0]}}

    ## Mid-cantilever
    elif example == 2:
        exampleName = 'MidCantilever'
        bcSettings = {'fixedNodes': np.arange(0, 2 * (nely + 1), 1),
                      'forceMagnitude': -1.,
                      'forceNodes': 2 * (nelx + 1) * (nely + 1) - (nely + 1),
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': True, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': False, 'midPt': 0.5 * nelx * elemSize[0]}}

    else:
        exampleName = 'None'
        bcSettings = 'None'
        symMap = 'None'

    return exampleName, bcSettings, symMap
