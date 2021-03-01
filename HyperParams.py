ImagePath = '../CelebFaces Dataset'
CheckpointPath = './Snapshots'

BatchSize = 128

ImageSize = 64

LatentVectorSize = 256

Ngf = 64

Ndf = 64

LearningRateG = 0.0002
LearningRateD = 0.0002

# Adam learning parameter
Beta = 0.5

NumEpochs = 10
CheckpointEvery = 1
ReportEvery = 100

NumWorkers = 6

Device = 'cuda:0'
