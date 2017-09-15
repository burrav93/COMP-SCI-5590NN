%TRAINING,VALIDATING ON BSIF FEATURES AND TESTING ON BSIF
% FEATURES OF DIGITAL SENSOR

%TRAINING ON ENTIRE DATA OF LBP FEATURES OF DIGITAL SENSOR
load('Train_All_Data_DigiBSIF.mat');
tr_BSIF=Train_All_Data_DigiBSIF*(10^8);
load('Train_All_Label_DigiBSIF.mat');
trl_BSIF=Train_All_Label_DigiBSIF;

%TESTING ON ENTIRE DATA OF BSIF FEATURES OF DIGITAL SENSOR
load('Test_All_Data_DigiBSIF.mat');
ts_BSIF=Test_All_Data_DigiBSIF*(10^8);
load('Test_All_Label_DigiBSIF.mat');
tsl_BSIF=Test_All_Label_DigiBSIF;

%PARAMETERS OF AUTOENCODER 1
hiddenSize1 =400;
autoenc1 = trainAutoencoder(tr_BSIF,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat1 = encode(autoenc1,tr_BSIF);%FEATURES 1

%PARAMETERS OF AUTOENCODER 2
hiddenSize2 = 150;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);%FEATURES 2

%TRAINING SOFTMAX LAYER
softnet = trainSoftmaxLayer(feat2,trl_BSIF,'MaxEpochs',400);

%CREATING A DEEP NEURAL NETWORK (CONTAINING BOTH AUTOENCODERS AND SOFTNET)
deepnet = stack(autoenc1,autoenc2,softnet);

%TRAINING THE DEEP NEURAL NETWORK
deepnet = train(deepnet,tr_BSIF,trl_BSIF);

%OUTPUT OF TRAINING NETWORK ROC AND PLOT TRAIN ROC CURVE
y=deepnet(tr_BSIF);
ezroc3(y,trl_BSIF,2,'',1);


%OUTPUT OF TESTING NETWORK ROC AND PLOT TEST ROC CURVE
y1=deepnet(ts_BSIF);
ezroc3(y,tsl_BSIF,2,'',1);