%TRAINING,VALIDATING ON BGP FEATURES AND TESTING ON BGP
% FEATURES OF SAGEM SENSOR

%TRAINING ON ENTIRE DATA OF BGP FEATURES OF SAGEM SENSOR
load('Train_All_Data_SageBGP.mat');
tr_BGP=Train_All_Data_SageBGP*(10^8);
load('Train_All_Label_SageBGP.mat');
trl_BGP=Train_All_Label_SageBGP;

%TESTING ON ENTIRE DATA OF BGP FEATURES OF SAGEM SENSOR
load('Test_All_Data_SageBGP.mat')
ts_BGP=Test_All_Data_SageBGP*(10^8);
load('Test_All_Label_SageBGP.mat')
tsl_BGP=Test_All_Label_SageBGP;

%PARAMETERS OF AUTOENCODER 1
hiddenSize1 =400;
autoenc1 = trainAutoencoder(tr_BGP,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat1 = encode(autoenc1,tr_BGP);%FEATURES 1

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
softnet = trainSoftmaxLayer(feat2,trl_BGP,'MaxEpochs',400);
%CREATING A DEEP NEURAL NETWORK (CONTAINING BOTH AUTOENCODERS AND SOFTNET)
deepnet = stack(autoenc1,autoenc2,softnet);
%TRAINING THE DEEP NEURAL NETWORK
deepnet = train(deepnet,tr_BGP,trl_BGP);
%OUTPUT OF TRAINING NETWORK ROC AND PLOT TRAIN ROC CURVE
y=deepnet(tr_BGP);
ezroc3(y,trl_BGP,2,'',1);
%OUTPUT OF TESTING NETWORK ROC AND PLOT TEST ROC CURVE
y1=deepnet(ts_BGP);
ezroc3(y1,tsl_BGP,2,'',1);