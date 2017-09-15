%TRAINING,VALIDATING ON LBP FEATURES AND TESTING ON LBP
% FEATURES OF DIGITAL SENSOR

%TRAINING ON ENTIRE DATA OF LBP FEATURES OF DIGITAL SENSOR
load('Train_All_Data_DigiLBP.mat');
tr_LBP=Train_All_Data_DigiLBP;
load('Train_All_Label_DigiLBP.mat');
trl_LBP=Train_All_Label_DigiLBP;

%TESTING ON ENTIRE DATA OF LBP FEATURES OF DIGITAL SENSOR
load('Test_ALL_Data_DigiLBP.mat');
ts_LBP=Test_All_Data_DigiLBP;
load('Test_All_Label_DigiLBP.mat');
tsl_LBP=Test_All_Label_DigiLBP;

%PARAMETERS OF AUTOENCODER 1
hiddenSize1 =400;
autoenc1 = trainAutoencoder(tr_LBP,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat1 = encode(autoenc1,tr_LBP);%FEATURES 1


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
softnet = trainSoftmaxLayer(feat2,trl_LBP,'MaxEpochs',400);

%CREATING A DEEP NEURAL NETWORK (CONTAINING BOTH AUTOENCODERS AND SOFTNET)
deepnet = stack(autoenc1,autoenc2,softnet);

%TRAINING THE DEEP NEURAL NETWORK
deepnet = train(deepnet,tr_LBP,trl_LBP);

%OUTPUT OF TRAINING NETWORK ROC AND PLOT TRAIN ROC CURVE
y=deepnet(tr_LBP);
ezroc3(y,trl_LBP,2,'',1);

%OUTPUT OF TESTING NETWORK ROC AND PLOT TEST ROC CURVE
y1=deepnet(ts_LBP);
ezroc3(y1,tsl_LBP,2,'',1);
