% TRAINING,VALIDATING ON BGP FEATURES OF DIGITAL SENSOR AND TESTING ON BGP
% FEATURES OF SAGEM SENSOR

%TRAINING ON BGP FEATURES OF DIGITAL SENSOR 
tr_BGP=[Train_All_Data_DigiBGP(:,1:916) Train_All_Data_DigiBGP(:,1016:1904)]*(10^8);  
trl_BGP=[Train_All_Label_DigiBGP(:,1:916) Train_All_Label_DigiBGP(:,1016:1904)];

%VALIDATING A SET OF DATA FROM TRAIN DATA OF BGP FEATURES OF DIGITAL SENSOR
trvali_BGP=[Train_All_Data_DigiBGP(:,916:1016) Train_All_Data_DigiBGP(:,1904:2004)].*(10^8); 
trlvali_BGP=[Train_All_Label_DigiBGP(:,916:1016) Train_All_Label_DigiBGP(:,1904:2004)];

%TESTING ENTIRELY ON BGP FEATURES DATA OF SAGEM SENSOR
ts_BGP=[Test_All_Data_SageBGP].*(10^8); 
tsl_BGP=[Test_All_Label_SageBGP];

%PARAMETERS OF AUTOENCODER 1
hiddenSize1 = 400;
autoenc1 = trainAutoencoder(tr_BGP,hiddenSize1, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat1 = encode(autoenc1,tr_BGP); %FEATURES 1 

%PARAMETERS OF AUTOENCODER 2
hiddenSize2 = 150;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat2 = encode(autoenc2,feat1); %FEATURES 2

%TRAINING SOFTMAX LAYER
softnet = trainSoftmaxLayer(feat2,trl_BGP,'MaxEpochs',100);

%CREATING A DEEP NEURAL NETWORK (CONTAINING BOTH AUTOENCODERS AND SOFTNET)
deepnet = stack(autoenc1,autoenc2,softnet);

%TRAINING THE DEEP NEURAL NETWORK
deepnet = train(deepnet,tr_BGP,trl_BGP);

%OUTPUT OF TRAINING NETWORK ROC AND PLOT TRAIN ROC CURVE
y1 = deepnet(tr_BGP); 
ezroc3(y1,trl_BGP,2,'',1);

%OUTPUT OF VALIDATING NETWORK  AND PLOT VALIDATE ROC CURVE
y2 = deepnet(trvali_BGP);
ezroc3(y2,trlvali_BGP,2,'',1);

%OUTPUT OF TESTING NETWORK ROC AND PLOT TEST ROC CURVE
y = deepnet(ts_BGP);
ezroc3(y,tsl_BGP,2,'',1);