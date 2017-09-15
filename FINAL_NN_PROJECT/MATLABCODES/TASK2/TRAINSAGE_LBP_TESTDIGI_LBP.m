% TRAINING,VALIDATING ON LBP FEATURES OF SAGEM SENSOR AND TESTING ON 
% LBP FEATURES OF DIGITAL SENSOR


%TRAINING ON LBP FEATURES OF DIGITAL SENSOR
tr_LBP=[Train_All_Data_SageLBP(:,1:916 )  Train_All_Data_SageLBP(:,1016:1916 )];
trl_LBP=[Train_All_Label_SageLBP(:,1:916) Train_All_Label_SageLBP(:,1016:1916) ];

%VALIDATING A SET OF DATA FROM TRAIN DATA OF LBP FEATURES OF SAGEM SENSOR
trvali_LBP=[Train_All_Data_SageLBP(:,916:1016) Train_All_Data_SageLBP(:,1916:2016)];
trlvali_LBP=[Train_All_Label_SageLBP(:,916:1016) Train_All_Label_SageLBP(:,1916:2016)];

%TESTING ENTIRELY ON LBP FEATURES DATA OF DIGITAL SENSOR
ts_LBP= [Test_All_Data_DigiLBP];
tsl_LBP= [Test_All_Label_DigiLBP];

%PARAMETERS OF AUTOENCODER 1
hiddenSize1 =400;
autoenc1 = trainAutoencoder(tr_LBP,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat1 = encode(autoenc1,tr_LBP);%FEATURES 1 

%PARAMETERS OF AUTOENCODER 2
hiddenSize2 = 150;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat2 = encode(autoenc2,feat1);%FEATURES 2

%TRAINING SOFTMAX LAYER
softnet = trainSoftmaxLayer(feat2,trl_LBP,'MaxEpochs',200);

%CREATING A DEEP NEURAL NETWORK (CONTAINING BOTH AUTOENCODERS AND SOFTNET)
deepnet = stack(autoenc1,autoenc2,softnet);

%TRAINING THE DEEP NEURAL NETWORK
deepnet = train(deepnet,tr_LBP,trl_LBP);


%OUTPUT OF TRAINING NETWORK ROC AND PLOT TRAIN ROC CURVE
y1 = deepnet(tr_LBP);
ezroc3(y1,trl_LBP,2,'',1);


%OUTPUT OF VALIDATING NETWORK  AND PLOT VALIDATE ROC CURVE
y2 = deepnet(trvali_LBP);
ezroc3(y2,trlvali_LBP,2,'',1);

%OUTPUT OF TESTING NETWORK ROC AND PLOT TEST ROC CURVE
y = deepnet(ts_LBP);
ezroc3(y,tsl_LBP,2,'',1);