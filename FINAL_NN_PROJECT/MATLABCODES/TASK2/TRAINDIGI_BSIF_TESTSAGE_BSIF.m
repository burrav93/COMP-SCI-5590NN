% TRAINING,VALIDATING ON BSIF FEATURES OF DIGITAL SENSOR AND TESTING ON 
% BSIF FEATURES OF SAGEM SENSOR
%I HAVE MULTIPLIED THE BSIF TRAIN AND TEST DATA WITH A FACTOR OF (10^8)


%TRAINING ON BSIF FEATURES OF DIGITAL SENSOR 
tr_BSIF=[Train_All_Data_DigiBSIF(:,1:916) Train_All_Data_DigiBSIF(:,1016:1904)]*(10^8);
trl_BSIF=[Train_All_Label_DigiBSIF(:,1:916) Train_All_Label_DigiBSIF(:,1016:1904)];

%VALIDATING A SET OF DATA FROM TRAIN DATA OF BSIF FEATURES OF DIGITAL SENSOR
trvali_BSIF=[Train_All_Data_DigiBSIF(:,916:1016) Train_All_Data_DigiBSIF(:,1904:2004)]*(10^8);
trlvali_BSIF=[Train_All_Label_DigiBSIF(:,916:1016) Train_All_Label_DigiBSIF(:,1904:2004)];

%TESTING ENTIRELY ON BSIF FEATURES DATA OF SAGEM SENSOR
ts_BSIF=[Test_All_Data_SageBSIF Test_All_Data_SageBSIF]*(10^8);
tsl_BSIF=[Test_All_Label_SageBSIF Test_All_Label_SageBSIF];

%PARAMETERS OF AUTOENCODER 1
hiddenSize1 =400;
autoenc1 = trainAutoencoder(tr_BSIF,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
plotWeights(autoenc1);
feat1 = encode(autoenc1,tr_BSIF);%FEATURES 1 

%PARAMETERS OF AUTOENCODER 2
hiddenSize2 = 150;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat2 = encode(autoenc2,feat1); %FEATURES 2


%TRAINING SOFTMAX LAYER
softnet = trainSoftmaxLayer(feat2,trl_BSIF,'MaxEpochs',200);

%CREATING A DEEP NEURAL NETWORK (CONTAINING BOTH AUTOENCODERS AND SOFTNET)
deepnet = stack(autoenc1,autoenc2,softnet);

%TRAINING THE DEEP NEURAL NETWORK
deepnet = train(deepnet,tr_BSIF,trl_BSIF);

%OUTPUT OF TRAINING NETWORK ROC AND PLOT TRAIN ROC CURVE
y1 = deepnet(tr_BSIF);
ezroc3(y1,trl_BSIF,2,'',1);

%OUTPUT OF VALIDATING NETWORK  AND PLOT VALIDATE ROC CURVE
y2 = deepnet(trvali_BSIF);
ezroc3(y2,trlvali_BSIF,2,'',1);

%OUTPUT OF TESTING NETWORK ROC AND PLOT TEST ROC CURVE
y = deepnet(ts_BSIF);
ezroc3(y,tsl_BSIF,2,'',1);