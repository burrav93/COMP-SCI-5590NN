%BONUS-FUSION OF FEATURES
%TRAINING,VALIDATING ON BGP FEATURES AND LBP FEATURES OF DIGITAL SENSOR AND
%TESTING ON BGP AND LBP FEATURES OF DIGITAL SENSOR
%APPLYING CONCATENATION


%TRAINING ON A SET OF BGP AND A SET OF LBP FEATURES OF DIGITAL SENSOR
%(CONCATENATING)
tr_BGP_LBP=[Train_All_Data_DigiBGP(:,1:916) Train_All_Data_DigiBGP(:,1016:1900)*(10^8); Train_All_Data_DigiLBP(:,1:916 ) Train_All_Data_DigiLBP(:,1016:1900)];
trl_BGP_LBP=[Train_All_Label_DigiBGP(:,1:916) Train_All_Label_DigiBGP(:,1016:1900); Train_All_Label_DigiLBP(:,1:916) Train_All_Label_DigiLBP(:,1016:1900)];

%VALIDATING A SET OF DATA FROM TRAIN DATA OF BGP AND LBP FEATURES OF
%DIGITAL SENSOR(CONCATENATING)
trvali_BGP_LBP=[Train_All_Data_DigiBGP(:,917:1016)  Train_All_Data_DigiBGP(:,1901:2004)*(10^8); Train_All_Data_DigiLBP(:,917:1016) Train_All_Data_DigiLBP(:,1901:2004)];
trlvali_BGP_LBP=[Train_All_Label_DigiBGP(:,917:1016)  Train_All_Label_DigiBGP(:,1901:2004); Train_All_Label_DigiLBP(:,917:1016) ,Train_All_Label_DigiLBP(:,1901:2004)];

%TESTING ENTIRELY ON BGP AND LBP FEATURES DATA OF SAGEM SENSOR
%(CONCATENATING)
ts_BGP_LBP=[Test_All_Data_DigiBGP*(10^5) ;Test_All_Data_DigiLBP]; 
tsl_BGP_LBP=[Test_All_Label_DigiBGP ;Test_All_Label_DigiLBP];

%PARAMETERS OF AUTOENCODER 1
hiddenSize1 =700;
autoenc1 = trainAutoencoder(tr_BGP_LBP,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);
feat1 = encode(autoenc1,tr_BGP_LBP);%FEATURES 1 

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
softnet = trainSoftmaxLayer(feat2,trl_BGP_LBP,'MaxEpochs',400);

%CREATING A DEEP NEURAL NETWORK (CONTAINING BOTH AUTOENCODERS AND SOFTNET)
deepnet = stack(autoenc1,autoenc2,softnet);

%TRAINING THE DEEP NEURAL NETWORK
deepnet = train(deepnet,tr_BGP_LBP,trl_BGP_LBP);

%OUTPUT OF TRAINING NETWORK ROC AND PLOT TRAIN ROC CURVE
y1 = deepnet(tr_BGP_LBP);
ezroc3(y1,trl_BGP_LBP,2,'',1);

%OUTPUT OF VALIDATING NETWORK  AND PLOT VALIDATE ROC CURVE
y2 = deepnet(trvali_BGP_LBP);
ezroc3(y2,trlvali_BGP_LBP,2,'',1);

%OUTPUT OF TESTING NETWORK ROC AND PLOT TEST ROC CURVE
y = deepnet(ts_BGP_LBP);
ezroc3(y,tsl_BGP_LBP,2,'',1);