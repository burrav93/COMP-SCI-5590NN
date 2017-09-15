 Data = imageSet('orl_faces','recursive');
data=cell(1,240);
 a = 1;

 for j=1:40
     for i=1:6;
         X= read(Data(j),i);
         X=double(X)/256;
         data{a} = X;
         a = a + 1;
     end;
 end;
rng('default');

        hiddenSize1 =[400,500,400];
        
        hiddenSize2 = [100,200,200];
        for ii=1:3
            hid1 =hiddenSize1(ii);
        autoenc1 = trainAutoencoder(data,hid1, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
        
        feat1 = encode(autoenc1,data);
        hid2=hiddenSize2(ii);
        autoenc2 = trainAutoencoder(feat1,hid2, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
        feat2 = encode(autoenc2,feat1);
        
        
 j=0;
 for i=1:1:40
     tTrain(i,:)=[ones(1,6),zeros(1,234)];   
     tTrain(i,:)=circshift(tTrain(i,:),[0 j]);
     j=j+6;
 end
 j=0;
 for i=1:1:40
     tTest(i,:)=[ones(1,4),zeros(1,156)];   
     tTest(i,:)=circshift(tTest(i,:),[0 j]);
     j=j+4;
 end
 softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
 deepnet = stack(autoenc1,autoenc2,softnet);
 testdata=cell(1,160);
  a = 1;

 for j=1:40
     for i=7:10;
         X= read(Data(j),i);
         X=double(X)/256;
         testdata{a} = X;
         a = a + 1;
     end;
 end;
 imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;
xTest = zeros(inputSize,numel(testdata));
for i = 1:numel(testdata)
    xTest(:,i) = testdata{i}(:);
end
y=zeros(40,160,3);
y1=zeros(40,160,3);
yTrain=zeros(40,240,3);
y(:,:,ii) = deepnet(xTest);
H2=ezroc3(y(:,:,ii),tTest,2,' Test ROC before fine-tune',1);
xTrain = zeros(inputSize,numel(data));
for i = 1:numel(data)
    xTrain(:,i) = data{i}(:);
end
yTrain(:,:,ii)=deepnet(xTrain);
ezroc3(yTrain(:,:,ii) ,tTrain,2,'Train ROC',1);
deepnet = train(deepnet,xTrain,tTrain);

y1(:,:,ii) = deepnet(xTest);
H3=ezroc3(y1(:,:,ii),tTest,2,' Test ROC after fine tune',1);
        end;
Y=y(:,:,1)+y(:,:,2)+y(:,:,3);
H=ezroc3(Y/3,tTest,2,'Test committee',1);
Y1=yTrain(:,:,1)+yTrain(:,:,2)+yTrain(:,:,3);
ezroc3(Y1/3,tTrain,2,'Train committee',1);

