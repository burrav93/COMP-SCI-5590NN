Data = imageSet('att_faces','recursive');
data=cell(1,240);
testdata=cell(1,160);
 a = 1;

 for j=1:40
     for i=1:6;
         X= read(Data(j),i);
         X=double(X)/256;
         data{a} = X;
         a = a + 1;
     end;
 end;
 a=1;
  for j=1:40
     for i=7:10;
         X= read(Data(j),i);
         X=double(X)/256;
         testdata{a} = X;
         a = a + 1;
     end;
 end;
rng('default');
        hiddenSize1 =400;
        autoenc1 = trainAutoencoder(data,hiddenSize1, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
        feat1 = encode(autoenc1,data);
        hiddenSize2 = 100;
        autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',1, ...
            'SparsityProportion',0.5, ...
            'ScaleData', false);
        feat2 = encode(autoenc2,feat1);
        j=0;
        tTrain=zeros(40,240);
        tTest=zeros(40,160);
        y=zeros(40,160);
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
 imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;
xTest = zeros(inputSize,numel(testdata));
for i = 1:numel(testdata)
    xTest(:,i) = testdata{i}(:);
end
xTrain = zeros(inputSize,numel(data));

for i = 1:numel(data)
    xTrain(:,i) = data{i}(:);
end
for i=1:1:40
    softnet = trainSoftmaxLayer(feat2,tTrain(i,:),'MaxEpochs',400);
 deepnet = stack(autoenc1,autoenc2,softnet); 
yTrain(i,:)=deepnet(xTrain);
end;
 H=ezroc3(yTrain,tTrain,2,' ',1);
for i=1:1:40
    softnet = trainSoftmaxLayer(feat2,tTrain(i,:),'MaxEpochs',400);
 deepnet = stack(autoenc1,autoenc2,softnet); 
y(i,:) = deepnet(xTest);
end;
H1=ezroc3(y,tTest,2,' ',1);
for i=1:1:40
    softnet = trainSoftmaxLayer(feat2,tTrain(i,:),'MaxEpochs',400);
 deepnet = stack(autoenc1,autoenc2,softnet); 
 deepnet = train(deepnet,xTrain,tTrain(i,:));
y(i,:) = deepnet(xTest);
end;
H2=ezroc3(y,tTest,2,' ',1);