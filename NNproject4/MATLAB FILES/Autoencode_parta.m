Data = imageSet('att_faces','recursive');
data=cell(1,240);
a = 1;
u=1;
for j=1:40
    for i=1:6;
        X= read(Data(j),i);
        X=double(X)/256;
        data{a} = X;
        a = a + 1;
    end;
end;
rng('default');

hiddenSize1 = [400,300,400];
autoenc1 = trainAutoencoder(data,hiddenSize1, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.4, ...
    'ScaleData', false);

plotWeights(autoenc1);
feat1 = encode(autoenc1,data);


hiddenSize2 = [100,200,200];
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',1, ...
    'SparsityProportion',0.25, ...
    'ScaleData', false);

plotWeights(autoenc2);
feat2 = encode(autoenc2,feat1);

intra=zeros(1,600);
k=1;
for pointer1=0:6:234;
    for i= 1:1:6
        for j=2:1:6
            if i<j
                    intra(1,k)=(mse(feat2(:,i+pointer1),feat2(:,j+pointer1)));
                     k=k+1;
            end;
        end;
    end;
end;
k1=1;
inter=zeros(1,28080);
for pointer1=0:6:234
    for pointer2=0:6:234;
        if(pointer1<pointer2)
            for i=1:1:6
                for  j=1:1:6;
                    inter(1,k1)=(mse(feat2(:,i+pointer1),feat2(:,j+pointer2)));
                          k1=k1+1;
                end;
            end;
        end;
    end;
end;

ratio=sum(inter(:))/sum(intra(:))

H=[-1*inter -1*intra];
T=[zeros(1,28080),ones(1,600)];
H1=ezroc3(H,T,2,' ',1);
