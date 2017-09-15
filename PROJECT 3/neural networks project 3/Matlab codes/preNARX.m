data=importdata('Person2.xls',' ',3);
c1=data.data.Circle(:,[1,3]);
t1=data.data.Triangle(:,[1,3]);
r1=data.data.Right(:,[1,3]);
d1=data.data.Down(:,[1,3]);
c1=num2cell(c1',1);
t1=num2cell(t1',1);
r1=num2cell(r1',1);
d1=num2cell(d1',1);
d=[c1,t1,r1,d1];
c2=data.data.Circle(:,[5,7]);
t2=data.data.Triangle(:,[5,7]);
r2=data.data.Right(:,[5,7]);
d2=data.data.Down(:,[5,7]);
c2=num2cell(c2',1);
t2=num2cell(t2',1);
r2=num2cell(r2',1);
d2=num2cell(d2',1);
dval=[c2,t2,r2,d2];
n=6;
n1=3;
h=5;

tc=num2cell([ones(1,300),-1*ones(1,900)]);
tt=num2cell([-1*ones(1,300),ones(1,300),-1*ones(1,600)]);
tr=num2cell([-1*ones(1,600),ones(1,300),-1*ones(1,300)]);
td=num2cell([-1*ones(1,900),ones(1,300)]);

H=zeros(4,4,5);

for i=1:5
net1 = narxnet(0:n,1:n1,h);
net1.trainFcn='trainbr';
net1.trainParam.epochs=100;
net1.layers{2}.transferFcn='tansig';
net1=openloop(net1);
[Xsc,Xic,Aic,Tsc] = preparets(net1,d,{},tc);
net1 = train(net1,Xsc,Tsc,Xic,Aic);
net1=closeloop(net1);
[Xsc,Xic,Aic,Tsc] = preparets(net1,dval,{},tc);

Yc=sim(net1,Xsc,Xic,Aic,Tsc);
yc=seq2con(Yc);
yc=yc{1};
net1=init(net1);
h11=[mean((yc(1:300-n))),mean((yc(301-n:600-n))),mean((yc(601-n:900-n))),mean((yc(901-n:1200-n)))];

net2 = narxnet(0:n,1:n1,h);
net2.trainFcn='trainbr';
net2.trainParam.epochs=100;
net2.layers{2}.transferFcn='tansig';
net2=openloop(net2);
[Xsc,Xic,Aic,Tsc] = preparets(net2,d,{},tt);
net2 = train(net2,Xsc,Tsc,Xic,Aic);
net2=closeloop(net2);
[Xsc,Xic,Aic,Tsc] = preparets(net2,dval,{},tt);
Yt=sim(net2,Xsc,Xic,Aic,Tsc);
yt=seq2con(Yt);
yt=yt{1};
net2=init(net2);
h12=[mean((yt(1:300-n))),mean((yt(301-n:600-n))),mean((yt(601-n:900-n))),mean((yt(901-n:1200-n)))];

net3 = narxnet(0:n,1:n1,h);
net3.trainFcn='trainbr';
net3.trainParam.epochs=100;
net3.layers{2}.transferFcn='tansig';
net3=openloop(net3);
[Xsc,Xic,Aic,Tsc] = preparets(net3,d,{},tr);
net3=closeloop(net3);
[Xsc,Xic,Aic,Tsc] = preparets(net3,dval,{},tr);
net3 = train(net3,Xsc,Tsc,Xic,Aic);
Yr=sim(net3,Xsc,Xic,Aic,Tsc);
yr=seq2con(Yr);
yr=yr{1};
net3=init(net3);
h13=[mean((yr(1:300-n))),mean((yr(301-n:600-n))),mean((yr(601-n:900-n))),mean((yr(901-n:1200-n)))];


net4 = narxnet(0:n,1:n1,h);
net4.trainFcn='trainbr';
net4.trainParam.epochs=100;
net4.layers{2}.transferFcn='tansig';
net4=openloop(net4);
[Xsc,Xic,Aic,Tsc] = preparets(net4,d,{},td);
net4=closeloop(net4);
[Xsc,Xic,Aic,Tsc] = preparets(net4,dval,{},td);
net4 = train(net4,Xsc,Tsc,Xic,Aic);
Yd=sim(net4,Xsc,Xic,Aic,Tsc);
yd=seq2con(Yd);
yd=yd{1};
net4=init(net4);
h14=[mean((yd(1:300-n))),mean((yd(301-n:600-n))),mean((yd(601-n:900-n))),mean((yd(901-n:1200-n)))];
H(:,:,i)=[h11;h12;h13;h14]; 
end

ezroc3(H,[],2,'',1);


