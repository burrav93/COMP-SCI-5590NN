function H1=foldnarx(d,dval)
%open loop run 1 and run 2 total 2400
tc=num2cell([ones(1,600),-1*ones(1,1800)]);
tt=num2cell([-1*ones(1,600),ones(1,600),-1*ones(1,1200)]);
tr=num2cell([-1*ones(1,1200),ones(1,600),-1*ones(1,600)]);
td=num2cell([-1*ones(1,1800),ones(1,600)]);

% closed loop with respect to run 3 total 1200
tc1=num2cell([ones(1,300),-1*ones(1,900)]);
tt1=num2cell([-1*ones(1,300),ones(1,300),-1*ones(1,600)]);
tr1=num2cell([-1*ones(1,600),ones(1,300),-1*ones(1,300)]);
td1=num2cell([-1*ones(1,900),ones(1,300)]);




n=6;
n1=3;
h=5;

net1 = narxnet(0:n,1:n1,h);
net1.trainFcn='trainbr';
net1.trainParam.epochs=100;
net1.layers{2}.transferFcn='tansig';
net1=openloop(net1);
[Xsc,Xic,Aic,Tsc] = preparets(net1,d,{},tc);
net1 = train(net1,Xsc,Tsc,Xic,Aic);
net1=closeloop(net1);
[Xsc,Xic,Aic,Tsc] = preparets(net1,dval,{},tc1);

Yc=sim(net1,Xsc,Xic,Aic,Tsc);
yc=seq2con(Yc);
yc=yc{1};

h11=[mean((yc(1:300-n))),mean((yc(301-n:600-n))),mean((yc(601-n:900-n))),mean((yc(901-n:1200-n)))];

net2 = narxnet(0:n,1:n1,h);
net2.trainFcn='trainbr';
net2.trainParam.epochs=100;
net2.layers{2}.transferFcn='tansig';
net2=openloop(net2);
[Xsc,Xic,Aic,Tsc] = preparets(net2,d,{},tt);
net2 = train(net2,Xsc,Tsc,Xic,Aic);
net2=closeloop(net2);
[Xsc,Xic,Aic,Tsc] = preparets(net2,dval,{},tt1);
Yt=sim(net2,Xsc,Xic,Aic,Tsc);
yt=seq2con(Yt);
yt=yt{1};

h12=[mean((yt(1:300-n))),mean((yt(301-n:600-n))),mean((yt(601-n:900-n))),mean((yt(901-n:1200-n)))];

net3 = narxnet(0:n,1:n1,h);
net3.trainFcn='trainbr';
net3.trainParam.epochs=100;
net3.layers{2}.transferFcn='tansig';
net3=openloop(net3);
[Xsc,Xic,Aic,Tsc] = preparets(net3,d,{},tr);
net3=closeloop(net3);
[Xsc,Xic,Aic,Tsc] = preparets(net3,dval,{},tr1);
net3 = train(net3,Xsc,Tsc,Xic,Aic);
Yr=sim(net3,Xsc,Xic,Aic,Tsc);
yr=seq2con(Yr);
yr=yr{1};

h13=[mean((yr(1:300-n))),mean((yr(301-n:600-n))),mean((yr(601-n:900-n))),mean((yr(901-n:1200-n)))];


net4 = narxnet(0:n,1:n1,h);
net4.trainFcn='trainbr';
net4.trainParam.epochs=100;
net4.layers{2}.transferFcn='tansig';
net4=openloop(net4);
[Xsc,Xic,Aic,Tsc] = preparets(net4,d,{},td);
net4=closeloop(net4);
[Xsc,Xic,Aic,Tsc] = preparets(net4,dval,{},td1);
net4 = train(net4,Xsc,Tsc,Xic,Aic);
Yd=sim(net4,Xsc,Xic,Aic,Tsc);
yd=seq2con(Yd);
yd=yd{1};

h14=[mean((yd(1:300-n))),mean((yd(301-n:600-n))),mean((yd(601-n:900-n))),mean((yd(901-n:1200-n)))];
H1=[h11;h12;h13;h14]; 

end





