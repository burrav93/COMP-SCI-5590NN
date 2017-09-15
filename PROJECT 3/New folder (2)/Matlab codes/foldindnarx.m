function H1=foldindnarx(d,dval)
%open loop 
tc=num2cell(repmat([ones(1,900),-1*ones(1,2700)],[1,5]));
tt=num2cell(repmat([-1*ones(1,900),ones(1,900),-1*ones(1,1800)],[1,5]));
tr=num2cell(repmat([-1*ones(1,1800),ones(1,900),-1*ones(1,900)],[1,5]));
td=num2cell(repmat([-1*ones(1,2700),ones(1,900)],[1,5]));
% closed loop no repmat
tc1=num2cell([ones(1,900),-1*ones(1,2700)]);
tt1=num2cell([-1*ones(1,900),ones(1,900),-1*ones(1,1800)]);
tr1=num2cell([-1*ones(1,1800),ones(1,900),-1*ones(1,900)]);
td1=num2cell([-1*ones(1,2700),ones(1,900)]);




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

h11=[mean((yc(1:900-n))),mean((yc(901-n:1800-n))),mean((yc(1801-n:2700-n))),mean((yc(2701-n:3600-n)))];

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

h12=[mean((yt(1:900-n))),mean((yt(901-n:1800-n))),mean((yt(1801-n:2700-n))),mean((yt(2701-n:3600-n)))];

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

h13=[mean((yr(1:900-n))),mean((yr(901-n:1800-n))),mean((yr(1801-n:2700-n))),mean((yr(2701-n:3600-n)))];


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

h14=[mean((yd(1:900-n))),mean((yd(901-n:1800-n))),mean((yd(1801-n:2700-n))),mean((yd(2701-n:3600-n)))];
H1=[h11;h12;h13;h14]; 

end





