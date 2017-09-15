function H1=fold(d,dtest)

%run1 wrt run2 total 2400
tc=num2cell([ones(1,600),-1*ones(1,1800)]);
tt=num2cell([-1*ones(1,600),ones(1,600),-1*ones(1,1200)]);
tr=num2cell([-1*ones(1,1200),ones(1,600),-1*ones(1,600)]);
td=num2cell([-1*ones(1,1800),ones(1,600)]);

n=20;

net1 = timedelaynet(1:n,10);
net1.trainFcn='trainbr';
net1.trainParam.epochs=100;
net1.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net1,d,tc);
net1 = train(net1,Xsc,Tsc,Xic,Aic);
Yc=sim(net1,dtest);
yc=seq2con(Yc);
yc=yc{1};
h11=[mean((yc(1:300-n))),mean((yc(301-n:600-n))),mean((yc(601-n:900-n))),mean((yc(901-n:1200-n)))];
net2 = timedelaynet(1:n,10);
net2.trainFcn='trainbr';
net2.trainParam.epochs=100;
net2.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net2,d,tt);
net2 = train(net2,Xsc,Tsc,Xic,Aic);
Yt=sim(net2,dtest);
yt=seq2con(Yt);
yt=yt{1};
h12=[mean((yt(1:300-n))),mean((yt(301-n:600-n))),mean((yt(601-n:900-n))),mean((yt(901-n:1200-n)))];
net3 = timedelaynet(1:n,10);
net3.trainFcn='trainbr';
net3.trainParam.epochs=100;
net3.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net3,d,tr);
net3 = train(net3,Xsc,Tsc,Xic,Aic);
Yr=sim(net3,dtest);
yr=seq2con(Yr);
yr=yr{1};
h13=[mean((yr(1:300-n))),mean((yr(301-n:600-n))),mean((yr(601-n:900-n))),mean((yr(901-n:1200-n)))];
net4 = timedelaynet(1:n,10);
net4.trainFcn='trainbr';
net4.trainParam.epochs=100;
net4.layers{2}.transferFcn='tansig';

[Xsc,Xic,Aic,Tsc] = preparets(net4,d,td);
net4 = train(net4,Xsc,Tsc,Xic,Aic);
Yd=sim(net4,dtest);
yd=seq2con(Yd);
yd=yd{1};
h14=[mean((yd(1:300-n))),mean((yd(301-n:600-n))),mean((yd(601-n:900-n))),mean((yd(901-n:1200-n)))];

H1=[h11;h12;h13;h14];
end





