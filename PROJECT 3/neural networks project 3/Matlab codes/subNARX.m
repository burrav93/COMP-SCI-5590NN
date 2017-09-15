data=importdata('Person1.xls',' ',3);
c1=data.data.Circle(:,[1,3]);
t1=data.data.Triangle(:,[1,3]);
r1=data.data.Right(:,[1,3]);
d1=data.data.Down(:,[1,3]);
c1=num2cell(c1',1);
t1=num2cell(t1',1);
r1=num2cell(r1',1);
d1=num2cell(d1',1);

c2=data.data.Circle(:,[5,7]);
t2=data.data.Triangle(:,[5,7]);
r2=data.data.Right(:,[5,7]);
d2=data.data.Down(:,[5,7]);
c2=num2cell(c2',1);
t2=num2cell(t2',1);
r2=num2cell(r2',1);
d2=num2cell(d2',1);


c3=data.data.Circle(:,[9,11]);
t3=data.data.Triangle(:,[9,11]);
r3=data.data.Right(:,[9,11]);
d3=data.data.Down(:,[9,11]);
c3=num2cell(c3',1);
t3=num2cell(t3',1);
r3=num2cell(r3',1);
d3=num2cell(d3',1);

    
dfold1=[c1,c2,t1,t2,r1,r2,d1,d2];
dfold2=[c1,c3,t1,t3,r1,r3,d1,d3];
dfold3=[c2,c3,t2,t3,r2,r3,d2,d3];
dtest1=[c3,t3,r3,d3];
dtest2=[c2,t2,r2,d2];
dtest3=[c1,t1,r1,d1];
H1=foldnarx(dfold1,dtest1);
H2=foldnarx(dfold2,dtest2);
H3=foldnarx(dfold3,dtest3);
ezroc3(cat(3,H1,H2,H3),[],2,' ',1);
