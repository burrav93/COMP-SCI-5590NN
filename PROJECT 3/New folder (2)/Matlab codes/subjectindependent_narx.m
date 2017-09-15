data1=importdata('Person1.xls',' ',3);
p1c1=data1.data.Circle(:,[1,3]);
p1t1=data1.data.Triangle(:,[1,3]);
p1r1=data1.data.Right(:,[1,3]);
p1d1=data1.data.Down(:,[1,3]);
p1c1=num2cell(p1c1',1);
p1t1=num2cell(p1t1',1);
p1r1=num2cell(p1r1',1);
p1d1=num2cell(p1d1',1);

p1c2=data1.data.Circle(:,[5,7]);
p1t2=data1.data.Triangle(:,[5,7]);
p1r2=data1.data.Right(:,[5,7]);
p1d2=data1.data.Down(:,[5,7]);
p1c2=num2cell(p1c2',1);
p1t2=num2cell(p1t2',1);
p1r2=num2cell(p1r2',1);
p1d2=num2cell(p1d2',1);


p1c3=data1.data.Circle(:,[9,11]);
p1t3=data1.data.Triangle(:,[9,11]);
p1r3=data1.data.Right(:,[9,11]);
p1d3=data1.data.Down(:,[9,11]);
p1c3=num2cell(p1c3',1);
p1t3=num2cell(p1t3',1);
p1r3=num2cell(p1r3',1);
p1d3=num2cell(p1d3',1);

da1=[p1c1,p1c2,p1c3,p1t1,p1t2,p1t3,p1r1,p1r2,p1r3,p1d1,p1d2,p1d3];

data2=importdata('Person2.xls',' ',3);
p2c1=data2.data.Circle(:,[1,3]);
p2t1=data2.data.Triangle(:,[1,3]);
p2r1=data2.data.Right(:,[1,3]);
p2d1=data2.data.Down(:,[1,3]);
p2c1=num2cell(p2c1',1);
p2t1=num2cell(p2t1',1);
p2r1=num2cell(p2r1',1);
p2d1=num2cell(p2d1',1);

p2c2=data2.data.Circle(:,[5,7]);
p2t2=data2.data.Triangle(:,[5,7]);
p2r2=data2.data.Right(:,[5,7]);
p2d2=data2.data.Down(:,[5,7]);
p2c2=num2cell(p2c2',1);
p2t2=num2cell(p2t2',1);
p2r2=num2cell(p2r2',1);
p2d2=num2cell(p2d2',1);


p2c3=data2.data.Circle(:,[9,11]);
p2t3=data2.data.Triangle(:,[9,11]);
p2r3=data2.data.Right(:,[9,11]);
p2d3=data2.data.Down(:,[9,11]);
p2c3=num2cell(p2c3',1);
p2t3=num2cell(p2t3',1);
p2r3=num2cell(p2r3',1);
p2d3=num2cell(p2d3',1);
da2=[p2c1,p2c2,p2c3,p2t1,p2t2,p2t3,p2r1,p2r2,p2r3,p2d1,p2d2,p2d3];


data3=importdata('Person3.xls',' ',3);
p3c1=data3.data.Circle(:,[1,3]);
p3t1=data3.data.Triangle(:,[1,3]);
p3r1=data3.data.Right(:,[1,3]);
p3d1=data3.data.Down(:,[1,3]);
p3c1=num2cell(p3c1',1);
p3t1=num2cell(p3t1',1);
p3r1=num2cell(p3r1',1);
p3d1=num2cell(p3d1',1);


p3c2=data3.data.Circle(:,[5,7]);
p3t2=data3.data.Triangle(:,[5,7]);
p3r2=data3.data.Right(:,[5,7]);
p3d2=data3.data.Down(:,[5,7]);
p3c2=num2cell(p3c2',1);
p3t2=num2cell(p3t2',1);
p3r2=num2cell(p3r2',1);
p3d2=num2cell(p3d2',1);


p3c3=data3.data.Circle(:,[9,11]);
p3t3=data3.data.Triangle(:,[9,11]);
p3r3=data3.data.Right(:,[9,11]);
p3d3=data3.data.Down(:,[9,11]);
p3c3=num2cell(p3c3',1);
p3t3=num2cell(p3t3',1);
p3r3=num2cell(p3r3',1);
p3d3=num2cell(p3d3',1);

da3=[p3c1,p3c2,p3c3,p3t1,p3t2,p3t3,p3r1,p3r2,p3r3,p3d1,p3d2,p3d3];



data4=importdata('Person4.xls',' ',3);
p4c1=data4.data.Circle(:,[1,3]);
p4t1=data4.data.Triangle(:,[1,3]);
p4r1=data4.data.Right(:,[1,3]);
p4d1=data4.data.Down(:,[1,3]);
p4c1=num2cell(p4c1',1);
p4t1=num2cell(p4t1',1);
p4r1=num2cell(p4r1',1);
p4d1=num2cell(p4d1',1);

p4c2=data4.data.Circle(:,[5,7]);
p4t2=data4.data.Triangle(:,[5,7]);
p4r2=data4.data.Right(:,[5,7]);
p4d2=data4.data.Down(:,[5,7]);
p4c2=num2cell(p4c2',1);
p4t2=num2cell(p4t2',1);
p4r2=num2cell(p4r2',1);
p4d2=num2cell(p4d2',1);


p4c3=data4.data.Circle(:,[9,11]);
p4t3=data4.data.Triangle(:,[9,11]);
p4r3=data4.data.Right(:,[9,11]);
p4d3=data4.data.Down(:,[9,11]);
p4c3=num2cell(p4c3',1);
p4t3=num2cell(p4t3',1);
p4r3=num2cell(p4r3',1);
p4d3=num2cell(p4d3',1);

da4=[p4c1,p4c2,p4c3,p4t1,p4t2,p4t3,p4r1,p4r2,p4r3,p4d1,p4d2,p4d3];


data5=importdata('Person5.xls',' ',3);
p5c1=data5.data.Circle(:,[1,3]);
p5t1=data5.data.Triangle(:,[1,3]);
p5r1=data5.data.Right(:,[1,3]);
p5d1=data5.data.Down(:,[1,3]);
p5c1=num2cell(p5c1',1);
p5t1=num2cell(p5t1',1);
p5r1=num2cell(p5r1',1);
p5d1=num2cell(p5d1',1);

p5c2=data5.data.Circle(:,[5,7]);
p5t2=data5.data.Triangle(:,[5,7]);
p5r2=data5.data.Right(:,[5,7]);
p5d2=data5.data.Down(:,[5,7]);
p5c2=num2cell(p5c2',1);
p5t2=num2cell(p5t2',1);
p5r2=num2cell(p5r2',1);
p5d2=num2cell(p5d2',1);


p5c3=data5.data.Circle(:,[9,11]);
p5t3=data5.data.Triangle(:,[9,11]);
p5r3=data5.data.Right(:,[9,11]);
p5d3=data5.data.Down(:,[9,11]);
p5c3=num2cell(p5c3',1);
p5t3=num2cell(p5t3',1);
p5r3=num2cell(p5r3',1);
p5d3=num2cell(p5d3',1);

da5=[p5c1,p5c2,p5c3,p5t1,p5t2,p5t3,p5r1,p5r2,p5r3,p5d1,p5d2,p5d3];




data6=importdata('Person6.xls',' ',3);
p6c1=data6.data.Circle(:,[1,3]);
p6t1=data6.data.Triangle(:,[1,3]);
p6r1=data6.data.Right(:,[1,3]);
p6d1=data6.data.Down(:,[1,3]);
p6c1=num2cell(p6c1',1);
p6t1=num2cell(p6t1',1);
p6r1=num2cell(p6r1',1);
p6d1=num2cell(p6d1',1);

p6c2=data6.data.Circle(:,[5,7]);
p6t2=data6.data.Triangle(:,[5,7]);
p6r2=data6.data.Right(:,[5,7]);
p6d2=data6.data.Down(:,[5,7]);
p6c2=num2cell(p6c2',1);
p6t2=num2cell(p6t2',1);
p6r2=num2cell(p6r2',1);
p6d2=num2cell(p6d2',1);


p6c3=data6.data.Circle(:,[9,11]);
p6t3=data6.data.Triangle(:,[9,11]);
p6r3=data6.data.Right(:,[9,11]);
p6d3=data6.data.Down(:,[9,11]);
p6c3=num2cell(p6c3',1);
p6t3=num2cell(p6t3',1);
p6r3=num2cell(p6r3',1);
p6d3=num2cell(p6d3',1);

da6=[p6c1,p6c2,p6c3,p6t1,p6t2,p6t3,p6r1,p6r2,p6r3,p6d1,p6d2,p6d3];




dat1=[da1,da2,da3,da4,da5];
dat2=[da1,da2,da3,da4,da6];
dat3=[da1,da2,da3,da5,da6];
dat4=[da1,da2,da4,da5,da6];
dat5=[da1,da3,da4,da5,da6];
dat6=[da2,da3,da4,da5,da6];

H1=foldindnarx(dat1,da6);

H2=foldindnarx(dat2,da5);

H3=foldindnarx(dat3,da4);

H4=foldindnarx(dat4,da3);

H5=foldindnarx(dat5,da2);

H6=foldindnarx(dat6,da1);

ezroc3(cat(3,H1,H2,H3,H4,H5,H6),[],2,' ',1);