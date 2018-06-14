clear all
clc
p1=[0;2]; p2=[1;0]; p3=[0;-2]; p4=[2;0];
p=[p1 p2 p3 p4]
t=[1; 1; 0; 0];
W(:,1)=[0;0];
b(1)=0
w = W;
bias = b;
count = 0;
count1 = 0;
j = 1;
for i=1:50
    a(j)=hardlim(w'*p(:,j)+bias);
    if a(j)==t(j)
        W(:,i+1)=W(:,i);
        b(i+1)=b(i);
    else
        e=t(j)-a(j);
        W(:,i+1)=W(:,i)+e*(p(:,j));
        b(i+1)=b(i)+e;
    end
    w = W(:,i+1);
    bias = b(i+1);
    count = count + 1;
    j = j + 1;
    if count == length(p)
        count = 0;
        for q = 1:length(p)
            if a(q) == t(q);
                count1 = count1 + 1;
            end
            j = 1;
        end
    end
    if count1 == length(p)
        break
    else
         count1 = 0;
    end
end
