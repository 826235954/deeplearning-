function [ f_new,f_new_freq ] = Filtering( f,h,type )
% f:���˲�ͼ��
% h:�˲�������
% type:�˲�������

% f_new:�˲����ͼ��
% f_new_freq:�˲���ͼ��Ƶ��

if h == 1
    %�����˲���
elseif h == 2
    %������˹�˲���
end

if type == 1
    %��ͨ
elseif type == 2
    %��ͨ
elseif type == 3
    %��ͨ
elseif type == 4
    %����
end

%% ����
I = f;
I = rgb2gray(I);
subplot(1,2,1),imshow(I),title('ԭʼ�Ҷ�ͼ��');
G=fft2(I);          %FFT               
G=fftshift(G);  % ת�����ݾ���
% imshow(uint8(abs(G/256)));
[M,N]=size(G);

w=40;                       % ����
nn=1;                       % 1�װ�����˹�˲���
d0=30;                      %��ֹƵ��Ϊ30
m=fix(M/2); n=fix(N/2);      %����(m,n)
%%
for i=1:M
       for j=1:N
           d=sqrt((i-m)^2+(j-n)^2);       %��(i,j)���˲������ĵľ��롣
           if(h==2&&type==1)
               if (d<d0)
                   h=0;
               else
                   h=1/(1+(d/d0)^(2*nn));% ���㴫�ݺ���
               end
           elseif(h==2&&type==2)
               if (d>d0)
                   h=0;
               else
                   h=1/(1+(d/d0)^(2*nn));% ���㴫�ݺ���
               end
           elseif(h==2&&type==3)
               if (d>d0)
                   h=0;
               else
                   h=1/(1+(d*w)/(d/d0)^(2*nn));% ���㴫�ݺ���
               end
           elseif(h==2&&type==4)
               if (d>d0)
                   h=0;
               else
                   h=1/(1+((d/d0)/(d*w))^(2*nn))% ���㴫�ݺ���
               end
           elseif(h==1&&type==1)
               if (d<=d0)
                   h=1;
               else
                   h=0;% ���㴫�ݺ���
               end
           elseif(h==1&&type==2)
               if (d<=d0)
                   h=0;
               else
                   h=1;% ���㴫�ݺ���
               end
           elseif(h==1&&type==3)
               if (20<d&&d<60)
                   h=1;
               else
                   h=0;% ���㴫�ݺ���
               end
           elseif(h==1&&type==4)
               if (d<20||d>60)
                   h=1;
               else
                   h=0;% ���㴫�ݺ���
               end
           end
           result(i,j)=h*G(i,j);
       end
end
%%
result=ifftshift(result);
Y2=ifft2(result);
Y3=uint8(real(Y2));
subplot(1,2,2),imshow(Y3),title('������˹�˲�')







end

