function [ f_new,f_new_freq ] = Filtering( f,h,type )
% f:待滤波图像
% h:滤波器种类
% type:滤波器类型

% f_new:滤波后的图像
% f_new_freq:滤波后图像频谱

if h == 1
    %理想滤波器
elseif h == 2
    %巴特沃斯滤波器
end

if type == 1
    %低通
elseif type == 2
    %高通
elseif type == 3
    %带通
elseif type == 4
    %带阻
end

%% 运算
I = f;
I = rgb2gray(I);
subplot(1,2,1),imshow(I),title('原始灰度图像');
G=fft2(I);          %FFT               
G=fftshift(G);  % 转换数据矩阵
% imshow(uint8(abs(G/256)));
[M,N]=size(G);

w=40;                       % 带宽
nn=1;                       % 1阶巴特沃斯滤波器
d0=30;                      %截止频率为30
m=fix(M/2); n=fix(N/2);      %中心(m,n)
%%
for i=1:M
       for j=1:N
           d=sqrt((i-m)^2+(j-n)^2);       %点(i,j)到滤波器中心的距离。
           if(h==2&&type==1)
               if (d<d0)
                   h=0;
               else
                   h=1/(1+(d/d0)^(2*nn));% 计算传递函数
               end
           elseif(h==2&&type==2)
               if (d>d0)
                   h=0;
               else
                   h=1/(1+(d/d0)^(2*nn));% 计算传递函数
               end
           elseif(h==2&&type==3)
               if (d>d0)
                   h=0;
               else
                   h=1/(1+(d*w)/(d/d0)^(2*nn));% 计算传递函数
               end
           elseif(h==2&&type==4)
               if (d>d0)
                   h=0;
               else
                   h=1/(1+((d/d0)/(d*w))^(2*nn))% 计算传递函数
               end
           elseif(h==1&&type==1)
               if (d<=d0)
                   h=1;
               else
                   h=0;% 计算传递函数
               end
           elseif(h==1&&type==2)
               if (d<=d0)
                   h=0;
               else
                   h=1;% 计算传递函数
               end
           elseif(h==1&&type==3)
               if (20<d&&d<60)
                   h=1;
               else
                   h=0;% 计算传递函数
               end
           elseif(h==1&&type==4)
               if (d<20||d>60)
                   h=1;
               else
                   h=0;% 计算传递函数
               end
           end
           result(i,j)=h*G(i,j);
       end
end
%%
result=ifftshift(result);
Y2=ifft2(result);
Y3=uint8(real(Y2));
subplot(1,2,2),imshow(Y3),title('巴特沃斯滤波')







end

