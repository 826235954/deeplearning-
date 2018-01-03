function [ output ] = filtering2( input, filter,n)

% 输入参数:
% input:原始图像;
% 均值滤波:filter=1
% 中值滤波:filter=2
% 拉普拉斯滤波:filter=3


%初始参数
I = input;
I = rgb2gray(I);%转换为灰度图像
x1 = double(I);
x2 = x1;
[height, width] = size(I);

%中值滤波参数
% if filter==2
%     n = input('请输入算子大小：3 or 5'); % 算子大小
% else
% %均值滤波参数
%     n = 3;
%     template = ones(n);
% end
template = ones(n);
if filter~=3
    for i = 1:height-n+1
        for j = 1:width-n+1
            if filter==2
                % 中值滤波
                c = x1(i:i+n-1,j:j+n-1);
                e = c(1,:);
                for k = 2:n
                    e = [e, c(k, :)];
                end
                tmp = median(e);
                x2(i+(n-1)/2,j+(n-1)/2) = tmp;
            end
            if filter==1
                % 均值滤波
                c = x1(i:i+n-1,j:j+n-1).*template;
                s = sum(sum(c));
                x2(i+(n-1)/2,j+(n-1)/2) = s/(n*n);
            end
        end
    end
else
    % 拉普拉斯滤波
    for i = 2:height-1
        for j = 2:width-1
            x2(i,j)=I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)-4*I(i,j);
            %             g(x,y)=[f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)]-4f(x,y);%四邻接
            %             h=[  0  -1  0; -1   4  -1;0  -1  0;]
            %             g(x,y)=8f(x,y)-[f(x-1,y-1)+f(x-1,y)+f(x-1,y+1)+f(x,y-1)+f(x,y+1)+f(x+1,y-1)+f(x+1,y)+f(x+1,y+1)]
            %             h=[  -1  -1  -1; -1   8  -1;-1  -1  -1;]
        end
    end
    x2 = x1-x2;
end

%输出
output = uint8(x2);
% imshow(output);



% I = imread('E:\lena.bmp');
% I = rgb2gray(I);%转换为灰度图像
% % imshow(I);
% J = imnoise(I,'gauss',0.02);              %添加高斯噪声
% J = imnoise(I,'salt & pepper',0.02);       %添加椒盐噪声
% % ave1=fspecial('average',3);              %产生3×3的均值模版
% % ave2=fspecial('average',5);              %产生5×5的均值模版
% % K = filter2(ave1,J)/255;                 %均值滤波3×3
% % L = filter2(ave2,J)/255;                 %均值滤波5×5
% M  =  medfilt2(J,[3  3]);                  %中值滤波3×3模板
% N  =  medfilt2(J,[5  5]);                   %中值滤波5×5模板
% imshow(x1);


end




