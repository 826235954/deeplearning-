function [ output ] = filtering2( input, filter,n)

% �������:
% input:ԭʼͼ��;
% ��ֵ�˲�:filter=1
% ��ֵ�˲�:filter=2
% ������˹�˲�:filter=3


%��ʼ����
I = input;
I = rgb2gray(I);%ת��Ϊ�Ҷ�ͼ��
x1 = double(I);
x2 = x1;
[height, width] = size(I);

%��ֵ�˲�����
% if filter==2
%     n = input('���������Ӵ�С��3 or 5'); % ���Ӵ�С
% else
% %��ֵ�˲�����
%     n = 3;
%     template = ones(n);
% end
template = ones(n);
if filter~=3
    for i = 1:height-n+1
        for j = 1:width-n+1
            if filter==2
                % ��ֵ�˲�
                c = x1(i:i+n-1,j:j+n-1);
                e = c(1,:);
                for k = 2:n
                    e = [e, c(k, :)];
                end
                tmp = median(e);
                x2(i+(n-1)/2,j+(n-1)/2) = tmp;
            end
            if filter==1
                % ��ֵ�˲�
                c = x1(i:i+n-1,j:j+n-1).*template;
                s = sum(sum(c));
                x2(i+(n-1)/2,j+(n-1)/2) = s/(n*n);
            end
        end
    end
else
    % ������˹�˲�
    for i = 2:height-1
        for j = 2:width-1
            x2(i,j)=I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)-4*I(i,j);
            %             g(x,y)=[f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)]-4f(x,y);%���ڽ�
            %             h=[  0  -1  0; -1   4  -1;0  -1  0;]
            %             g(x,y)=8f(x,y)-[f(x-1,y-1)+f(x-1,y)+f(x-1,y+1)+f(x,y-1)+f(x,y+1)+f(x+1,y-1)+f(x+1,y)+f(x+1,y+1)]
            %             h=[  -1  -1  -1; -1   8  -1;-1  -1  -1;]
        end
    end
    x2 = x1-x2;
end

%���
output = uint8(x2);
% imshow(output);



% I = imread('E:\lena.bmp');
% I = rgb2gray(I);%ת��Ϊ�Ҷ�ͼ��
% % imshow(I);
% J = imnoise(I,'gauss',0.02);              %��Ӹ�˹����
% J = imnoise(I,'salt & pepper',0.02);       %��ӽ�������
% % ave1=fspecial('average',3);              %����3��3�ľ�ֵģ��
% % ave2=fspecial('average',5);              %����5��5�ľ�ֵģ��
% % K = filter2(ave1,J)/255;                 %��ֵ�˲�3��3
% % L = filter2(ave2,J)/255;                 %��ֵ�˲�5��5
% M  =  medfilt2(J,[3  3]);                  %��ֵ�˲�3��3ģ��
% N  =  medfilt2(J,[5  5]);                   %��ֵ�˲�5��5ģ��
% imshow(x1);


end




