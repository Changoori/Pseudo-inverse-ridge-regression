clear; close all
% Data missing scenario
% 옵션: regression_mode = 'ols' | 'ridge' | 'lasso'
regression_mode = 'ridge';
lambda = 0.5;  % 정규화 강도

% 데이터 로드
% load th.txt  % loam 
% load data\data\sandy_loam\th.txt % sandy_loam
load data\data\silt_loam\th.txt % silt_loam
% load data\data\loam_homo\th.txt % loam_homo
% load data\data\loam_high\th.txt % loam_high
backupth=th;
trimdata=3;

switch trimdata
    case 1
        th=th(1:13001,1:102);
        time1 = [0:1/100:130];
    case 2
        th=th(1:3001,1:102);
        time1 = [0:1/100:30];
    case 3
        th=th(3001:13001,1:102);
        time1 = [30:1/100:130];
end

data=th(:,6:10:26)';
% data = data + 0.1 * std(data(:)) * randn(size(data));
imagesc(data)
xlabel('시간 (하루에 100번 측정)'); ylabel('Depth  1.0 m')
% The sampling temporal resolution is still set as 0.01 day

ll=[0.05:0.1:0.25]; 
[nz, nt] = size(data);

clear data_t data_z data_zz data_zzz
for i = 1:nz
    data_t(i,:) = gradient(data(i,:), time1);
end
for i = 1:nt
    data_z(:,i) = gradient(data(:,i), ll);
end
for i = 1:nt
    data_zz(:,i) = gradient(data_z(:,i), ll);
end
for i = 1:nt
    data_zzz(:,i) = gradient(data_zz(:,i), ll);
end

data_z2 = data_z.^2;
data_zz2 = data_zz.^2;
data_zzz2 = data_zzz.^2;
data_z_zz = data_z .* data_zz;
data_z_zzz = data_z .* data_zzz;
data_zz_zzz = data_zz .* data_zzz;

terms = {'1', 'theta_z', 'theta_zz', 'theta_zzz', 'theta_z^2',...
    'theta_zz^2', 'theta_zzz^2', 'theta_z*theta_zz',...
    'theta_z*theta_zzz', 'theta_zz*theta_zzz'};

% 전체 디자인 행렬과 목표 벡터 구성
A = [ones(nt*nz,1), data_z(:), data_zz(:), data_zzz(:), data_z2(:),...
    data_zz2(:), data_zzz2(:), data_z_zz(:), data_z_zzz(:), data_zz_zzz(:)];
B = data_t(:);
An = A ./ vecnorm(A);
Bn = B / norm(B);

% AIC 기반 후보 조합 탐색
num_terms = length(terms);
num_models = 2^num_terms;
Xi_list = [];
aic_list = [];

for m = 1:num_models
    term_mask = logical(dec2bin(m-1,num_terms)-'0');
    if sum(term_mask) == 0
        continue;
    end
    A_sub = An(:, term_mask);
    switch regression_mode
        case 'ols'
            Xi_sub = A_sub \ Bn;
        case 'ridge'
            Xi_sub = (A_sub' * A_sub + lambda * eye(sum(term_mask))) \ (A_sub' * Bn);
        case 'lasso'
            [Xi_tmp, ~] = lasso(A_sub, Bn, 'Lambda', lambda);
            Xi_sub = Xi_tmp;
        otherwise
            error('Unknown regression mode');
    end
    B_pred = A_sub * Xi_sub;
    residual = Bn - B_pred;
    rss = sum(residual.^2);
    k = sum(term_mask);
    n = length(Bn);
    aic = n*log(rss/n) + 2*k;
    Xi_full = zeros(num_terms,1);
    Xi_full(term_mask) = Xi_sub;
    Xi_list = [Xi_list, Xi_full];
    aic_list = [aic_list, aic];
end

% AIC 최소 모델 선택
[~, best_idx] = min(aic_list);
Xi = Xi_list(:, best_idx);
Xi = Xi .* (norm(B) ./ vecnorm(A)');

% 결과 출력
fprintf('\nFinal equation by AIC (%s regression):\n', upper(regression_mode))
equation = 'theta_t = ';
first = true;
for i = 1:length(Xi)
    if abs(Xi(i)) > 1e-8
        if first
            equation = [equation, sprintf('%.4e * %s', Xi(i), terms{i})];
            first = false;
        else
            equation = [equation, sprintf(' + %.4e * %s', Xi(i), terms{i})];
        end
    end
end

disp(equation)
