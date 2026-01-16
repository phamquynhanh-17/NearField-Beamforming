clc; clear all; close all;
addpath("functions/");
set(groot,'defaultAxesTickLabelInterpreter','latex');

%% 1. Parameters
para = para_init();
para.P_total = 1; 
user_r = rand(para.K, 1) * 10 + 5; 
user_theta = sort(rand(para.K, 1) * pi); 

%% 2. Generate channel matrix
[H] = generate_channel(para, user_r, user_theta);

%% 3. Fully-digital approximation (FDA) approach (Đề xuất)
fprintf('--- Đang chạy thuật toán đề xuất FDA... ---\n');
[R_convergence, penalty_convergence, A, D] = algorithm_FDA_penalty(para, H, user_r, user_theta);
R_FDA = R_convergence(end);

%% 4. Whale Optimization Algorithm (WOA) - Cải tiến dải tìm kiếm
% Tăng số lượng cá thể và vòng lặp để WOA có cơ hội tìm thấy kết quả
SearchAgents_no = 40;            
Max_iter = length(R_convergence); 
dim = 2 * para.M * para.K;       

% Thu hẹp lb/ub để WOA không bị loãng trong không gian quá rộng
lb = -0.1;                       
ub = 0.1;                        

fobj = @(x) objective_WOA(x, para, H); 
fprintf('\n--- Đang chạy WOA (Cấu hình mạnh hơn), vui lòng đợi... ---\n');
[Best_Score, Best_Pos, WOA_conv_curve] = WOA_optimizer(SearchAgents_no, Max_iter, lb, ub, dim, fobj);

R_WOA_convergence = abs(WOA_conv_curve); 

%% 5. Plotting Results (Sửa lỗi hiển thị)
figure('Color', [1 1 1], 'Position', [100, 100, 1000, 400]);

subplot(1,2,1); 
plot(R_convergence, '-b', 'LineWidth', 2); hold on;
plot(R_WOA_convergence, '--r', 'LineWidth', 2); 
xlabel('Number of outer-loop iterations', 'Interpreter', 'Latex');
ylabel('Spectral efficiency (bit/s/Hz)', 'Interpreter', 'Latex');
legend('Proposed FDA', 'WOA Algorithm', 'Location', 'SouthEast');
title('\textbf{Convergence Comparison}', 'Interpreter', 'Latex');
grid on;

% Tự động điều chỉnh trục Y để thấy rõ cả 2 đường nếu khoảng cách quá xa
ylim([0, max(R_convergence)*1.2]); 

subplot(1,2,2);
semilogy(penalty_convergence, '-b', 'LineWidth', 1.5);
xlabel('Number of outer-loop iterations', 'Interpreter', 'Latex');
ylabel('Penalty value', 'Interpreter', 'Latex');
title('\textbf{Constraint Satisfaction (FDA)}', 'Interpreter', 'Latex');
grid on;

fprintf('\nFinal Rate FDA: %.4f bit/s/Hz\n', R_FDA);
fprintf('Final Rate WOA: %.4f bit/s/Hz\n', R_WOA_convergence(end));