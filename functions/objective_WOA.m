function fitness = objective_WOA(x, para, H)
    M = para.M; % 512
    K = para.K; % 4
    N_sub = size(H, 3);
    num_el = M * K;
    
    % 1. Khôi phục ma trận phức W (M x K)
    W_real = reshape(x(1 : num_el), M, K);
    W_imag = reshape(x(num_el + 1 : 2*num_el), M, K);
    W = W_real + 1i * W_imag;
    
    % 2. Chuẩn hóa công suất
    P_limit = para.P_total;
    p_curr = norm(W, 'fro')^2;
    if p_curr > P_limit
        W = W * sqrt(P_limit / p_curr);
    end

    R_sum = 0;
    for n = 1:N_sub
        H_n = H(:,:,n); 
        
        % 3. Kiểm tra và thực hiện phép nhân ma trận
        % Mục tiêu: Tạo ra ma trận (K x K) để tính định thức
        [rows, cols] = size(H_n);
        
        if rows == M && cols == K
            % H_n là (512 x 4), W là (512 x 4)
            % Phép nhân đúng: (4x512) * (512x4) = (4x4)
            term = eye(K) + (H_n' * W * W' * H_n);
        elseif rows == K && cols == M
            % H_n là (4 x 512), W là (512 x 4)
            % Phép nhân đúng: (4x512) * (512x4) = (4x4)
            term = eye(K) + (H_n * W * W' * H_n');
        else
            % Trường hợp kích thước không xác định, bỏ qua vòng lặp này
            continue;
        end
        
        % 4. Tính toán log2(det) an toàn bằng trị riêng
        e = real(eig(term)); 
        e(e < 1e-12) = 1e-12; 
        R_sum = R_sum + sum(log2(e));
    end
    
    fitness = -(R_sum / N_sub); 
    if isnan(fitness), fitness = 0; end
end