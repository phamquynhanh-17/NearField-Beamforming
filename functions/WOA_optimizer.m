function [Best_Score, Best_Pos, Convergence_curve] = WOA_optimizer(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    Best_Pos = zeros(1, dim);
    Best_Score = inf; 
    
    % Khởi tạo vị trí ngẫu nhiên
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    Convergence_curve = zeros(1, Max_iter);
    
    for t = 1:Max_iter
        for i = 1:size(Positions, 1)
            % Kiểm tra biên
            Positions(i,:) = max(min(Positions(i,:), ub), lb);
            fitness = fobj(Positions(i,:));
            
            if fitness < Best_Score
                Best_Score = fitness;
                Best_Pos = Positions(i,:);
            end
        end
        
        a = 2 - t * (2 / Max_iter); 
        for i = 1:size(Positions, 1)
            r1 = rand(); r2 = rand();
            A = 2 * a * r1 - a;
            C = 2 * r2;
            p = rand();
            l = (rand()*2) - 1;
            
            if p < 0.5
                if abs(A) < 1
                    D = abs(C * Best_Pos - Positions(i,:));
                    Positions(i,:) = Best_Pos - A * D;
                else
                    rand_idx = floor(SearchAgents_no * rand() + 1);
                    X_rand = Positions(rand_idx, :);
                    D = abs(C * X_rand - Positions(i,:));
                    Positions(i,:) = X_rand - A * D;
                end
            else
                % Tấn công hình xoắn ốc
                dist = abs(Best_Pos - Positions(i,:));
                Positions(i,:) = dist * exp(l) .* cos(l * 2 * pi) + Best_Pos;
            end
        end
        Convergence_curve(t) = Best_Score;
        fprintf('WOA Iteration %d: Best Rate = %.4f\n', t, abs(Best_Score));
    end
end