% WENO code for ideal MHD

% Nov. 27, 2023
% Jung Min Ki, Lee Sang-Min
% Topics in Advanced Scientific Computation 1
% Final Project

% Reference paper :
% Guang-Shan Jiang and Cheng-chin Wu, "A High-Order WENO Finite Difference
% Scheme for the Equations of Ideal Magnetohydrodynamics" Journal of
% Computational Physics 150, 561-594 (1999)

clc;
clear;
close all;

%% 3.4 Orszag-Tang MHD Turbulence Problem

% get initial conditions for the 2D Orszag-Tang MHD Turbulence Problem
GAMMA = 5./3.;
[x, dx, state] = getInitialCondition(GAMMA);
U = convertStateToU(state, GAMMA);

% time stepping iteration
t = 0.;
t_end = 3.0;
CFL_number = 0.8;
step_counter = 0;
tic
while t < t_end
    % evaluate CFL condition
    dt = getCFL(U, dx, CFL_number, GAMMA);
    if t + dt > t_end
        dt = t_end - t;
    end

    % 4th-order non-TVD Runge-Kutta scheme (p. 570)
    U_0   = U;
    L_U_0 = getL_U(U_0, dx, GAMMA);
    U_1   = U_0 + dt / 2. * L_U_0;
    L_U_1 = getL_U(U_1, dx, GAMMA);
    U_2   = U_1 + dt / 2. * (-L_U_0 + L_U_1);
    L_U_2 = getL_U(U_2, dx, GAMMA);
    U_3   = U_2 + dt / 2. * (-L_U_1 + 2. * L_U_2);
    L_U_3 = getL_U(U_3, dx, GAMMA);
    U_4   = U_3 + dt / 6. * (L_U_0 + 2. * L_U_1 - 4. * L_U_2 + L_U_3);
    U     = U_4;

    % advance time
    t = t + dt;
    step_counter = step_counter + 1;
    fprintf("[step #%d] t = %.3e [dt = %.2e]\n", step_counter, t, dt);

end
toc

% visualize the final results
state = convertUToState(U, GAMMA);
rho = squeeze(state(1, :, :));
v_x = squeeze(state(2, :, :));
v_y = squeeze(state(3, :, :));
B_x = squeeze(state(5, :, :));
B_y = squeeze(state(6, :, :));
p   = squeeze(state(8, :, :));

% visualize the initialization
figure();
sgtitle('Simulation results for the 2D Orszag-Tang MHD Turbulence problem');
subplot(2, 2, 1); contourf(squeeze(x(1, :, :)), squeeze(x(2, :, :)), rho);    title('$\rho$', 'Interpreter', 'latex');
subplot(2, 2, 2); contourf(squeeze(x(1, :, :)), squeeze(x(2, :, :)), p);      title('$p$', 'Interpreter', 'latex');
subplot(2, 2, 3); quiver(squeeze(x(1, :, :)), squeeze(x(2, :, :)), v_x, v_y); title('$v_{xy}$', 'Interpreter', 'latex');
subplot(2, 2, 4); quiver(squeeze(x(1, :, :)), squeeze(x(2, :, :)), B_x, B_y); title('$B_{xy}$', 'Interpreter', 'latex');

save('state2D.mat', 'state');

%% Function implementations

% Initial conditions for [3.4 Orszag-Tang MHD Turbulence Problem]
function [x, dx, state] = getInitialCondition(GAMMA)
    % spatial resolution setup
    grid_count = [192 + 1, 192 + 1];
    x_min = [0., 0.];
    x_max = [2. * pi, 2. * pi];
    dx = [(x_max(1) - x_min(1)) / (grid_count(1) - 1), ...
        (x_max(2) - x_min(2)) / (grid_count(2) - 1)];

    % define arrays for each variable
    state = zeros(8, grid_count(1), grid_count(2));   % state vector/matrix
    x     = zeros(2, grid_count(1), grid_count(2));   % x-y coordinate
    rho   = zeros(grid_count);         % density
    v_x   = zeros(grid_count);         % velocity in x direction
    v_y   = zeros(grid_count);         % velocity in y direction
    v_z   = zeros(grid_count);         % velocity in z direction
    B_x   = zeros(grid_count);         % magnetic field in x direction
    B_y   = zeros(grid_count);         % magnetic field in y direction
    B_z   = zeros(grid_count);         % magnetic field in z direction
    p     = zeros(grid_count);         % pressure

    % fill in initial values
    for i = 1:grid_count(1)
        for j = 1:grid_count(2)
            x(1, i, j) = (i - 1) * dx(1) + x_min(1);
            x(2, i, j) = (j - 1) * dx(2) + x_min(2);

            rho(i, j) = GAMMA^2;
            v_x(i, j) = -sin(x(2, i, j));      % -sin(y)
            v_y(i, j) = +sin(x(1, i, j));      % +sin(x)
            v_z(i, j) = 0.;
            B_x(i, j) = -sin(x(2, i, j));      % -sin(y)
            B_y(i, j) = +sin(2. * x(1, i, j)); % +sin(2x)
            B_z(i, j) = 0.;
            p(i, j)   = GAMMA;
        end
    end
  
    % package each variable into the state vector/matrix
    state(1, :, :) = rho;
    state(2, :, :) = v_x;
    state(3, :, :) = v_y;
    state(4, :, :) = v_z;
    state(5, :, :) = B_x;
    state(6, :, :) = B_y;
    state(7, :, :) = B_z;
    state(8, :, :) = p;
end

% convert physical state variables to eigen-system variables (U)
% e.g. vx -> rho * vx
function U = convertStateToU(state, GAMMA)

    U = zeros(size(state));
    rho = state(1, :, :);
    v_x = state(2, :, :);
    v_y = state(3, :, :);
    v_z = state(4, :, :);
    B_x = state(5, :, :);
    B_y = state(6, :, :);
    B_z = state(7, :, :);
    p   = state(8, :, :);

    e = rho .* (v_x.^2 + v_y.^2 + v_z.^2) / 2. ...
        + (B_x.^2 + B_y.^2 + B_z.^2) / 2. + p / (GAMMA - 1.);

    U(1, :, :) = rho;
    U(2, :, :) = rho .* v_x;
    U(3, :, :) = rho .* v_y;
    U(4, :, :) = rho .* v_z;
    U(5, :, :) = B_x;
    U(6, :, :) = B_y;
    U(7, :, :) = B_z;
    U(8, :, :) = e;
   
end

% convert eigen-system variables (U) to physical state variables
% e.g. rho * vx -> vx
function state = convertUToState(U, GAMMA)

    state   = zeros(size(U));
    rho     = U(1, :, :);
    rho_v_x = U(2, :, :);
    rho_v_y = U(3, :, :);
    rho_v_z = U(4, :, :);
    B_x     = U(5, :, :);
    B_y     = U(6, :, :);
    B_z     = U(7, :, :);
    e       = U(8, :, :);

    v_x = rho_v_x ./ rho;
    v_y = rho_v_y ./ rho;
    v_z = rho_v_z ./ rho;
    p  = (GAMMA - 1.) .* (e - rho .* (v_x.^2 + v_y.^2 + v_z.^2) / 2. ...
        - (B_x.^2 + B_y.^2 + B_z.^2) / 2.);

    state(1, :, :) = rho;
    state(2, :, :) = v_x;
    state(3, :, :) = v_y;
    state(4, :, :) = v_z;
    state(5, :, :) = B_x;
    state(6, :, :) = B_y;
    state(7, :, :) = B_z;
    state(8, :, :) = p;

end

% CFL condition evaluation
function dt = getCFL(U, dx, CFL_number, GAMMA)
    state = convertUToState(U, GAMMA);
    v_x = abs(state(2, :, :));
    v_y = abs(state(3, :, :));
    [c_f_x, c_f_y] = getCf(state, GAMMA);
    dt = CFL_number / (max(v_x + c_f_x, [], "all") / dx(1) + max(v_y + c_f_y, [], "all") / dx(2)); % Eqn. (3.2)
end

function [c_f_x, c_f_y] = getCf(state, GAMMA)

    rho = squeeze(state(1, :, :));
    B_x = squeeze(state(5, :, :));
    B_y = squeeze(state(6, :, :));
    B_z = squeeze(state(7, :, :));
    p   = squeeze(state(8, :, :));

    grid_count = size(state, 2:3);

    c_f_x = zeros(grid_count);
    c_f_y = zeros(grid_count);

    for i = 1:grid_count(1)
        for j = 1:grid_count(2)
            rho_i = rho(i, j);
            B_x_i = B_x(i, j);
            B_y_i = B_y(i, j);
            B_z_i = B_z(i, j);
            p_i   = p(i, j);
        
            b_x = B_x_i / sqrt(rho_i);
            b_y = B_y_i / sqrt(rho_i);
            b_z = B_z_i / sqrt(rho_i);
            b   = sqrt(b_x^2 + b_y^2 + b_z^2);
            a   = sqrt(GAMMA * p_i / rho_i);
            c_f_x(i, j) = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
            c_f_y(i, j) = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_y^2)) / 2.);
        end
    end
end

% WENO solver
function L_U = getL_U(U, dx, GAMMA)
    grid_count = size(U, 2:3);
    L_U = zeros(size(U));
    for i = 1:grid_count(1)
        for j = 1:grid_count(2)
            [F_hat_right, F_hat_left] = getF_hat(U, i, j, GAMMA);
            [G_hat_right, G_hat_left] = getG_hat(U, i, j, GAMMA);
            L_U(:, i, j) = -(F_hat_right - F_hat_left) / dx(1) - (G_hat_right - G_hat_left) / dx(2); % Eqn. (2.19)
        end
    end
end

% WENO flux function
function [F_hat_right, F_hat_left] = getF_hat(U, i, j, GAMMA)

    grid_count = size(U, 2:3);
    U_padded = zeros(8, grid_count(1) + 6); % boundary padding (3 on each side)
    for k = 1:grid_count(1)
        U_padded(:, k + 3) = U(:, k, j); % copy each value
    end
    % periodic boundary condition
    for k = 1:3
        U_padded(:, k) = U(:, grid_count(1) + k - 3, j); % copy boundary value (left)
        U_padded(:, k + grid_count(1) + 3) = U(:, k, j); % copy boundary value (right)
    end

    state = convertUToState(U_padded, GAMMA);
    F_U = getF(U_padded, GAMMA);

    alpha_s = getMaxEigenvalueX(state, GAMMA); % global Lax-Friedrichs flux splitting

    % solve for F_hat_right
    [L, R] = getLReigenvectorX(state, i + 3, GAMMA);

    F_i = zeros(8, 4);
    for s = 1:8
        for k = 1:4
            F_i(:, k) = F_i(:, k) + dot(L(s, :), F_U(:, i+k+1)) .* R(s, :)'; % Eqn. (2.16)
        end
    end
    F_hat_right = (-F_i(:, 1) + 7. * F_i(:, 2) + 7. * F_i(:, 3) - F_i(:, 4)) / 12.; % Eqn. (2.20)
    for s = 1:8
        F_i_s_plus = zeros(5, 1);
        for k = 1:5
            F_i_s_plus(k) = (dot(L(s, :), F_U(:, i+k)) + alpha_s(s) * dot(L(s, :), U_padded(:, i+k))) / 2.;
        end
        Phi_N_plus = getPhiN(F_i_s_plus(2) - F_i_s_plus(1), F_i_s_plus(3) - F_i_s_plus(2), ...
            F_i_s_plus(4) - F_i_s_plus(3), F_i_s_plus(5) - F_i_s_plus(4));
        F_i_s_minus = zeros(5, 1);
        for k = 1:5
            F_i_s_minus(6-k) = (dot(L(s, :), F_U(:, i+k+1)) - alpha_s(s) * dot(L(s, :), U_padded(:, i+k+1))) / 2.;
        end
        Phi_N_minus = getPhiN(F_i_s_minus(1) - F_i_s_minus(2), F_i_s_minus(2) - F_i_s_minus(3), ...
            F_i_s_minus(3) - F_i_s_minus(4), F_i_s_minus(4) - F_i_s_minus(5));
        F_hat_right = F_hat_right + (-Phi_N_plus + Phi_N_minus) .* R(s, :)'; % Eqn. (2.20)
    end

    % solve for F_hat_left
    [L, R] = getLReigenvectorX(state, i + 2, GAMMA);
    F_i = zeros(8, 4);
    for s = 1:8
        for k = 1:4
            F_i(:, k) = F_i(:, k) + dot(L(s, :), F_U(:, i+k)) .* R(s, :)'; % Eqn. (2.16)
        end
    end
    F_hat_left = (-F_i(:, 1) + 7. * F_i(:, 2) + 7. * F_i(:, 3) - F_i(:, 4)) / 12.; % Eqn. (2.20)
    for s = 1:8
        F_i_s_plus = zeros(5, 1);
        for k = 1:5
            F_i_s_plus(k) = (dot(L(s, :), F_U(:, i+k-1)) + alpha_s(s) * dot(L(s, :), U_padded(:, i+k-1))) / 2.;
        end
        Phi_N_plus = getPhiN(F_i_s_plus(2) - F_i_s_plus(1), F_i_s_plus(3) - F_i_s_plus(2), ...
            F_i_s_plus(4) - F_i_s_plus(3), F_i_s_plus(5) - F_i_s_plus(4));
        F_i_s_minus = zeros(5, 1);
        for k = 1:5
            F_i_s_minus(6-k) = (dot(L(s, :), F_U(:, i+k)) - alpha_s(s) * dot(L(s, :), U_padded(:, i+k))) / 2.;
        end
        Phi_N_minus = getPhiN(F_i_s_minus(1) - F_i_s_minus(2), F_i_s_minus(2) - F_i_s_minus(3), ...
            F_i_s_minus(3) - F_i_s_minus(4), F_i_s_minus(4) - F_i_s_minus(5));
        F_hat_left = F_hat_left + (-Phi_N_plus + Phi_N_minus) .* R(s, :)'; % Eqn. (2.20)
    end
end

function [G_hat_right, G_hat_left] = getG_hat(U, i, j, GAMMA)

    grid_count = size(U, 2:3);
    U_padded = zeros(8, grid_count(2) + 6); % boundary padding (3 on each side)
    for k = 1:grid_count(2)
        U_padded(:, k + 3) = U(:, i, k); % copy each value
    end
    % periodic boundary condition
    for k = 1:3
        U_padded(:, k) = U(:, i, grid_count(2) + k - 3); % copy boundary value (left)
        U_padded(:, k + grid_count(2) + 3) = U(:, i, k); % copy boundary value (right)
    end

    state = convertUToState(U_padded, GAMMA);
    G_U = getG(U_padded, GAMMA);

    alpha_s = getMaxEigenvalueY(state, GAMMA); % global Lax-Friedrichs flux splitting

    % solve for F_hat_right
    [L, R] = getLReigenvectorY(state, j + 3, GAMMA);

    G_i = zeros(8, 4);
    for s = 1:8
        for k = 1:4
            G_i(:, k) = G_i(:, k) + dot(L(s, :), G_U(:, j+k+1)) .* R(s, :)'; % Eqn. (2.16)
        end
    end
    G_hat_right = (-G_i(:, 1) + 7. * G_i(:, 2) + 7. * G_i(:, 3) - G_i(:, 4)) / 12.; % Eqn. (2.21)
    for s = 1:8
        G_i_s_plus = zeros(5, 1);
        for k = 1:5
            G_i_s_plus(k) = (dot(L(s, :), G_U(:, j+k)) + alpha_s(s) * dot(L(s, :), U_padded(:, j+k))) / 2.;
        end
        Phi_N_plus = getPhiN(G_i_s_plus(2) - G_i_s_plus(1), G_i_s_plus(3) - G_i_s_plus(2), ...
            G_i_s_plus(4) - G_i_s_plus(3), G_i_s_plus(5) - G_i_s_plus(4));
        G_i_s_minus = zeros(5, 1);
        for k = 1:5
            G_i_s_minus(6-k) = (dot(L(s, :), G_U(:, j+k+1)) - alpha_s(s) * dot(L(s, :), U_padded(:, j+k+1))) / 2.;
        end
        Phi_N_minus = getPhiN(G_i_s_minus(1) - G_i_s_minus(2), G_i_s_minus(2) - G_i_s_minus(3), ...
            G_i_s_minus(3) - G_i_s_minus(4), G_i_s_minus(4) - G_i_s_minus(5));
        G_hat_right = G_hat_right + (-Phi_N_plus + Phi_N_minus) .* R(s, :)'; % Eqn. (2.21)
    end

    % solve for F_hat_left
    [L, R] = getLReigenvectorY(state, j + 2, GAMMA);

    G_i = zeros(8, 4);
    for s = 1:8
        for k = 1:4
            G_i(:, k) = G_i(:, k) + dot(L(s, :), G_U(:, j+k)) .* R(s, :)'; % Eqn. (2.16)
        end
    end
    G_hat_left = (-G_i(:, 1) + 7. * G_i(:, 2) + 7. * G_i(:, 3) - G_i(:, 4)) / 12.; % Eqn. (2.21)
    for s = 1:8
        G_i_s_plus = zeros(5, 1);
        for k = 1:5
            G_i_s_plus(k) = (dot(L(s, :), G_U(:, j+k-1)) + alpha_s(s) * dot(L(s, :), U_padded(:, j+k-1))) / 2.;
        end
        Phi_N_plus = getPhiN(G_i_s_plus(2) - G_i_s_plus(1), G_i_s_plus(3) - G_i_s_plus(2), ...
            G_i_s_plus(4) - G_i_s_plus(3), G_i_s_plus(5) - G_i_s_plus(4));
        G_i_s_minus = zeros(5, 1);
        for k = 1:5
            G_i_s_minus(6-k) = (dot(L(s, :), G_U(:, j+k)) - alpha_s(s) * dot(L(s, :), U_padded(:, j+k))) / 2.;
        end
        Phi_N_minus = getPhiN(G_i_s_minus(1) - G_i_s_minus(2), G_i_s_minus(2) - G_i_s_minus(3), ...
            G_i_s_minus(3) - G_i_s_minus(4), G_i_s_minus(4) - G_i_s_minus(5));
        G_hat_left = G_hat_left + (-Phi_N_plus + Phi_N_minus) .* R(s, :)'; % Eqn. (2.21)
    end
end

function alpha_s = getMaxEigenvalueX(state, GAMMA)

    rho = state(1, :); v_x = state(2, :);
    B_x = state(5, :); B_y  = state(6, :); B_z = state(7, :); p = state(8, :);

    grid_count = size(state, 2);

    alpha_s = zeros(8, 1);

    for i = 1:grid_count-1
        rho_i = (rho(i) + rho(i + 1)) / 2.;
        v_x_i = (v_x(i) + v_x(i + 1)) / 2.;
        B_x_i = (B_x(i) + B_x(i + 1)) / 2.;
        B_y_i = (B_y(i) + B_y(i + 1)) / 2.;
        B_z_i = (B_z(i) + B_z(i + 1)) / 2.;
        p_i   = (p(i) + p(i + 1)) / 2.;
    
        b_x = B_x_i / sqrt(rho_i);
        b_y = B_y_i / sqrt(rho_i);
        b_z = B_z_i / sqrt(rho_i);
        b   = sqrt(b_x^2 + b_y^2 + b_z^2);
        a   = sqrt(GAMMA * p_i / rho_i);
        c_a = abs(b_x);
        c_f = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
        c_s = sqrt((a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
        if (a^2 + b^2)^2 - 4. * a^2 * b_x^2 < 0.
            c_f = sqrt((a^2 + b^2) / 2.);
            c_s = sqrt((a^2 + b^2) / 2.);
        elseif a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2) < 0.
            c_s = 0.;
        end

        if abs(v_x_i - c_f) >= alpha_s(1)
            alpha_s(1) = abs(v_x_i - c_f);
        end
        if abs(v_x_i + c_f) >= alpha_s(8)
            alpha_s(8) = abs(v_x_i + c_f);
        end
        if abs(v_x_i - c_a) >= alpha_s(2)
            alpha_s(2) = abs(v_x_i - c_a);
        end
        if abs(v_x_i + c_a) >= alpha_s(7)
            alpha_s(7) = abs(v_x_i + c_a);
        end
        if abs(v_x_i - c_s) >= alpha_s(3)
            alpha_s(3) = abs(v_x_i - c_s);
        end
        if abs(v_x_i + c_s) >= alpha_s(6)
            alpha_s(6) = abs(v_x_i + c_s);
        end
        if abs(v_x_i) >= alpha_s(4)
            alpha_s(4) = abs(v_x_i);
        end
        if abs(v_x_i) >= alpha_s(5)
            alpha_s(5) = abs(v_x_i);
        end
    end
end

function alpha_s = getMaxEigenvalueY(state, GAMMA)

    rho = state(1, :); v_y = state(3, :);
    B_x = state(5, :); B_y  = state(6, :); B_z = state(7, :); p = state(8, :);

    grid_count = size(state, 2);

    alpha_s = zeros(8, 1);

    for i = 1:grid_count-1
        rho_i = (rho(i) + rho(i + 1)) / 2.;
        v_y_i = (v_y(i) + v_y(i + 1)) / 2.;
        B_x_i = (B_x(i) + B_x(i + 1)) / 2.;
        B_y_i = (B_y(i) + B_y(i + 1)) / 2.;
        B_z_i = (B_z(i) + B_z(i + 1)) / 2.;
        p_i   = (p(i) + p(i + 1)) / 2.;
    
        b_x = B_x_i / sqrt(rho_i);
        b_y = B_y_i / sqrt(rho_i);
        b_z = B_z_i / sqrt(rho_i);
        b   = sqrt(b_x^2 + b_y^2 + b_z^2);
        a   = sqrt(GAMMA * p_i / rho_i);
        c_a = abs(b_y);
        c_f = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_y^2)) / 2.);
        c_s = sqrt((a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_y^2)) / 2.);
        if (a^2 + b^2)^2 - 4. * a^2 * b_y^2 < 0.
            c_f = sqrt((a^2 + b^2) / 2.);
            c_s = sqrt((a^2 + b^2) / 2.);
        elseif a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_y^2) < 0.
            c_s = 0.;
        end

        if abs(v_y_i - c_f) >= alpha_s(1)
            alpha_s(1) = abs(v_y_i - c_f);
        end
        if abs(v_y_i + c_f) >= alpha_s(8)
            alpha_s(8) = abs(v_y_i + c_f);
        end
        if abs(v_y_i - c_a) >= alpha_s(2)
            alpha_s(2) = abs(v_y_i - c_a);
        end
        if abs(v_y_i + c_a) >= alpha_s(7)
            alpha_s(7) = abs(v_y_i + c_a);
        end
        if abs(v_y_i - c_s) >= alpha_s(3)
            alpha_s(3) = abs(v_y_i - c_s);
        end
        if abs(v_y_i + c_s) >= alpha_s(6)
            alpha_s(6) = abs(v_y_i + c_s);
        end
        if abs(v_y_i) >= alpha_s(4)
            alpha_s(4) = abs(v_y_i);
        end
        if abs(v_y_i) >= alpha_s(5)
            alpha_s(5) = abs(v_y_i);
        end
    end
end

function [L, R] = getLReigenvectorX(state, i, GAMMA)
    delta = 1e-12;

    rho = state(1, :); v_x = state(2, :); v_y = state(3, :); v_z = state(4, :);
    B_x = state(5, :); B_y = state(6, :); B_z = state(7, :); p   = state(8, :);

    v = sqrt(v_x.^2 + v_y.^2 + v_z.^2);
    B = sqrt(B_x.^2 + B_y.^2 + B_z.^2);

    rho = (rho(i) + rho(i + 1)) / 2.;
    v_x = (v_x(i) + v_x(i + 1)) / 2.;
    v_y = (v_y(i) + v_y(i + 1)) / 2.;
    v_z = (v_z(i) + v_z(i + 1)) / 2.;
    v   = (v(i)   + v(i + 1))   / 2.;
    B_x = (B_x(i) + B_x(i + 1)) / 2.;
    B_y = (B_y(i) + B_y(i + 1)) / 2.;
    B_z = (B_z(i) + B_z(i + 1)) / 2.;
    B   = (B(i)   + B(i + 1))   / 2.;
    p   = (p(i)   + p(i + 1))   / 2.;

    b_x = B_x / sqrt(rho);
    b_y = B_y / sqrt(rho);
    b_z = B_z / sqrt(rho);
    b   = sqrt(b_x^2 + b_y^2 + b_z^2);
    a   = sqrt(GAMMA * p / rho);
    c_f = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
    c_s = sqrt((a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.); 
    if (a^2 + b^2)^2 - 4. * a^2 * b_x^2 < 0.
        c_f = sqrt((a^2 + b^2) / 2.);
        c_s = sqrt((a^2 + b^2) / 2.);
    elseif a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2) < 0.
        c_s = 0.;
    end

    if B_y^2 + B_z^2 > delta * B^2
        beta_y = B_y / sqrt(B_y^2 + B_z^2);
        beta_z = B_z / sqrt(B_y^2 + B_z^2);
    else
        beta_y = 1. / sqrt(2);
        beta_z = 1. / sqrt(2);
    end
    if B_y^2 + B_z^2 > delta * B^2 || GAMMA * p - B_x^2 > delta * GAMMA * p
        alpha_f = sqrt(a^2 - c_s^2) / sqrt(c_f^2 - c_s^2);
        alpha_s = sqrt(c_f^2 - a^2) / sqrt(c_f^2 - c_s^2);
        if a^2 - c_s^2 < 0.
            alpha_f = 0.;
        end
        if c_f^2 - a^2 < 0.
            alpha_s = 0.;
        end
    else
        alpha_f = 1. / sqrt(2.);
        alpha_s = 1. / sqrt(2.);
    end
    if B_x >= 0.
        sgn_B_x = +1.;
    else
        sgn_B_x = -1.;
    end
    gamma_1 = (GAMMA - 1.) / 2.;
    gamma_2 = (GAMMA - 2.) / (GAMMA - 1.);
    tau     = (GAMMA - 1.) / a^2;
    Gamma_f = alpha_f * c_f * v_x - alpha_s * c_s * sgn_B_x * (beta_y * v_y + beta_z * v_z);
    Gamma_a = sgn_B_x * (beta_z * v_y - beta_y * v_z);
    Gamma_s = alpha_s * c_s * v_x + alpha_f * c_f * sgn_B_x * (beta_y * v_y + beta_z * v_z);

    L = zeros(8, 8); R = zeros(8, 8);

    L(1, 1) = (gamma_1 * alpha_f * v^2 + Gamma_f) / (2. * a^2);
    L(1, 2) = ((1. - GAMMA) * alpha_f * v_x - alpha_f * c_f) / (2. * a^2);
    L(1, 3) = ((1. - GAMMA) * alpha_f * v_y + c_s * alpha_s * beta_y * sgn_B_x) / (2. * a^2);
    L(1, 4) = ((1. - GAMMA) * alpha_f * v_z + c_s * alpha_s * beta_z * sgn_B_x) / (2. * a^2);
    L(1, 5) = -B_x * ((GAMMA - 1.) * alpha_f) / (2. * a^2);
    L(1, 6) = ((1. - GAMMA) * alpha_f * B_y + sqrt(rho) * a * alpha_s * beta_y) / (2. * a^2);
    L(1, 7) = ((1. - GAMMA) * alpha_f * B_z + sqrt(rho) * a * alpha_s * beta_z) / (2. * a^2);
    L(1, 8) = ((GAMMA - 1.) * alpha_f) / (2. * a^2);

    L(8, 1) = (gamma_1 * alpha_f * v^2 - Gamma_f) / (2. * a^2);
    L(8, 2) = ((1. - GAMMA) * alpha_f * v_x + alpha_f * c_f) / (2. * a^2);
    L(8, 3) = ((1. - GAMMA) * alpha_f * v_y - c_s * alpha_s * beta_y * sgn_B_x) / (2. * a^2);
    L(8, 4) = ((1. - GAMMA) * alpha_f * v_z - c_s * alpha_s * beta_z * sgn_B_x) / (2. * a^2);
    L(8, 5) = -B_x * ((GAMMA - 1.) * alpha_f) / (2. * a^2);
    L(8, 6) = ((1. - GAMMA) * alpha_f * B_y + sqrt(rho) * a * alpha_s * beta_y) / (2. * a^2);
    L(8, 7) = ((1. - GAMMA) * alpha_f * B_z + sqrt(rho) * a * alpha_s * beta_z) / (2. * a^2);
    L(8, 8) = ((GAMMA - 1.) * alpha_f) / (2. * a^2);

    R(1, 1) = alpha_f;
    R(1, 2) = alpha_f * (v_x - c_f);
    R(1, 3) = alpha_f * v_y + c_s * alpha_s * beta_y * sgn_B_x;
    R(1, 4) = alpha_f * v_z + c_s * alpha_s * beta_z * sgn_B_x;
    R(1, 5) = 0.;
    R(1, 6) = a * alpha_s * beta_y / sqrt(rho);
    R(1, 7) = a * alpha_s * beta_z / sqrt(rho);
    R(1, 8) = alpha_f * (v^2 / 2. + c_f^2 - gamma_2 * a^2) - Gamma_f;

    R(8, 1) = alpha_f;
    R(8, 2) = alpha_f * (v_x + c_f);
    R(8, 3) = alpha_f * v_y - c_s * alpha_s * beta_y * sgn_B_x;
    R(8, 4) = alpha_f * v_z - c_s * alpha_s * beta_z * sgn_B_x;
    R(8, 5) = 0.;
    R(8, 6) = a * alpha_s * beta_y / sqrt(rho);
    R(8, 7) = a * alpha_s * beta_z / sqrt(rho);
    R(8, 8) = alpha_f * (v^2 / 2. + c_f^2 - gamma_2 * a^2) + Gamma_f;

    L(2, 1) = Gamma_a / 2.;
    L(2, 2) = 0.;
    L(2, 3) = -beta_z * sgn_B_x / 2.;
    L(2, 4) = beta_y * sgn_B_x / 2.;
    L(2, 5) = 0.;
    L(2, 6) = -sqrt(rho) * beta_z / 2.;
    L(2, 7) = +sqrt(rho) * beta_y / 2.;
    L(2, 8) = 0.;

    L(7, 1) = Gamma_a / 2.;
    L(7, 2) = 0.;
    L(7, 3) = -beta_z * sgn_B_x / 2.;
    L(7, 4) = beta_y * sgn_B_x / 2.;
    L(7, 5) = 0.;
    L(7, 6) = +sqrt(rho) * beta_z / 2.;
    L(7, 7) = -sqrt(rho) * beta_y / 2.;
    L(7, 8) =  0.;

    R(2, 1) = 0.;
    R(2, 2) = 0.;
    R(2, 3) = -beta_z * sgn_B_x;
    R(2, 4) = beta_y * sgn_B_x;
    R(2, 5) = 0.;
    R(2, 6) = -beta_z / sqrt(rho);
    R(2, 7) = +beta_y / sqrt(rho);
    R(2, 8) = -Gamma_a;

    R(7, 1) = 0.;
    R(7, 2) = 0.;
    R(7, 3) = -beta_z * sgn_B_x;
    R(7, 4) = beta_y * sgn_B_x;
    R(7, 5) = 0.;
    R(7, 6) = +beta_z / sqrt(rho);
    R(7, 7) = -beta_y / sqrt(rho);
    R(7, 8) = -Gamma_a;

    L(3, 1) = (gamma_1 * alpha_s * v^2 + Gamma_s) / (2. * a^2);
    L(3, 2) = ((1. - GAMMA) * alpha_s * v_x - alpha_s * c_s) / (2. * a^2);
    L(3, 3) = ((1. - GAMMA) * alpha_s * v_y - c_f * alpha_f * beta_y * sgn_B_x) / (2. * a^2);
    L(3, 4) = ((1. - GAMMA) * alpha_s * v_z - c_f * alpha_f * beta_z * sgn_B_x) / (2. * a^2);
    L(3, 5) = -B_x * ((GAMMA - 1.) * alpha_s) / (2. * a^2);
    L(3, 6) = ((1. - GAMMA) * alpha_s * B_y - sqrt(rho) * a * alpha_f * beta_y) / (2. * a^2);
    L(3, 7) = ((1. - GAMMA) * alpha_s * B_z - sqrt(rho) * a * alpha_f * beta_z) / (2. * a^2);
    L(3, 8) = ((GAMMA - 1.) * alpha_s) / (2. * a^2);

    L(6, 1) = (gamma_1 * alpha_s * v^2 - Gamma_s) / (2. * a^2);
    L(6, 2) = ((1. - GAMMA) * alpha_s * v_x + alpha_s * c_s) / (2. * a^2);
    L(6, 3) = ((1. - GAMMA) * alpha_s * v_y + c_f * alpha_f * beta_y * sgn_B_x) / (2. * a^2);
    L(6, 4) = ((1. - GAMMA) * alpha_s * v_z + c_f * alpha_f * beta_z * sgn_B_x) / (2. * a^2);
    L(6, 5) = -B_x * ((GAMMA - 1.) * alpha_s) / (2. * a^2);
    L(6, 6) = ((1. - GAMMA) * alpha_s * B_y - sqrt(rho) * a * alpha_f * beta_y) / (2. * a^2);
    L(6, 7) = ((1. - GAMMA) * alpha_s * B_z - sqrt(rho) * a * alpha_f * beta_z) / (2. * a^2);
    L(6, 8) = ((GAMMA - 1.) * alpha_s) / (2. * a^2);

    R(3, 1) = alpha_s;
    R(3, 2) = alpha_s * (v_x - c_s);
    R(3, 3) = alpha_s * v_y - c_f * alpha_f * beta_y * sgn_B_x;
    R(3, 4) = alpha_s * v_z - c_f * alpha_f * beta_z * sgn_B_x;
    R(3, 5) = 0.;
    R(3, 6) = -a * alpha_f * beta_y / sqrt(rho);
    R(3, 7) = -a * alpha_f * beta_z / sqrt(rho);
    R(3, 8) = alpha_s * (v^2 / 2. + c_s^2 - gamma_2 * a^2) - Gamma_s;

    R(6, 1) = alpha_s;
    R(6, 2) = alpha_s * (v_x + c_s);
    R(6, 3) = alpha_s * v_y + c_f * alpha_f * beta_y * sgn_B_x;
    R(6, 4) = alpha_s * v_z + c_f * alpha_f * beta_z * sgn_B_x;
    R(6, 5) = 0.;
    R(6, 6) = -a * alpha_f * beta_y / sqrt(rho);
    R(6, 7) = -a * alpha_f * beta_z / sqrt(rho);
    R(6, 8) = alpha_s * (v^2 / 2. + c_s^2 - gamma_2 * a^2) + Gamma_s;

    L(4, 1) = 1. - tau * v^2 / 2.;
    L(4, 2) = tau * v_x;
    L(4, 3) = tau * v_y;
    L(4, 4) = tau * v_z;
    L(4, 5) = -B_x * -tau;
    L(4, 6) = tau * B_y;
    L(4, 7) = tau * B_z;
    L(4, 8) = -tau;

    R(4, 1) = 1.;
    R(4, 2) = v_x;
    R(4, 3) = v_y;
    R(4, 4) = v_z;
    R(4, 5) = 0.;
    R(4, 6) = 0.;
    R(4, 7) = 0.;
    R(4, 8) = v^2 / 2.;

    L(5, 1) = 0.;
    L(5, 2) = 0.;
    L(5, 3) = 0.;
    L(5, 4) = 0.;
    L(5, 5) = 1.;
    L(5, 6) = 0.;
    L(5, 7) = 0.;
    L(5, 8) = 0.;

    R(5, 1) = 0.;
    R(5, 2) = 0.;
    R(5, 3) = 0.;
    R(5, 4) = 0.;
    R(5, 5) = 1.;
    R(5, 6) = 0.;
    R(5, 7) = 0.;
    R(5, 8) = B_x;
end

function [L, R] = getLReigenvectorY(state, i, GAMMA)
    delta = 1e-12;

    rho = state(1, :); v_x = state(3, :); v_y = state(2, :); v_z = state(4, :);
    B_x = state(6, :); B_y = state(5, :); B_z = state(7, :); p   = state(8, :);

    v = sqrt(v_x.^2 + v_y.^2 + v_z.^2);
    B = sqrt(B_x.^2 + B_y.^2 + B_z.^2);

    rho = (rho(i) + rho(i + 1)) / 2.;
    v_x = (v_x(i) + v_x(i + 1)) / 2.;
    v_y = (v_y(i) + v_y(i + 1)) / 2.;
    v_z = (v_z(i) + v_z(i + 1)) / 2.;
    v   = (v(i)   + v(i + 1))   / 2.;
    B_x = (B_x(i) + B_x(i + 1)) / 2.;
    B_y = (B_y(i) + B_y(i + 1)) / 2.;
    B_z = (B_z(i) + B_z(i + 1)) / 2.;
    B   = (B(i)   + B(i + 1))   / 2.;
    p   = (p(i)   + p(i + 1))   / 2.;

    b_x = B_x / sqrt(rho);
    b_y = B_y / sqrt(rho);
    b_z = B_z / sqrt(rho);
    b   = sqrt(b_x^2 + b_y^2 + b_z^2);
    a   = sqrt(GAMMA * p / rho);
    c_f = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
    c_s = sqrt((a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.); 
    if (a^2 + b^2)^2 - 4. * a^2 * b_x^2 < 0.
        c_f = sqrt((a^2 + b^2) / 2.);
        c_s = sqrt((a^2 + b^2) / 2.);
    elseif a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2) < 0.
        c_s = 0.;
    end

    if B_y^2 + B_z^2 > delta * B^2
        beta_y = B_y / sqrt(B_y^2 + B_z^2);
        beta_z = B_z / sqrt(B_y^2 + B_z^2);
    else
        beta_y = 1. / sqrt(2);
        beta_z = 1. / sqrt(2);
    end
    if B_y^2 + B_z^2 > delta * B^2 && GAMMA * p - B_x^2 > delta * GAMMA * p
        alpha_f = sqrt(a^2 - c_s^2) / sqrt(c_f^2 - c_s^2);
        alpha_s = sqrt(c_f^2 - a^2) / sqrt(c_f^2 - c_s^2);
        if a^2 - c_s^2 < 0.
            alpha_f = 0.;
        end
        if c_f^2 - a^2 < 0.
            alpha_s = 0.;
        end
    else
        alpha_f = 1. / sqrt(2);
        alpha_s = 1. / sqrt(2);
    end
    if B_x >= 0.
        sgn_B_x = +1.;
    else
        sgn_B_x = -1.;
    end
    gamma_1 = (GAMMA - 1.) / 2.;
    gamma_2 = (GAMMA - 2.) / (GAMMA - 1.);
    tau     = (GAMMA - 1.) / a^2;
    Gamma_f = alpha_f * c_f * v_x - alpha_s * c_s * sgn_B_x * (beta_y * v_y + beta_z * v_z);
    Gamma_a = sgn_B_x * (beta_z * v_y - beta_y * v_z);
    Gamma_s = alpha_s * c_s * v_x + alpha_f * c_f * sgn_B_x * (beta_y * v_y + beta_z * v_z);

    L = zeros(8, 8); R = zeros(8, 8);

    L(1, 1) = (gamma_1 * alpha_f * v^2 + Gamma_f) / (2. * a^2);
    L(1, 3) = ((1. - GAMMA) * alpha_f * v_x - alpha_f * c_f) / (2. * a^2);
    L(1, 2) = ((1. - GAMMA) * alpha_f * v_y + c_s * alpha_s * beta_y * sgn_B_x) / (2. * a^2);
    L(1, 4) = ((1. - GAMMA) * alpha_f * v_z + c_s * alpha_s * beta_z * sgn_B_x) / (2. * a^2);
    L(1, 6) = -B_x * ((GAMMA - 1.) * alpha_f) / (2. * a^2);
    L(1, 5) = ((1. - GAMMA) * alpha_f * B_y + sqrt(rho) * a * alpha_s * beta_y) / (2. * a^2);
    L(1, 7) = ((1. - GAMMA) * alpha_f * B_z + sqrt(rho) * a * alpha_s * beta_z) / (2. * a^2);
    L(1, 8) = ((GAMMA - 1.) * alpha_f) / (2. * a^2);

    L(8, 1) = (gamma_1 * alpha_f * v^2 - Gamma_f) / (2. * a^2);
    L(8, 3) = ((1. - GAMMA) * alpha_f * v_x + alpha_f * c_f) / (2. * a^2);
    L(8, 2) = ((1. - GAMMA) * alpha_f * v_y - c_s * alpha_s * beta_y * sgn_B_x) / (2. * a^2);
    L(8, 4) = ((1. - GAMMA) * alpha_f * v_z - c_s * alpha_s * beta_z * sgn_B_x) / (2. * a^2);
    L(8, 6) = -B_x * ((GAMMA - 1.) * alpha_f) / (2. * a^2);
    L(8, 5) = ((1. - GAMMA) * alpha_f * B_y + sqrt(rho) * a * alpha_s * beta_y) / (2. * a^2);
    L(8, 7) = ((1. - GAMMA) * alpha_f * B_z + sqrt(rho) * a * alpha_s * beta_z) / (2. * a^2);
    L(8, 8) = ((GAMMA - 1.) * alpha_f) / (2. * a^2);

    R(1, 1) = alpha_f;
    R(1, 3) = alpha_f * (v_x - c_f);
    R(1, 2) = alpha_f * v_y + c_s * alpha_s * beta_y * sgn_B_x;
    R(1, 4) = alpha_f * v_z + c_s * alpha_s * beta_z * sgn_B_x;
    R(1, 6) = 0.;
    R(1, 5) = a * alpha_s * beta_y / sqrt(rho);
    R(1, 7) = a * alpha_s * beta_z / sqrt(rho);
    R(1, 8) = alpha_f * (v^2 / 2. + c_f^2 - gamma_2 * a^2) - Gamma_f;

    R(8, 1) = alpha_f;
    R(8, 3) = alpha_f * (v_x + c_f);
    R(8, 2) = alpha_f * v_y - c_s * alpha_s * beta_y * sgn_B_x;
    R(8, 4) = alpha_f * v_z - c_s * alpha_s * beta_z * sgn_B_x;
    R(8, 6) = 0.;
    R(8, 5) = a * alpha_s * beta_y / sqrt(rho);
    R(8, 7) = a * alpha_s * beta_z / sqrt(rho);
    R(8, 8) = alpha_f * (v^2 / 2. + c_f^2 - gamma_2 * a^2) + Gamma_f;

    L(2, 1) = Gamma_a / 2.;
    L(2, 3) = 0.;
    L(2, 2) = -beta_z * sgn_B_x / 2.;
    L(2, 4) = beta_y * sgn_B_x / 2.;
    L(2, 6) = 0.;
    L(2, 5) = -sqrt(rho) * beta_z / 2.;
    L(2, 7) = +sqrt(rho) * beta_y / 2.;
    L(2, 8) = 0.;

    L(7, 1) = Gamma_a / 2.;
    L(7, 3) = 0.;
    L(7, 2) = -beta_z * sgn_B_x / 2.;
    L(7, 4) = beta_y * sgn_B_x / 2.;
    L(7, 6) = 0.;
    L(7, 5) = +sqrt(rho) * beta_z / 2.;
    L(7, 7) = -sqrt(rho) * beta_y / 2.;
    L(7, 8) =  0.;

    R(2, 1) = 0.;
    R(2, 3) = 0.;
    R(2, 2) = -beta_z * sgn_B_x;
    R(2, 4) = beta_y * sgn_B_x;
    R(2, 6) = 0.;
    R(2, 5) = -beta_z / sqrt(rho);
    R(2, 7) = +beta_y / sqrt(rho);
    R(2, 8) = -Gamma_a;

    R(7, 1) = 0.;
    R(7, 3) = 0.;
    R(7, 2) = -beta_z * sgn_B_x;
    R(7, 4) = beta_y * sgn_B_x;
    R(7, 6) = 0.;
    R(7, 5) = +beta_z / sqrt(rho);
    R(7, 7) = -beta_y / sqrt(rho);
    R(7, 8) = -Gamma_a;

    L(3, 1) = (gamma_1 * alpha_s * v^2 + Gamma_s) / (2. * a^2);
    L(3, 3) = ((1. - GAMMA) * alpha_s * v_x - alpha_s * c_s) / (2. * a^2);
    L(3, 2) = ((1. - GAMMA) * alpha_s * v_y - c_f * alpha_f * beta_y * sgn_B_x) / (2. * a^2);
    L(3, 4) = ((1. - GAMMA) * alpha_s * v_z - c_f * alpha_f * beta_z * sgn_B_x) / (2. * a^2);
    L(3, 6) = -B_x * ((GAMMA - 1.) * alpha_s) / (2. * a^2);
    L(3, 5) = ((1. - GAMMA) * alpha_s * B_y - sqrt(rho) * a * alpha_f * beta_y) / (2. * a^2);
    L(3, 7) = ((1. - GAMMA) * alpha_s * B_z - sqrt(rho) * a * alpha_f * beta_z) / (2. * a^2);
    L(3, 8) = ((GAMMA - 1.) * alpha_s) / (2. * a^2);

    L(6, 1) = (gamma_1 * alpha_s * v^2 - Gamma_s) / (2. * a^2);
    L(6, 3) = ((1. - GAMMA) * alpha_s * v_x + alpha_s * c_s) / (2. * a^2);
    L(6, 2) = ((1. - GAMMA) * alpha_s * v_y + c_f * alpha_f * beta_y * sgn_B_x) / (2. * a^2);
    L(6, 4) = ((1. - GAMMA) * alpha_s * v_z + c_f * alpha_f * beta_z * sgn_B_x) / (2. * a^2);
    L(6, 6) = -B_x * ((GAMMA - 1.) * alpha_s) / (2. * a^2);
    L(6, 5) = ((1. - GAMMA) * alpha_s * B_y - sqrt(rho) * a * alpha_f * beta_y) / (2. * a^2);
    L(6, 7) = ((1. - GAMMA) * alpha_s * B_z - sqrt(rho) * a * alpha_f * beta_z) / (2. * a^2);
    L(6, 8) = ((GAMMA - 1.) * alpha_s) / (2. * a^2);

    R(3, 1) = alpha_s;
    R(3, 3) = alpha_s * (v_x - c_s);
    R(3, 2) = alpha_s * v_y - c_f * alpha_f * beta_y * sgn_B_x;
    R(3, 4) = alpha_s * v_z - c_f * alpha_f * beta_z * sgn_B_x;
    R(3, 6) = 0.;
    R(3, 5) = -a * alpha_f * beta_y / sqrt(rho);
    R(3, 7) = -a * alpha_f * beta_z / sqrt(rho);
    R(3, 8) = alpha_s * (v^2 / 2. + c_s^2 - gamma_2 * a^2) - Gamma_s;

    R(6, 1) = alpha_s;
    R(6, 3) = alpha_s * (v_x + c_s);
    R(6, 2) = alpha_s * v_y + c_f * alpha_f * beta_y * sgn_B_x;
    R(6, 4) = alpha_s * v_z + c_f * alpha_f * beta_z * sgn_B_x;
    R(6, 6) = 0.;
    R(6, 5) = -a * alpha_f * beta_y / sqrt(rho);
    R(6, 7) = -a * alpha_f * beta_z / sqrt(rho);
    R(6, 8) = alpha_s * (v^2 / 2. + c_s^2 - gamma_2 * a^2) + Gamma_s;

    L(4, 1) = 1. - tau * v^2 / 2.;
    L(4, 3) = tau * v_x;
    L(4, 2) = tau * v_y;
    L(4, 4) = tau * v_z;
    L(4, 6) = -B_x * -tau;
    L(4, 5) = tau * B_y;
    L(4, 7) = tau * B_z;
    L(4, 8) = -tau;

    R(4, 1) = 1.;
    R(4, 3) = v_x;
    R(4, 2) = v_y;
    R(4, 4) = v_z;
    R(4, 6) = 0.;
    R(4, 5) = 0.;
    R(4, 7) = 0.;
    R(4, 8) = v^2 / 2.;

    L(5, 1) = 0.;
    L(5, 3) = 0.;
    L(5, 2) = 0.;
    L(5, 4) = 0.;
    L(5, 6) = 1.;
    L(5, 5) = 0.;
    L(5, 7) = 0.;
    L(5, 8) = 0.;

    R(5, 1) = 0.;
    R(5, 3) = 0.;
    R(5, 2) = 0.;
    R(5, 4) = 0.;
    R(5, 6) = 1.;
    R(5, 5) = 0.;
    R(5, 7) = 0.;
    R(5, 8) = B_x;

end

function F = getF(U, GAMMA)

    e = U(8, :);

    state = convertUToState(U, GAMMA);
    rho   = state(1, :);
    v_x = state(2, :); v_y = state(3, :); v_z = state(4, :);
    B_x = state(5, :); B_y = state(6, :); B_z = state(7, :);
    p  = state(8, :);

    B = sqrt(B_x.^2 + B_y.^2 + B_z.^2);

    F = zeros(size(U));
    F(1, :) = rho .* v_x;
    F(2, :) = rho .* v_x.^2 + p + B.^2 / 2. - B_x.^2;
    F(3, :) = rho .* v_x .* v_y - B_x .* B_y;
    F(4, :) = rho .* v_x .* v_z - B_x .* B_z;
    F(5, :) = 0.;
    F(6, :) = v_x .* B_y - v_y .* B_x;
    F(7, :) = v_x .* B_z - v_z .* B_x;
    F(8, :) = v_x .* (e + p + B.^2 / 2.) - B_x .* (v_x .* B_x + v_y .* B_y + v_z .* B_z);

end

function G = getG(U, GAMMA)

    e = U(8, :);

    state = convertUToState(U, GAMMA);
    rho   = state(1, :);
    v_x = state(2, :); v_y = state(3, :); v_z = state(4, :);
    B_x = state(5, :); B_y = state(6, :); B_z = state(7, :);
    p  = state(8, :);

    B = sqrt(B_x.^2 + B_y.^2 + B_z.^2);

    G = zeros(size(U));
    G(1, :) = rho .* v_y;
    G(2, :) = rho .* v_y .* v_x - B_y .* B_x;
    G(3, :) = rho .* v_y.^2 + p + B.^2 / 2. - B_y.^2;
    G(4, :) = rho .* v_y .* v_z - B_y .* B_z;
    G(5, :) = v_y .* B_x - v_x .* B_y;
    G(6, :) = 0.;
    G(7, :) = v_y .* B_z - v_z .* B_y;
    G(8, :) = v_y .* (e + p + B.^2 / 2.) - B_y .* (v_x .* B_x + v_y .* B_y + v_z .* B_z);

end

function Phi_N = getPhiN(a, b, c, d)
    epsilon = 1e-6;
    
    IS_0    = 13. * (a - b)^2 + 3. * (a - 3. * b)^2;
    IS_1    = 13. * (b - c)^2 + 3. * (b + c)^2;
    IS_2    = 13. * (c - d)^2 + 3. * (3. * c - d)^2;

    alpha_0 = 1. / (epsilon + IS_0)^2;
    alpha_1 = 6. / (epsilon + IS_1)^2;
    alpha_2 = 3. / (epsilon + IS_2)^2;
    omega_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2);
    omega_2 = alpha_2 / (alpha_0 + alpha_1 + alpha_2);

    Phi_N = omega_0 * (a - 2. * b + c) / 3. + (omega_2 - 1. / 2.) * (b - 2. * c + d) / 6.;

end