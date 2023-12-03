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

%% 3.1 One-Dimensional Riemann Problems 

% get initial conditions for the 1D Riemann problem
GAMMA = 2.;
Bx    = 0.75;

[x, dx, state] = getInitialCondition();
U = convertStateToU(state, GAMMA, Bx);
rho = state(1, :); By = state(5, :); p = state(7, :);

% visualize the initialization
figure(); hold on
title('initial conditions for the 1D Riemann problem')
plot(x, rho, '--'); 
plot(x, By , '-.');
plot(x, p); hold off
legend('$\rho$', '$B_y$', '$p$', 'Interpreter', 'latex');
xlabel('x')

% time stepping iteration
t = 0.;
t_end = 0.2; %t_end = 0.012;
CFL_number = 0.8;
step_counter = 0;
tic
while t < t_end
    % evaluate CFL condition
    dt = getCFL(U, dx, CFL_number, GAMMA, Bx);
    if t + dt > t_end
        dt = t_end - t;
    end

    % 4th-order non-TVD Runge-Kutta scheme (p. 570)
    U_0   = U;
    L_U_0 = getL_U(U_0, dx, GAMMA, Bx);
    U_1   = U_0 + dt / 2. * L_U_0;
    L_U_1 = getL_U(U_1, dx, GAMMA, Bx);
    U_2   = U_1 + dt / 2. * (-L_U_0 + L_U_1);
    L_U_2 = getL_U(U_2, dx, GAMMA, Bx);
    U_3   = U_2 + dt / 2. * (-L_U_1 + 2. * L_U_2);
    L_U_3 = getL_U(U_3, dx, GAMMA, Bx);
    U_4   = U_3 + dt / 6. * (L_U_0 + 2. * L_U_1 - 4. * L_U_2 + L_U_3);
    U     = U_4;

    % advance time
    t = t + dt;
    step_counter = step_counter + 1;
    fprintf("[step #%d] t = %.3e [dt = %.2e]\n", step_counter, t, dt);

end
toc

% visualize the final results
state = convertUToState(U, GAMMA, Bx);
rho = state(1, :); vx = state(2, :); vy = state(3, :); By = state(5, :); p = state(7, :);
figure();


sgtitle('Simulation results for the 1D Riemann problem [FIG. 2]');
subplot(3,2,[1,2]); plot(x, rho, '.'); xlabel('x'); ylabel('\rho'); ylim([0. 1.25]);
subplot(3,2,3);     plot(x, vx, '.');  xlabel('x'); ylabel('v_x');  ylim([-.3 .7]);
subplot(3,2,4);     plot(x, vy, '.');  xlabel('x'); ylabel('v_y');  ylim([-1.7 .1]);
subplot(3,2,5);     plot(x, By, '.');  xlabel('x'); ylabel('B_y');  ylim([-1.1 1.1]);
subplot(3,2,6);     plot(x, p, '.');   xlabel('x'); ylabel('p');    ylim([0. 1.1]);


%{
sgtitle('Simulation results for the 1D Riemann problem [FIG. 4]');
v = sqrt(vx.^2 + vy.^2);
subplot(2,2,1); plot(x, rho, '.');   xlabel('x'); ylabel('\rho'); ylim([0. 1.2]);
subplot(2,2,2); plot(x, v, '.');     xlabel('x'); ylabel('v');    ylim([-5. 35.]);
subplot(2,2,3); plot(x, By, '.');    xlabel('x'); ylabel('B_y');  ylim([-3.5 1.5]);
subplot(2,2,4); semilogy(x, p, '.'); xlabel('x'); ylabel('p');    ylim([.5e-1 2e3]);
%}

save('state.mat', 'state');

%% Function implementations

function [x, dx, state] = initializerForIntermediateShock()

    GAMMA = 5. / 3.;

    grid_count = 2560 + 1;
    x_min = +0.;
    x_max = +1.;
    dx = (x_max - x_min) / (grid_count - 1);

    % define arrays for each variable
    state = zeros(7, grid_count);   % state vector/matrix
    x     = zeros(grid_count, 1);   % x coordinate
    rho   = zeros(size(x));         % density
    vx    = zeros(size(x));         % velocity in x direction
    vy    = zeros(size(x));         % velocity in y direction
    vz    = zeros(size(x));         % velocity in z direction
    Bx    = zeros(size(x));
    By    = zeros(size(x));         % magnetic field in y direction
    Bz    = zeros(size(x));         % magnetic field in z direction
    p     = zeros(size(x));         % pressure

    for i = 1:grid_count
        x(i)  = (i - 1) * dx + x_min;
        rho(i) = 1.;   
        Bx(i) = 1.;
        By(i) = .5 * sin(2. * pi * x(i));
        p(i) = 1.;
    end

    for i = 2:grid_count
        delta = 1e-12;
        
        v = sqrt(vx(i-1).^2 + vy(i-1).^2);
        B = sqrt(Bx(i-1).^2 + By(i-1).^2);
    
        b_x = Bx(i-1) / sqrt(rho(i-1));
        b_y = By(i-1) / sqrt(rho(i-1));
        b_z = Bz(i-1) / sqrt(rho(i-1));
        b   = sqrt(b_x^2 + b_y^2 + b_z^2);
        a   = sqrt(GAMMA * p(i-1) / rho(i-1));
        c_f = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
        c_s = sqrt((a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.); 
        if By(i-1)^2 + Bz(i-1)^2 > delta * B^2
            beta_y = By(i-1) / sqrt(By(i-1)^2 + Bz(i-1)^2);
            beta_z = Bz(i-1) / sqrt(By(i-1)^2 + Bz(i-1)^2);
        else
            beta_y = 1. / sqrt(2);
            beta_z = 1. / sqrt(2);
        end
        if By(i-1)^2 + Bz(i-1)^2 > delta * B^2 || GAMMA * p(i-1) - Bx(i-1)^2 > delta * GAMMA * p(i-1)
            alpha_f = sqrt(a^2 - c_s^2) / sqrt(c_f^2 - c_s^2);
            alpha_s = sqrt(c_f^2 - a^2) / sqrt(c_f^2 - c_s^2);
        else
            alpha_f = 1. / sqrt(2);
            alpha_s = 1. / sqrt(2);
        end
        if Bx(i-1) >= 0.
            sgn_B_x = +1.;
        else
            sgn_B_x = -1.;
        end
        %gamma_1 = (GAMMA - 1.) / 2.;
        gamma_2 = (GAMMA - 2.) / (GAMMA - 1.);
        %tau     = (GAMMA - 1.) / a^2;
        %Gamma_f = alpha_f * c_f * vx(i-1) - alpha_s * c_s * sgn_B_x * (beta_y * vy(i-1) + beta_z * vz(i-1));
        %Gamma_a = sgn_B_x * (beta_z * vy(i-1) - beta_y * vz(i-1));
        Gamma_s = alpha_s * c_s * vx(i-1) + alpha_f * c_f * sgn_B_x * (beta_y * vy(i-1) + beta_z * vz(i-1));
        
        R = zeros(7, 1);
        R(1) = alpha_s;
        R(2) = alpha_s * (vx(i-1) + c_s);
        R(3) = alpha_s * vy(i-1) + c_f * alpha_f * beta_y * sgn_B_x;
        R(4) = alpha_s * vz(i-1) + c_f * alpha_f * beta_z * sgn_B_x;
        R(5) = -a * alpha_f * beta_y / sqrt(rho(i-1));
        R(6) = -a * alpha_f * beta_z / sqrt(rho(i-1));
        R(7) = alpha_s * (v^2 / 2. + c_s^2 - gamma_2 * a^2) + Gamma_s;

        U = zeros(7, 1);
        e = rho(i-1) .* (vx(i-1).^2 + vy(i-1).^2) / 2. ...
            + (Bx(i-1).^2 + By(i-1).^2) / 2. + p(i-1) / (GAMMA - 1.);
    
        U(1, :) = rho(i-1);
        U(2, :) = rho(i-1) .* vx(i-1);
        U(3, :) = rho(i-1) .* vy(i-1);
        U(4, :) = rho(i-1) .* vz(i-1);
        U(5, :) = By(i-1);
        U(6, :) = Bz(i-1);
        U(7, :) = p(i-1);

        dU = R;
        dBydx = .5*cos(2.*pi*x(i-1))*2.*pi;
        dU(1) = dBydx * By(i-1) / (c_s^2 - a^2);
        dU(2) = dBydx * (vx(i-1) + c_s) * By(i-1) / (c_s^2 - a^2);
        dU(3) = dBydx * (a^2*Bx(i-1) - c_s^2 + c_s*vy(i-1)*By(i-1)) / (c_s^3 - c_s*a^2);
        dU(4) = 0.;
        dU(5) = 0.;
        dU(6) = 0.;
        dU(7) = dBydx * a^2 * By(i-1) / (c_s^2 - a^2);
        U = U + dU * dx;

        rho(i)     = U(1, :);
        rho_vx  = U(2, :);
        rho_vy  = U(3, :);
        rho_vz  = U(4, :);
        %By(i)      = U(5, :);
        Bz(i)      = U(6, :);
        p(i)       = U(7, :);
    
        vx(i) = rho_vx ./ rho(i);
        vy(i) = rho_vy ./ rho(i);
        vz(i) = rho_vz ./ rho(i);
        %p(i)  = (GAMMA - 1.) .* (e - rho(i) .* (vx(i).^2 + vy(i).^2) / 2. ...
        %    - (Bx(i).^2 + By(i).^2 + Bz(i).^2) / 2.);
    end

    state(1, :) = rho;
    state(2, :) = vx;
    state(3, :) = vy;
    state(4, :) = vz;
    state(5, :) = By;
    state(6, :) = Bz;
    state(7, :) = p;

end

% Initial conditions for [3.1 One-Dimensional Riemann Problems]
function [x, dx, state] = getInitialCondition()
    % spatial resolution setup
    grid_count = 800 + 1; % grid_count = 200 + 1;
    x_min = -1.;
    x_max = +1.;
    dx = (x_max - x_min) / (grid_count - 1);

    % define arrays for each variable
    state = zeros(7, grid_count);   % state vector/matrix
    x     = zeros(grid_count, 1);   % x coordinate
    rho   = zeros(size(x));         % density
    vx    = zeros(size(x));         % velocity in x direction
    vy    = zeros(size(x));         % velocity in y direction
    vz    = zeros(size(x));         % velocity in z direction
    By    = zeros(size(x));         % magnetic field in y direction
    Bz    = zeros(size(x));         % magnetic field in z direction
    p     = zeros(size(x));         % pressure

    % fill in initial values
    for i = 1:grid_count
        x(i) = (i - 1) * dx + x_min;
        if x(i) < 0.
            rho(i) = +1.000;
            vx(i)  = +0.;
            vy(i)  = +0.;
            vz(i)  = +0.;
            By(i)  = +1.;
            Bz(i)  = +0.;
            p(i)   = +1.; % p(i) = +1000.;
        elseif x(i) == 0. % average of two sides at the discontinuity
            rho(i) = +0.5625;
            vx(i)  = +0.;
            vy(i)  = +0.;
            vz(i)  = +0.;
            By(i)  = +0.;
            Bz(i)  = +0.;
            p(i)   = +0.55; % p(i) = +500.05;
        else
            rho(i) = +0.125;
            vx(i)  = +0.;
            vy(i)  = +0.;
            vz(i)  = +0.;
            By(i)  = -1.;
            Bz(i)  = +0.;
            p(i)   = +0.1;
        end
    end

    % package each variable into the state vector/matrix
    state(1, :) = rho;
    state(2, :) = vx;
    state(3, :) = vy;
    state(4, :) = vz;
    state(5, :) = By;
    state(6, :) = Bz;
    state(7, :) = p;
end

% convert physical state variables to eigen-system variables (U)
% e.g. vx -> rho * vx
function U = convertStateToU(state, GAMMA, Bx)

    U = zeros(size(state));
    rho = state(1, :);
    vx  = state(2, :);
    vy  = state(3, :);
    vz  = state(4, :);
    By  = state(5, :);
    Bz  = state(6, :);
    p   = state(7, :);

    e = rho .* (vx.^2 + vy.^2 + vz.^2) / 2. ...
        + (Bx.^2 + By.^2 + Bz.^2) / 2. + p / (GAMMA - 1.);

    U(1, :) = rho;
    U(2, :) = rho .* vx;
    U(3, :) = rho .* vy;
    U(4, :) = rho .* vz;
    U(5, :) = By;
    U(6, :) = Bz;
    U(7, :) = e;
end

% convert eigen-system variables (U) to physical state variables
% e.g. rho * vx -> vx
function state = convertUToState(U, GAMMA, Bx)

    state   = zeros(size(U));
    rho     = U(1, :);
    rho_vx  = U(2, :);
    rho_vy  = U(3, :);
    rho_vz  = U(4, :);
    By      = U(5, :);
    Bz      = U(6, :);
    e       = U(7, :);

    vx = rho_vx ./ rho;
    vy = rho_vy ./ rho;
    vz = rho_vz ./ rho;
    p  = (GAMMA - 1.) .* (e - rho .* (vx.^2 + vy.^2 + vz.^2) / 2. ...
        - (Bx.^2 + By.^2 + Bz.^2) / 2.);

    state(1, :) = rho;
    state(2, :) = vx;
    state(3, :) = vy;
    state(4, :) = vz;
    state(5, :) = By;
    state(6, :) = Bz;
    state(7, :) = p;
end

% CFL condition evaluation
function dt = getCFL(U, dx, CFL_number, GAMMA, Bx)
    state = convertUToState(U, GAMMA, Bx);
    vx = state(2, :);   
    vx = abs(vx)';
    c_f = getCf(state, GAMMA, Bx);
    dt = CFL_number * dx / max(vx + c_f);
end

function c_f = getCf(state, GAMMA, Bx)

    rho = state(1, :); By = state(5, :); Bz = state(6, :); p = state(7, :);

    grid_count = size(state, 2);

    c_f = zeros(grid_count, 1);

    for i = 1:grid_count-1
        rho_i = (rho(i) + rho(i + 1)) / 2.;
        B_x_i = Bx;
        B_y_i = (By(i) + By(i + 1)) / 2.;
        B_z_i = (Bz(i) + Bz(i + 1)) / 2.;
        p_i   = (p(i) + p(i + 1)) / 2.;
    
        b_x = B_x_i / sqrt(rho_i);
        b_y = B_y_i / sqrt(rho_i);
        b_z = B_z_i / sqrt(rho_i);
        b   = sqrt(b_x^2 + b_y^2 + b_z^2);
        a   = sqrt(GAMMA * p_i / rho_i);
        c_f(i) = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
    end
end

% WENO solver
function L_U = getL_U(U, dx, GAMMA, Bx)
    grid_count = size(U, 2);
    L_U = zeros(size(U));
    for i = 1:grid_count
        [F_hat_right, F_hat_left] = getF_hat(U, i, GAMMA, Bx);
        L_U(:, i) = -1. / dx * (F_hat_right - F_hat_left); % Eqn. (2.15)
    end
end

% WENO flux function
function [F_hat_right, F_hat_left] = getF_hat(U, i, GAMMA, Bx)

    grid_count = size(U, 2);
    U_padded = zeros(7, grid_count + 6); % boundary padding (3 on each side)
    for j = 1:grid_count
        U_padded(:, j + 3) = U(:, j); % copy each value
    end

    % normal boundary condition
    for j = 1:3
        U_padded(:, j) = U(:, 1); % copy boundary value (left)
        U_padded(:, j + grid_count + 3) = U(:, grid_count); % copy boundary value (right)
    end

    % periodic boundary condition
    %{
    for j = 1:3
        U_padded(:, j) = U(:, grid_count + j - 3); % copy boundary value (left)
        U_padded(:, j + grid_count + 3) = U(:, j); % copy boundary value (right)
    end
    %}

    state = convertUToState(U_padded, GAMMA, Bx);
    F_U = getF(U_padded, GAMMA, Bx);

    alpha_s = getMaxEigenvalue(state, GAMMA, Bx); % global Lax-Friedrichs flux splitting

    % solve for F_hat_right
    [L, R] = getLReigenvector(state, i + 3, GAMMA, Bx);

    F_i = zeros(7, 4);
    for s = 1:7
        for j = 1:4
            F_i(:, j) = F_i(:, j) + dot(L(s, :), F_U(:, i+j+1)) .* R(s, :)'; % Eqn. (2.16)
        end
    end
    F_hat_right = (-F_i(:, 1) + 7. * F_i(:, 2) + 7. * F_i(:, 3) - F_i(:, 4)) / 12.; % Eqn. (2.17)
    for s = 1:7
        F_i_s_plus = zeros(5, 1);
        for j = 1:5
            F_i_s_plus(j) = (dot(L(s, :), F_U(:, i+j)) + alpha_s(s) * dot(L(s, :), U_padded(:, i+j))) / 2.;
        end
        Phi_N_plus = getPhiN(F_i_s_plus(2) - F_i_s_plus(1), F_i_s_plus(3) - F_i_s_plus(2), ...
            F_i_s_plus(4) - F_i_s_plus(3), F_i_s_plus(5) - F_i_s_plus(4));
        F_i_s_minus = zeros(5, 1);
        for j = 1:5
            F_i_s_minus(6-j) = (dot(L(s, :), F_U(:, i+j+1)) - alpha_s(s) * dot(L(s, :), U_padded(:, i+j+1))) / 2.;
        end
        Phi_N_minus = getPhiN(F_i_s_minus(1) - F_i_s_minus(2), F_i_s_minus(2) - F_i_s_minus(3), ...
            F_i_s_minus(3) - F_i_s_minus(4), F_i_s_minus(4) - F_i_s_minus(5));
        F_hat_right = F_hat_right + (-Phi_N_plus + Phi_N_minus) .* R(s, :)'; % Eqn. (2.17)
    end

    % solve for F_hat_left
    [L, R] = getLReigenvector(state, i + 2, GAMMA, Bx);
    F_i = zeros(7, 4);
    for s = 1:7
        for j = 1:4
            F_i(:, j) = F_i(:, j) + dot(L(s, :), F_U(:, i+j)) .* R(s, :)'; % Eqn. (2.16)
        end
    end
    F_hat_left = (-F_i(:, 1) + 7. * F_i(:, 2) + 7. * F_i(:, 3) - F_i(:, 4)) / 12.; % Eqn. (2.17)
    for s = 1:7
        F_i_s_plus = zeros(5, 1);
        for j = 1:5
            F_i_s_plus(j) = (dot(L(s, :), F_U(:, i+j-1)) + alpha_s(s) * dot(L(s, :), U_padded(:, i+j-1))) / 2.;
        end
        Phi_N_plus = getPhiN(F_i_s_plus(2) - F_i_s_plus(1), F_i_s_plus(3) - F_i_s_plus(2), ...
            F_i_s_plus(4) - F_i_s_plus(3), F_i_s_plus(5) - F_i_s_plus(4));
        F_i_s_minus = zeros(5, 1);
        for j = 1:5
            F_i_s_minus(6-j) = (dot(L(s, :), F_U(:, i+j)) - alpha_s(s) * dot(L(s, :), U_padded(:, i+j))) / 2.;
        end
        Phi_N_minus = getPhiN(F_i_s_minus(1) - F_i_s_minus(2), F_i_s_minus(2) - F_i_s_minus(3), ...
            F_i_s_minus(3) - F_i_s_minus(4), F_i_s_minus(4) - F_i_s_minus(5));
        F_hat_left = F_hat_left + (-Phi_N_plus + Phi_N_minus) .* R(s, :)'; % Eqn. (2.17)
    end

end

function alpha_s = getMaxEigenvalue(state, GAMMA, Bx)

    rho = state(1, :); vx = state(2, :);
    By  = state(5, :); Bz = state(6, :); p = state(7, :);

    grid_count = size(state, 2);

    alpha_s = zeros(7, 1);

    for i = 1:grid_count-1
        rho_i = (rho(i) + rho(i + 1)) / 2.;
        v_x_i = (vx(i) + vx(i + 1)) / 2.;
        B_x_i = Bx;
        B_y_i = (By(i) + By(i + 1)) / 2.;
        B_z_i = (Bz(i) + Bz(i + 1)) / 2.;
        p_i   = (p(i) + p(i + 1)) / 2.;
    
        b_x = B_x_i / sqrt(rho_i);
        b_y = B_y_i / sqrt(rho_i);
        b_z = B_z_i / sqrt(rho_i);
        b   = sqrt(b_x^2 + b_y^2 + b_z^2);
        a   = sqrt(GAMMA * p_i / rho_i);
        c_a = abs(b_x);
        c_f = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
        c_s = sqrt((a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);

        if abs(v_x_i - c_f) >= alpha_s(1)
            alpha_s(1) = abs(v_x_i - c_f);
        end
        if abs(v_x_i + c_f) >= alpha_s(7)
            alpha_s(7) = abs(v_x_i + c_f);
        end
        if abs(v_x_i - c_a) >= alpha_s(2)
            alpha_s(2) = abs(v_x_i - c_a);
        end
        if abs(v_x_i + c_a) >= alpha_s(6)
            alpha_s(6) = abs(v_x_i + c_a);
        end
        if abs(v_x_i - c_s) >= alpha_s(3)
            alpha_s(3) = abs(v_x_i - c_s);
        end
        if abs(v_x_i + c_s) >= alpha_s(5)
            alpha_s(5) = abs(v_x_i + c_s);
        end
        if abs(v_x_i) >= alpha_s(4)
            alpha_s(4) = abs(v_x_i);
        end
    end

end

function [L, R] = getLReigenvector(state, i, GAMMA, Bx)
    delta = 1e-12;

    rho = state(1, :); vx = state(2, :); vy = state(3, :); vz = state(4, :);
    By  = state(5, :); Bz = state(6, :); p  = state(7, :);

    v = sqrt(vx.^2 + vy.^2 + vz.^2);
    B = sqrt(Bx.^2 + By.^2 + Bz.^2);

    rho = (rho(i) + rho(i + 1)) / 2.;
    v_x = (vx(i) + vx(i + 1)) / 2.;
    v_y = (vy(i) + vy(i + 1)) / 2.;
    v_z = (vz(i) + vz(i + 1)) / 2.;
    v   = (v(i) + v(i + 1)) / 2.;
    B_x = Bx;
    B_y = (By(i) + By(i + 1)) / 2.;
    B_z = (Bz(i) + Bz(i + 1)) / 2.;
    B   = (B(i) + B(i+1)) / 2.;
    p   = (p(i) + p(i + 1)) / 2.;

    b_x = B_x / sqrt(rho);
    b_y = B_y / sqrt(rho);
    b_z = B_z / sqrt(rho);
    b   = sqrt(b_x^2 + b_y^2 + b_z^2);
    a   = sqrt(GAMMA * p / rho);
    c_f = sqrt((a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.);
    c_s = sqrt((a^2 + b^2 - sqrt((a^2 + b^2)^2 - 4. * a^2 * b_x^2)) / 2.); 
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

    L = zeros(7, 7); R = zeros(7, 7);

    L(1, 1) = (gamma_1 * alpha_f * v^2 + Gamma_f) / (2. * a^2);
    L(1, 2) = ((1. - GAMMA) * alpha_f * v_x - alpha_f * c_f) / (2. * a^2);
    L(1, 3) = ((1. - GAMMA) * alpha_f * v_y + c_s * alpha_s * beta_y * sgn_B_x) / (2. * a^2);
    L(1, 4) = ((1. - GAMMA) * alpha_f * v_z + c_s * alpha_s * beta_z * sgn_B_x) / (2. * a^2);
    L(1, 5) = ((1. - GAMMA) * alpha_f * B_y + sqrt(rho) * a * alpha_s * beta_y) / (2. * a^2);
    L(1, 6) = ((1. - GAMMA) * alpha_f * B_z + sqrt(rho) * a * alpha_s * beta_z) / (2. * a^2);
    L(1, 7) = ((GAMMA - 1.) * alpha_f) / (2. * a^2);

    L(7, 1) = (gamma_1 * alpha_f * v^2 - Gamma_f) / (2. * a^2);
    L(7, 2) = ((1. - GAMMA) * alpha_f * v_x + alpha_f * c_f) / (2. * a^2);
    L(7, 3) = ((1. - GAMMA) * alpha_f * v_y - c_s * alpha_s * beta_y * sgn_B_x) / (2. * a^2);
    L(7, 4) = ((1. - GAMMA) * alpha_f * v_z - c_s * alpha_s * beta_z * sgn_B_x) / (2. * a^2);
    L(7, 5) = ((1. - GAMMA) * alpha_f * B_y + sqrt(rho) * a * alpha_s * beta_y) / (2. * a^2);
    L(7, 6) = ((1. - GAMMA) * alpha_f * B_z + sqrt(rho) * a * alpha_s * beta_z) / (2. * a^2);
    L(7, 7) = ((GAMMA - 1.) * alpha_f) / (2. * a^2);

    R(1, 1) = alpha_f;
    R(1, 2) = alpha_f * (v_x - c_f);
    R(1, 3) = alpha_f * v_y + c_s * alpha_s * beta_y * sgn_B_x;
    R(1, 4) = alpha_f * v_z + c_s * alpha_s * beta_z * sgn_B_x;
    R(1, 5) = a * alpha_s * beta_y / sqrt(rho);
    R(1, 6) = a * alpha_s * beta_z / sqrt(rho);
    R(1, 7) = alpha_f * (v^2 / 2. + c_f^2 - gamma_2 * a^2) - Gamma_f;

    R(7, 1) = alpha_f;
    R(7, 2) = alpha_f * (v_x + c_f);
    R(7, 3) = alpha_f * v_y - c_s * alpha_s * beta_y * sgn_B_x;
    R(7, 4) = alpha_f * v_z - c_s * alpha_s * beta_z * sgn_B_x;
    R(7, 5) = a * alpha_s * beta_y / sqrt(rho);
    R(7, 6) = a * alpha_s * beta_z / sqrt(rho);
    R(7, 7) = alpha_f * (v^2 / 2. + c_f^2 - gamma_2 * a^2) + Gamma_f;

    L(2, 1) = Gamma_a / 2.;
    L(2, 2) = 0.;
    L(2, 3) = -beta_z * sgn_B_x / 2.;
    L(2, 4) = beta_y * sgn_B_x / 2.;
    L(2, 5) = -sqrt(rho) * beta_z / 2.;
    L(2, 6) = +sqrt(rho) * beta_y / 2.;
    L(2, 7) = 0.;

    L(6, 1) = Gamma_a / 2.;
    L(6, 2) = 0.;
    L(6, 3) = -beta_z * sgn_B_x / 2.;
    L(6, 4) = beta_y * sgn_B_x / 2.;
    L(6, 5) = +sqrt(rho) * beta_z / 2.;
    L(6, 6) = -sqrt(rho) * beta_y / 2.;
    L(6, 7) =  0.;

    R(2, 1) = 0.;
    R(2, 2) = 0.;
    R(2, 3) = -beta_z * sgn_B_x;
    R(2, 4) = beta_y * sgn_B_x;
    R(2, 5) = -beta_z / sqrt(rho);
    R(2, 6) = +beta_y / sqrt(rho);
    R(2, 7) = -Gamma_a;

    R(6, 1) = 0.;
    R(6, 2) = 0.;
    R(6, 3) = -beta_z * sgn_B_x;
    R(6, 4) = beta_y * sgn_B_x;
    R(6, 5) = +beta_z / sqrt(rho);
    R(6, 6) = -beta_y / sqrt(rho);
    R(6, 7) = -Gamma_a;

    L(3, 1) = (gamma_1 * alpha_s * v^2 + Gamma_s) / (2. * a^2);
    L(3, 2) = ((1. - GAMMA) * alpha_s * v_x - alpha_s * c_s) / (2. * a^2);
    L(3, 3) = ((1. - GAMMA) * alpha_s * v_y - c_f * alpha_f * beta_y * sgn_B_x) / (2. * a^2);
    L(3, 4) = ((1. - GAMMA) * alpha_s * v_z - c_f * alpha_f * beta_z * sgn_B_x) / (2. * a^2);
    L(3, 5) = ((1. - GAMMA) * alpha_s * B_y - sqrt(rho) * a * alpha_f * beta_y) / (2. * a^2);
    L(3, 6) = ((1. - GAMMA) * alpha_s * B_z - sqrt(rho) * a * alpha_f * beta_z) / (2. * a^2);
    L(3, 7) = ((GAMMA - 1.) * alpha_s) / (2. * a^2);

    L(5, 1) = (gamma_1 * alpha_s * v^2 - Gamma_s) / (2. * a^2);
    L(5, 2) = ((1. - GAMMA) * alpha_s * v_x + alpha_s * c_s) / (2. * a^2);
    L(5, 3) = ((1. - GAMMA) * alpha_s * v_y + c_f * alpha_f * beta_y * sgn_B_x) / (2. * a^2);
    L(5, 4) = ((1. - GAMMA) * alpha_s * v_z + c_f * alpha_f * beta_z * sgn_B_x) / (2. * a^2);
    L(5, 5) = ((1. - GAMMA) * alpha_s * B_y - sqrt(rho) * a * alpha_f * beta_y) / (2. * a^2);
    L(5, 6) = ((1. - GAMMA) * alpha_s * B_z - sqrt(rho) * a * alpha_f * beta_z) / (2. * a^2);
    L(5, 7) = ((GAMMA - 1.) * alpha_s) / (2. * a^2);

    R(3, 1) = alpha_s;
    R(3, 2) = alpha_s * (v_x - c_s);
    R(3, 3) = alpha_s * v_y - c_f * alpha_f * beta_y * sgn_B_x;
    R(3, 4) = alpha_s * v_z - c_f * alpha_f * beta_z * sgn_B_x;
    R(3, 5) = -a * alpha_f * beta_y / sqrt(rho);
    R(3, 6) = -a * alpha_f * beta_z / sqrt(rho);
    R(3, 7) = alpha_s * (v^2 / 2. + c_s^2 - gamma_2 * a^2) - Gamma_s;

    R(5, 1) = alpha_s;
    R(5, 2) = alpha_s * (v_x + c_s);
    R(5, 3) = alpha_s * v_y + c_f * alpha_f * beta_y * sgn_B_x;
    R(5, 4) = alpha_s * v_z + c_f * alpha_f * beta_z * sgn_B_x;
    R(5, 5) = -a * alpha_f * beta_y / sqrt(rho);
    R(5, 6) = -a * alpha_f * beta_z / sqrt(rho);
    R(5, 7) = alpha_s * (v^2 / 2. + c_s^2 - gamma_2 * a^2) + Gamma_s;

    L(4, 1) = 1. - tau * v^2 / 2.;
    L(4, 2) = tau * v_x;
    L(4, 3) = tau * v_y;
    L(4, 4) = tau * v_z;
    L(4, 5) = tau * B_y;
    L(4, 6) = tau * B_z;
    L(4, 7) = -tau;

    R(4, 1) = 1.;
    R(4, 2) = v_x;
    R(4, 3) = v_y;
    R(4, 4) = v_z;
    R(4, 5) = 0.;
    R(4, 6) = 0.;
    R(4, 7) = v^2 / 2.;
end

function F = getF(U, GAMMA, Bx)

    e = U(7, :);

    state = convertUToState(U, GAMMA, Bx);
    rho   = state(1, :); vx = state(2, :); vy = state(3, :); vz = state(4, :);
    By    = state(5, :); Bz = state(6, :); p  = state(7, :);

    B = sqrt(Bx.^2 + By.^2 + Bz.^2);

    F = zeros(size(U));
    F(1, :) = rho .* vx;
    F(2, :) = rho .* vx.^2 + p + B.^2 / 2.;
    F(3, :) = rho .* vx .* vy - Bx .* By;
    F(4, :) = rho .* vx .* vz - Bx .* Bz;
    F(5, :) = vx .* By - vy .* Bx;
    F(6, :) = vx .* Bz - vz .* Bx;
    F(7, :) = vx .* (e + p + B.^2 / 2.) - Bx .* (vx .* Bx + vy .* By + vz .* Bz);
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