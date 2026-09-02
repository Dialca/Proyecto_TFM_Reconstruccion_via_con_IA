% out struct contiene:
% out.t        -> tiempo IMU
% out.v        -> v(t) sincronizada a IMU
% out.rho      -> curvatura (fusión por defecto)
% out.rho_g    -> curvatura por gyro
% out.rho_a    -> curvatura por aceleración (si fuse_accel=true)
% out.yaw_rate -> ψ̇(t) proyectada a vertical del mundo (sin yaw absoluto)
% out.rhop     -> dρ/ds (para detectar marcas |ρ'(s)| grandes)
% out.s        -> abscisa acumulada s(t)
% out.qlt      -> máscara de validez (v>v_min)
%
% === Variante "_strapdown_basico" de estimate_curvature_imu_odo.m ===
% Corrige dos problemas del original:
%   1) BUG: más abajo (sección del filtro Butterworth sobre rho) había un
%      "fs = 500;" que SOBRESCRIBÍA el fs real del IMU (fs = 1/median(diff(t)),
%      calculado al inicio de la función). Eso diseñaba el pasabajos con una
%      frecuencia de muestreo incorrecta si el IMU no corre exactamente a
%      500 Hz. Aquí se eliminó ese "fs = 500;" y se reutiliza el fs real.
%   2) El divisor mágico "rho1/2.2" se expone como opts.rho_calib_scale
%      (mismo valor por defecto, 2.2, para no cambiar resultados ya
%      calibrados) en vez de un literal sin nombre ni explicación.
% El resto de la lógica numérica es idéntica al original.
%% =================== FUNCIONES LOCALES ===================
function out = estimate_curvature_imu_odo_strapdown_basico(imu, odo, opts)
  % --- columnas
    t  = imu.t(:);
    ax = imu.ax(:); ay = imu.ay(:); az = imu.az(:);
    wx = imu.wx(:); wy = imu.wy(:); wz = imu.wz(:);
    t_o = odo.t(:); v_o = odo.v(:);

    % --- valor por defecto del factor de calibración de rho (antes "/2.2" fijo) ---
    if ~isfield(opts, 'rho_calib_scale') || isempty(opts.rho_calib_scale)
        opts.rho_calib_scale = 2.2;   % factor de calibración empírico (ajustar con verdad de terreno)
    end

    % --- SG seguro
    fs   = 1/median(diff(t));
    sgN  = max(5, 2*floor((opts.sg_win_sec*fs)/2)+1);   % impar
    orderSG = max(2, min([opts.sg_order, 5, sgN-2]));   % 2..5 y < sgN

    % v al tiempo IMU
    v = interp1(t_o, v_o, t, 'pchip','extrap');
    v = fillmissing(v,'nearest');
    v = sgolayfilt(v, orderSG, sgN);

    % up (solo tilt) desde gravedad
    acc_lp = lowpass_vec([ax ay az], fs, opts.cutLP_acc);
    gb = acc_lp ./ max(vecnorm(acc_lp,2,2), 1e-6);
    u_up_b = -gb; % ENU: up = -g

    % yaw rate (nav) = w_b · up_b
    yaw_rate_raw = wx.*u_up_b(:,1) + wy.*u_up_b(:,2) + wz.*u_up_b(:,3);
    if isfinite(opts.hp_wz_tau) && opts.hp_wz_tau>0
        yaw_rate = hp1st(yaw_rate_raw, 1/fs, opts.hp_wz_tau);
    else
        yaw_rate = yaw_rate_raw;
    end
    yaw_rate = sgolayfilt(yaw_rate, orderSG, sgN);

    % rho^(2) = yaw_rate / |v|
    v_eff = max(v, opts.v_min);
    rho2 = yaw_rate ./ v_eff;

    % nivelado para a_lateral
    axl = acc_lp(:,1); ayl = acc_lp(:,2); azl = acc_lp(:,3);
    phi   = atan2( ayl,  azl);
    theta = atan2(-axl, hypot(ayl,azl));

    a_b = [ax ay az]; N = numel(t); a_l = zeros(N,3);
    for k = 1:N
        cph = cos(-phi(k));  sph = sin(-phi(k));
        cth = cos(-theta(k)); sth = sin(-theta(k));
        Rx = [1 0 0; 0 cph -sph; 0 sph cph];
        Ry = [cth 0 sth; 0 1 0; -sth 0 cth];
        Rlb = Ry * Rx;
        a_l(k,:) = (Rlb * a_b(k,:).').';
    end
    a_l(:,3) = a_l(:,3) - opts.g0;
    a_lat = a_l(:,2);
    a_lat = sgolayfilt(a_lat, orderSG, sgN);

    % rho^(3) = a_y / |v|^2
    rho3 = a_lat ./ max(v.^2, opts.v_min^2);

    % rho^(1) = sign(a_y) * yaw_rate^2 / max(|a_y|, ay_min)
    denom = max(abs(a_lat), opts.ay_min);
    rho1  = sign(a_lat) .* (yaw_rate.^2) ./ denom;

    % suavizados
    rho1 = sgolayfilt(rho1, orderSG, sgN);
    rho2 = sgolayfilt(rho2, orderSG, sgN);
    rho3 = sgolayfilt(rho3, orderSG, sgN);

    % rho fusionada (opcional)
    if isfield(opts,'fuse_accel') && opts.fuse_accel
        w = clamp((v - opts.fuse_v_thr) ./ max(opts.fuse_v_full - opts.fuse_v_thr, 1e-6), 0, 1);
        rho = (1 - 0.5*w).*rho2 + (0.5*w).*rho3;
    else
        rho = rho2;
    end
    rho = sgolayfilt(rho, orderSG, sgN);

% --- Filtrado wavelet con Daubechies (db4), nivel 4, usando MODWT ---
% x: tu señal columna (por ej. out.drho_dt(:))
% t: vector de tiempo (s) del mismo largo que x
x=rho(:)*(pi/(180));
% 1) magnitud analítica (envolvente "rápida")
% --- FIX strapdown_basico: NO se re-hardcodea fs aquí; se reutiliza el fs
%     real del IMU calculado arriba (fs = 1/median(diff(t))). El original
%     tenía "fs = 500;" en este punto, que sobrescribía el valor correcto.
fc  = .022;              % <-- frecuencia de corte que quieres (Hz)
n   = 4;              % orden del filtro (4 es buen compromiso)
Wn  = fc/(fs/2);      % normalización
[b,a] = butter(n, Wn, 'low');
rho1 = filtfilt(b, a, x);   % x es tu señal original
rho = rho1/opts.rho_calib_scale;
% ==== DERIVADAS **TEMPORALES** ====
    drho1_dt = gradient(rho1, t);
    drho2_dt = gradient(rho2, t);
    drho3_dt = gradient(rho3, t);
    drho_dt  = gradient(rho , t);

    % salida
    out = struct();
    out.t        = t;
    out.v        = v;
    out.yaw_rate = yaw_rate;
    out.a_lat    = a_lat;

    out.rho1  = rho1;  out.drho1_dt = drho1_dt;
    out.rho2  = rho2;  out.drho2_dt = drho2_dt;
    out.rho3  = rho3;  out.drho3_dt = drho3_dt;
    out.rho   = rho;   out.drho_dt  = drho_dt/2;

    % Alias por compatibilidad: rhop ahora significa dρ/dt
    out.rhop  = out.drho_dt;   % <-- derivada temporal
end

function acc_lp = lowpass_vec(acc, fs, fc)
    if fc <= 0, acc_lp = acc; return; end
    [b,a] = butter(2, fc/(fs/2), 'low');
    acc_lp = filtfilt(b,a,acc);
end

function y_hp = hp1st(x, Ts, tau)
    alpha = tau/(tau+Ts);
    y_hp = zeros(size(x)); y_prev = 0; x_prev = x(1);
    for k = 1:numel(x)
        y_hp(k) = alpha*(y_prev + x(k) - x_prev);
        y_prev = y_hp(k); x_prev = x(k);
    end
end

function y = clamp(x, lo, hi), y = min(max(x,lo),hi); end
