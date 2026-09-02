%% Algoritmo_2_strapdown_basico.m
%% Variante corregida de Algoritmo_2.m — IMU + Marcas + Odómetro + Algoritmo 2
%
% Este script carga IMU, alineación/quita-gravedad, marcas, odómetro, integración
% ZUPT, estimación de curvatura, Feature_Estimation, postprocess_features,
% corrección de odometría por matching y gráficas.


clc; close all; clear all;


carpeta_base   = fileparts(mfilename('fullpath'));
carpeta_sesion = fullfile(carpeta_base, 'UIS_Guatiguara_2');   % <-- carpeta de la sesión
sesion_id      = '015';                                        % <-- ID de sesión

% --- GPX: autodetectado en carpeta_sesion/gps/Track_*.gpx ---
gpx_dir  = fullfile(carpeta_sesion, 'gps');
gpx_list = dir(fullfile(gpx_dir, 'Track_*.gpx'));
if isempty(gpx_list)
    error(['Algoritmo_2_strapdown_basico: no se encontró ningún "Track_*.gpx" en "%s". ' ...
           'Define gpxFile manualmente si tu archivo tiene otro nombre/patrón.'], gpx_dir);
end
gpxFile = fullfile(gpx_list(1).folder, gpx_list(1).name);
fprintf('Archivo GPX (auto): %s\n', gpxFile);
fname1 = string(gpxFile);

% --- IMU / ODOMETRO / TIMESTAMPS: nombre fijo "<prefijo>_<sesion_id>.csv" ---
fname_imu       = fullfile(carpeta_sesion, ['DatosIMU_' sesion_id '.csv']);  %%%%%AQUI%%%%%%%%%%%%%
fname_odo       = fullfile(carpeta_sesion, ['hall_' sesion_id '.csv']);
timestamps_file = fullfile(carpeta_sesion, ['timestamps_' sesion_id '.csv']);
fname_mark      = '';   % '' = sin marcas (opcional); pon una ruta si tienes CSV de marcas

% ==================        GNSS       ==================
[F, idxF,q_truth,tg,CoordTruth] = curvature_features_from_gpx_strapdown_basico(fname1);

% ==================                          ==================
% ==================_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_        ==================
% ================== IMU MARCAS ODOMETRO ==================

% ================== PARÁMETROS GLOBALES ==================
% (fname_imu, fname_mark, fname_odo ya quedaron fijados en CONFIGURACIÓN arriba)

% Heurísticas/constantes
dt_guess_imu = 0.01;      % [s] si el IMU NO trae tiempo
dt_guess_odo = 0.50;      % [s] si el odómetro NO trae tiempo (tu caso)
g0            = 9.80665;  % [m/s^2]
remove_bias_after = true;

% Odómetro (tus datos)
wheel_radius_m        = 0.14;                           % 14 cm
ticks_per_rev         = 6;                              % 6 pulsos por vuelta
wheel_circumference_m = 2*pi*wheel_radius_m;           % ≈ 0.43982 m
bin_dt                = 0.5;                            % integrar/contar cada 0.5 s

% ================== 1) CARGA IMU (ruta fijada en CONFIGURACIÓN) ==================
if exist(fname_imu, 'file') ~= 2
    error('Algoritmo_2_strapdown_basico: no existe el archivo IMU "%s". Revisa CONFIGURACIÓN.', fname_imu);
end

T = read_csv_generic(fname_imu);

% --- Normaliza nombres
origNames = T.Properties.VariableNames;
vn = matlab.lang.makeValidName(lower(origNames), 'ReplacementStyle','delete');
T.Properties.VariableNames = vn;
fprintf('\n== Encabezados IMU (normalizados) ==\n'); disp(T.Properties.VariableNames.');

% --- Pick columnas
it  = pickVarSmart(T, ["t","time","tiempo","timestamp","sec","seconds","t_s","ts","tsec","t_ms","tus","t_us"], ...
                      ["time","tiempo","timestamp","sec","seconds","t_s","ts","tms","tus","tus"]);
iax = pickVarSmart(T, ["ax","ax_g","accx","a_x","ax_b","acelx","accelx","acc_x"], ...
                      ["ax","accx","acel","accel","a_x","axg"]);
iay = pickVarSmart(T, ["ay","ay_g","accy","a_y","ay_b","acely","accely","acc_y"], ...
                      ["ay","accy","acel","accel","a_y","ayg"]);
iaz = pickVarSmart(T, ["az","az_g","accz","a_z","az_b","acelz","accelz","acc_z"], ...
                      ["az","accz","acel","accel","a_z","azg"]);
iwx = pickVarSmart(T, ["wx","gx","gyro_x","wxb","wx_b","gyrx","w_x"], ...
                      ["wx","gx","gyro","gyr","omega","w_x"]);
iwy = pickVarSmart(T, ["wy","gy","gyro_y","wyb","wy_b","gyry","w_y"], ...
                      ["wy","gy","gyro","gyr","omega","w_y"]);
iwz = pickVarSmart(T, ["wz","gz","gyro_z","wzb","wz_b","gyrz","w_z"], ...
                      ["wz","gz","gyro","gyr","omega","w_z"]);
iroll  = pickVarSmart(T, ["roll","phi","rol","roll_deg","roll_rad"], ["roll","phi","rol"]);
ipitch = pickVarSmart(T, ["pitch","theta","inclino","incl","inclinometro","tilt","pitch_deg","pitch_rad","theta_deg","theta_rad"], ...
                         ["pitch","theta","inclino","incl","tilt"]);
iyaw   = pickVarSmart(T, ["yaw","psi","heading","rumbo","yaw_deg","yaw_rad"], ["yaw","psi","heading","rumbo"]);

% --- Reporte
reportMatch('tiempo', it, T);
reportMatch('ax', iax, T); reportMatch('ay', iay, T); reportMatch('az', iaz, T);
reportMatch('wx', iwx, T); reportMatch('wy', iwy, T); reportMatch('wz', iwz, T);
reportMatch('roll', iroll, T); reportMatch('pitch', ipitch, T); reportMatch('yaw', iyaw, T);

N = height(T);

% --- Tiempo IMU
if ~isempty(it)
    t = toDoubleColumn(T{:,it});
    dt_med = median(diff(t(isfinite(t))));
    if isfinite(dt_med) && dt_med > 1 && max(t,[],'omitnan') > 1000
        t = t * 1e-3; fprintf('IMU: tiempo parecía estar en ms → convertido a s.\n');
    end
else
    t = (0:N-1).' * dt_guess_imu;
    fprintf('IMU: sin tiempo; se creó con dt = %.4f s.\n', dt_guess_imu);
end

% --- FIX strapdown_basico: factor de corrección de tiempo DERIVADO de
% timestamps_0XX.csv
[t, factor_tiempo, info_tiempo] = corregir_tiempo_imu_strapdown_basico(t, timestamps_file);
fprintf('IMU: %s\n', info_tiempo.message);


% --- Señales crudas
ax = valOrNaNRob(T, iax); ay = valOrNaNRob(T, iay); az = valOrNaNRob(T, iaz);
wx = valOrNaNRob(T, iwx); wy = valOrNaNRob(T, iwy); wz = valOrNaNRob(T, iwz);
roll  = valOrNaNRob(T, iroll);
pitch = valOrNaNRob(T, ipitch);
yaw   = valOrNaNRob(T, iyaw);

% --- Unidades
[ax, ay, az, accUnits, units_state] = normalize_acc_units(ax, ay, az, ...
    varNameOrEmpty(T, iax), varNameOrEmpty(T, iay), varNameOrEmpty(T, iaz), g0);
fprintf('IMU: Unidades aceleración: %s (units_state=%s)\n', accUnits, units_state);
[wx, ~, c1] = ensureRadians(wx); [wy, ~, c2] = ensureRadians(wy); [wz, ~, c3] = ensureRadians(wz);
gyroUnits = 'rad/s'; if any([c1,c2,c3]), gyroUnits='rad/s (desde °/s)'; end
[roll,  rollUnits,  ~] = ensureAngleRadians(roll);
[pitch, pitchUnits, ~] = ensureAngleRadians(pitch);
[yaw,   yawUnits,   ~] = ensureAngleRadians(yaw);

% --- Magnitudes y "reposo"
amag = sqrt(ax.^2 + ay.^2 + az.^2);
wmag = sqrt(wx.^2 + wy.^2 + wz.^2);
dt   = median(diff(t(isfinite(t)))); fs = 1/max(dt,eps);

% ================== 2) Alineación y sustracción de gravedad ==================
Araw = [ax ay az];

% Detectar "quieto"
if ~all(isnan(wmag))
    still_w  = movmean(wmag, max(1,round(0.5*fs))) < 0.15;
else
    still_w  = true(size(t));
end
dr      = [zeros(1,3); diff(Araw)];
still_a = vecnorm(dr,2,2) < prctile(vecnorm(dr,2,2), 60);
still   = still_w & still_a;
if nnz(still) < 0.2*numel(t), still = t <= (t(1)+2.0); end

% Gravedad medida y escala por si el IMU entró en "g"
g_meas0 = median(Araw(still,:),1).';
gnorm0  = norm(g_meas0);
if gnorm0 > 6 && gnorm0 < 13, scale = 1.0;
elseif gnorm0 < 3,             scale = g0 / max(gnorm0,eps);
else,                          scale = 1.0;
end
A = Araw * scale; g_meas = g_meas0 * scale;

% Rotar y quitar g
R      = rot_from_two_vectors(g_meas, [0;0;g0]);
A_rot  = (R * A.').';
ax_lin = A_rot(:,1); ay_lin = A_rot(:,2); az_lin = A_rot(:,3) - g0;

% Sesgo en reposo (opcional)
A_lin_mat = [ax_lin, ay_lin, az_lin];
if remove_bias_after
    bias = median(A_lin_mat(still,:), 1, 'omitnan');
    ax_lin = ax_lin - bias(1);
    ay_lin = ay_lin - bias(2);
    az_lin = az_lin - bias(3);
    A_lin_mat = [ax_lin, ay_lin, az_lin];
end
alin = sqrt(sum(A_lin_mat.^2,2));

ang_deg = acosd( max(-1,min(1, dot(g_meas/norm(g_meas), [0;0;1])) ) );
fprintf('IMU: ||g_meas0||=%.3f  scale=%.3f  ángulo(R)=%.2f°  med(||a_lin||, still)=%.3f\n', ...
        gnorm0, scale, ang_deg, median(alin(still),'omitnan'));

% ================== 3) (OPCIONAL) CARGAR MARCAS (sin diálogo) ==================
t_mark = []; mark_label = strings(0,1);

Tm = [];
if strlength(string(fname_mark)) > 0
    if exist(char(fname_mark), 'file') == 2
        Tm = read_csv_generic(fname_mark);
    else
        warning('MARCAS: fname_mark="%s" no existe; se omiten marcas.', char(fname_mark));
    end
end

if ~isempty(Tm)
    names_m = lower(Tm.Properties.VariableNames);
    it_m = pickVarSmart(Tm, ["t","time","tiempo","timestamp","sec","seconds","t_s","ts","t_ms"], ...
                            ["time","tiempo","timestamp","sec","seconds","t_s","ts","tms"]);
    ilab = find(contains(names_m, {'label','name','evento','event','mark','beacon','tipo'}), 1, 'first');
    if isempty(it_m)
        t_mark = (0:height(Tm)-1).' * dt_guess_odo;   % suponer paso fijo si no hay tiempo
        fprintf('MARCAS: sin columna de tiempo → se usó dt=%.3f s.\n', dt_guess_odo);
    else
        t_mark = toDoubleColumn(Tm{:,it_m});
        dtm = median(diff(t_mark(isfinite(t_mark))));
        if isfinite(dtm) && dtm > 1 && max(t_mark,[],'omitnan') > 1000
            t_mark = t_mark*1e-3; fprintf('MARCAS: tiempo ms→s.\n');
        end
    end
    if isempty(ilab)
        mark_label = strings(numel(t_mark),1);
    else
        rawLab = Tm{:,ilab};
        if iscell(rawLab), mark_label = string(rawLab);
        elseif isstring(rawLab), mark_label = rawLab;
        else, mark_label = string(rawLab);
        end
    end
    fprintf('MARCAS: %d eventos cargados.\n', numel(t_mark));
end

% ================== 4) CARGAR ODOMETRO (sin diálogo) ==================

odo_loaded = cargar_odometro_strapdown_basico(fname_odo, ...
    'wheel_radius_m', wheel_radius_m, 'ticks_per_rev', ticks_per_rev, ...
    'bin_dt', bin_dt, 'dt_guess_odo', dt_guess_odo);

tb         = odo_loaded.t;
speed_bin  = odo_loaded.v;
cum_dist   = odo_loaded.dist_odo;
pulses_bin = odo_loaded.pulses_bin;
dist_bin   = odo_loaded.dist_bin;
t_odo      = odo_loaded.t_raw;
pulses_raw = odo_loaded.pulses_raw;

% ================== 5) GRÁFICAS ==================
% Outliers suavizados para visualización (aceleración lineal)
try
    ax_v = filloutliers(ax_lin,'linear','median','ThresholdFactor',6);
    ay_v = filloutliers(ay_lin,'linear','median','ThresholdFactor',6);
    az_v = filloutliers(az_lin,'linear','median','ThresholdFactor',6);
catch
    ax_v = ax_lin; ay_v = ay_lin; az_v = az_lin;
end

% 5.1 Aceleraciones (crudas)
figure('Name','Aceleraciones (crudas)');
ax1 = subplot(4,1,1); plot(t, ax); grid on; ylabel(['a_x [' accUnits ']']); title('Aceleraciones (crudas)');
ax2 = subplot(4,1,2); plot(t, ay); grid on; ylabel(['a_y [' accUnits ']']);
ax3 = subplot(4,1,3); plot(t, az); grid on; ylabel(['a_z [' accUnits ']']);
ax4 = subplot(4,1,4); plot(t, amag); grid on; hold on;
try yline(g0,'--','g'); catch, line([t(1) t(end)],[g0 g0],'LineStyle','--'); end
ylabel(['||a|| [' accUnits ']']); xlabel('t [s]');
linkaxes([ax1,ax2,ax3,ax4],'x');
add_markers([ax1,ax2,ax3,ax4], t_mark, mark_label, [0 0 0]); % marcas en negro

% 5.2 Giros
if ~all(isnan(wx)) || ~all(isnan(wy)) || ~all(isnan(wz))
    figure('Name','Girológicos (IMU)');
    bx1 = subplot(4,1,1); plot(t, wx); grid on; ylabel(['\omega_x [' gyroUnits ']']); title('Giros (IMU)');
    bx2 = subplot(4,1,2); plot(t, wy); grid on; ylabel(['\omega_y [' gyroUnits ']']);
    bx3 = subplot(4,1,3); plot(t, wz); grid on; ylabel(['\omega_z [' gyroUnits ']']);
    bx4 = subplot(4,1,4); plot(t, wmag); grid on; ylabel(['||\omega|| [' gyroUnits ']']); xlabel('t [s]');
    linkaxes([bx1,bx2,bx3,bx4],'x');
end
% 5.3 Aceleración lineal (sin g)
figure('Name','Aceleración lineal (gravedad removida)');
dl1 = subplot(4,1,1); plot(t, ax_v); grid on; ylabel('a_x^{lin} [m/s^2]'); title('Aceleración lineal');
dl2 = subplot(4,1,2); plot(t, ay_v); grid on; ylabel('a_y^{lin} [m/s^2]');
dl3 = subplot(4,1,3); plot(t, az_v); grid on; ylabel('a_z^{lin} [m/s^2]');
dl4 = subplot(4,1,4); plot(t, alin); grid on; ylabel('||a||^{lin} [m/s^2]'); xlabel('t [s]');
linkaxes([dl1,dl2,dl3,dl4],'x');
add_markers([dl1,dl2,dl3,dl4], t_mark, mark_label, [0 0 0]);

% 5.4 Odómetro — pulsos/0.5 s, velocidad, distancia
if ~isempty(tb)
    figure('Name','Odómetro (bins de 0.5 s)');
    od1 = subplot(3,1,1); stairs(tb, pulses_bin, 'b-', 'LineWidth',1.1); grid on;
          ylabel(sprintf('pulsos / %.1f s', bin_dt)); title('Odómetro (conteo por bin)');
    od2 = subplot(3,1,2); plot(tb, speed_bin, 'g-', 'LineWidth',1.1); grid on;
          ylabel('vel [m/s]');
    od3 = subplot(3,1,3); plot(tb, cum_dist, 'k-', 'LineWidth',1.2); grid on;
          ylabel('dist acum [m]'); xlabel('t [s]');
    linkaxes([od1,od2,od3],'x');
    add_markers([od1,od2,od3], t_mark, mark_label, [0 0 0]);
end

disp('✔ Listo. Revisa las figuras y las líneas de marca.');

% ================== 6) INTEGRACIÓN SIMPLE (ZUPT por segmentos) ==================
A = [ax_lin ay_lin az_lin];
N = numel(t); v = zeros(N,3); p = zeros(N,3);
moving = ~still;
d = diff([false; moving; false]);
starts = find(d==1); ends = find(d==-1)-1;
for s = 1:numel(starts)
    k0 = starts(s); k1 = ends(s);
    for k = k0+1:k1
        dtk = t(k)-t(k-1);
        v(k,:) = v(k-1,:) + 0.5*(A(k,:)+A(k-1,:))*dtk;
        p(k,:) = p(k-1,:) + 0.5*(v(k,:)+v(k-1,:))*dtk;
    end
    if k1 < N, v(k1,:) = 0; end  % ZUPT
end
rangeX = max(p(:,1)) - min(p(:,1));
rangeY = max(p(:,2)) - min(p(:,2));
rangeZ = max(p(:,3)) - min(p(:,3));
fprintf('Desplazamiento aprox [m]: X=%.3f  Y=%.3f  Z=%.3f\n', rangeX, rangeY, rangeZ);

% Visual 3D + marcas temporales (si hay)
figure('Name','Trayectoria aproximada 3D');
plot3(p(:,1),p(:,2),p(:,3),'k-','LineWidth',1.2); grid on; axis equal;
xlabel('E [m]'); ylabel('N [m]'); zlabel('U [m]'); title('Trayectoria aprox (IMU)');
hold on;
if ~isempty(t_mark)
    for i=1:numel(t_mark)
        [~,kk] = min(abs(t - t_mark(i)));
        plot3(p(kk,1),p(kk,2),p(kk,3),'ro','MarkerSize',6,'LineWidth',1.2);
    end
end
% ==================             ==================             ===========

%                                Algoritmo 2

% ==================             ==================             ===========
imu = struct('t',t(:), 'ax',ax(:), 'ay',ay(:), 'az',az(:), ...
                       'wx',wx(:), 'wy',wy(:), 'wz',wz(:));
odo = struct('t',tb(:), 'v',speed_bin(:), 'dist_odo',cum_dist(:));
% Opciones
opts = struct();

%% ====== OPCIONES ======
opts.g0          = 9.80665;   % gravedad
opts.v_min       = 0.50;      % m/s  (evita divisiones por ~0)
opts.ay_min      = 0.20;      % m/s^2 (evita singularidad de ρ^(1))
opts.cutLP_acc   = 2.0;       % Hz   (LP para estimar tilt)
opts.hp_wz_tau   = 200.0;     % s    (HP muy suave para drift de yaw-rate)
opts.sg_win_sec  = 0.25;      % s    (ventana Savitzky–Golay)
opts.sg_order    = 3;         % orden SG (se limita internamente para que sea válido)
opts.fuse_accel  = true;      % usa mezcla simple ρ^(2) con ρ^(3)
opts.fuse_v_thr  = 6.0;       % m/s (inicio de fusión)
opts.fuse_v_full = 18.0;      % m/s (fusión plena)
opts.show_plots  = true;
opts.rho_calib_scale = 2.2;   % FIX strapdown_basico: antes literal "/2.2" oculto en estimate_curvature_imu_odo.m

%% ====== CÁLCULO ======
out = estimate_curvature_imu_odo_strapdown_basico(imu, odo, opts);

%% ====== RESULTADOS / GRÁFICAS ======
% === Comparación: rho(t) arriba y d rho/ds abajo ===
%% ===== GRÁFICAS (ρ arriba, dρ/dt abajo) =====
if opts.show_plots
    % === PREVIO: Todas las curvaturas ρ^(1), ρ^(2), ρ^(3) y ρ (fusionada) ===
    figure('Name','Todas las \rho y sus derivadas (temporales)');
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    % (A) Curvaturas
    axA = nexttile; hold(axA,'on'); grid(axA,'on');
    plot(axA, out.t, out.rho1, 'DisplayName','\rho^{(1)}=\psi̇^2/a_y');
    plot(axA, out.t, out.rho2, 'DisplayName','\rho^{(2)}=\psi̇/\|v\|');
    plot(axA, out.t, out.rho3, 'DisplayName','\rho^{(3)}=a_y/\|v\|^2');
    plot(axA, out.t, out.rho , 'k', 'LineWidth',1.2, 'DisplayName','\rho (fusionada)');
    yline(axA,0,':');
    ylabel(axA,'\rho [1/m]');
    legend(axA,'Location','best');
    title(axA,'Curvaturas \rho(t)');

    % (B) Derivadas temporales dρ/dt
    axB = nexttile; hold(axB,'on'); grid(axB,'on');
    plot(axB, out.t, out.drho1_dt, 'DisplayName','d\rho^{(1)}/dt');
    plot(axB, out.t, out.drho2_dt, 'DisplayName','d\rho^{(2)}/dt');
    plot(axB, out.t, out.drho3_dt, 'DisplayName','d\rho^{(3)}/dt');
    plot(axB, out.t, out.drho_dt , 'k', 'LineWidth',1.2, 'DisplayName','d\rho/dt (fusionada)');
    yline(axB,0,':');
    ylabel(axB,'d\rho/dt [1/(m·s)]'); xlabel(axB,'t [s]');
    legend(axB,'Location','best');
    title(axB,'Derivadas temporales d\rho/dt');
    linkaxes([axA,axB],'x');

    % === FINAL: tu figura original apilada (ρ arriba, dρ/dt abajo) ===
    figure('Name','\rho(t) y d\rho/dt');
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    ax1 = nexttile; hold(ax1,'on'); grid(ax1,'on');
    plot(ax1, out.t, out.rho, 'LineWidth',1.2, 'DisplayName','\rho(t)');
    yline(ax1,0,':'); ylabel(ax1,'\rho [1/m]'); legend(ax1,'Location','best');
    title(ax1,'Curvatura (fusionada)');

    ax2 = nexttile; hold(ax2,'on'); grid(ax2,'on');
    plot(ax2, out.t, out.drho_dt, 'LineWidth',1.2, 'DisplayName','d\rho/dt');
    yline(ax2,0,':'); ylabel(ax2,'d\rho/dt [1/(m·s)]'); xlabel(ax2,'t [s]');
    legend(ax2,'Location','best'); title(ax2,'Derivada temporal');

    linkaxes([ax1,ax2],'x');
end

% 2) Derivada por distancia (la del paper):  ρ' = (1/v) * dρ/dt
%drho_dq = out.drho_dt ./ v;
    ctx = struct();
    % Campos mínimos necesarios con defaults
    if ~isfield(ctx,'q_LM') || ~isfinite(ctx.q_LM)
        ctx.q_LM = -inf;             % límite inferior de abscisa en mapa
    end
    if ~isfield(ctx,'eta_l') || ~isfinite(ctx.eta_l)
        ctx.eta_l = 0.03;            % umbral bajo para |rho'|
    end
    if ~isfield(ctx,'eta_h') || ~isfinite(ctx.eta_h)
        ctx.eta_h = 0.12;            % umbral alto (si usas histéresis)
    end

%% ====== Algoritmo 2: Feature_Estimation (usa dρ/dt) ======

% Umbral η_l (robusto) y detección
eta_l = 2.0 * mad(abs(out.drho_dt), 1);   % puedes ajustar el 3.0
[features, idx_feat] = Feature_Estimation(out.t, out.rho,  out.drho_dt, eta_l);

% (Opcional) marcar los rasgos en la figura de rho vs drho/dt
% === Mostrar el umbral eta_l en drho/dt (ax2) ===
% Sanitizar idx_feat por si acaso
idx_feat = unique( idx_feat(isfinite(idx_feat) & idx_feat>=1 & idx_feat<=numel(out.t)) );

% --- Valores absolutos de la derivada ---
abs_rhop = abs(out.drho_dt);

% Plots si existen ejes
if exist('ax1','var') && exist('ax2','var') && isgraphics(ax1) && isgraphics(ax2)
    % Marcar rasgos en rho(t)
    plot(ax1, out.t(idx_feat), out.rho(idx_feat), 'ro', 'DisplayName','rasgos');
    legend(ax1,'Location','best');
    % === ax2: |dρ/dt|, umbral y máximos ===
    hold(ax2,'on');
    % Curva |dρ/dt|
    plot(ax2, out.t, abs_rhop, 'DisplayName','|d\rho/dt|');

    % Umbral (solo positivo, porque graficamos valor absoluto)
    if exist('yline','file') || exist('yline','builtin')
        yline(ax2, eta_l, '--', sprintf('\\eta_l = %.3g', eta_l), ...
              'LabelHorizontalAlignment','left', ...
              'LabelVerticalAlignment','bottom', ...
              'DisplayName','\eta_l');
    else
        xl = xlim(ax2);
        line(ax2, xl, [eta_l eta_l], 'LineStyle','--', 'Color',[0 0 0], ...
             'DisplayName','\eta_l');
    end

    % Banda sombreada [0, eta_l]
    xl = xlim(ax2);
    p = patch(ax2, [xl(1) xl(2) xl(2) xl(1)], [0 0 eta_l eta_l], ...
              [0.5 0.5 0.5], 'FaceAlpha',0.12, 'EdgeColor','none', ...
              'DisplayName','|{\it\rho''}| < \eta_l');
    uistack(p,'bottom');

    % Máximos en idx_feat
    plot(ax2, out.t(idx_feat), abs_rhop(idx_feat), 'kd', ...
         'MarkerFaceColor','w', 'LineStyle','none', ...
         'DisplayName','máximos');

    legend(ax2,'Location','best');
end
% ==================                                     ==================
% F: Nx3 -> [q, rho, rhop]

ctx.F_map = array2table(F, 'VariableNames', {'qtruth','rho','rhop'});
% agregar índice si no existe
if ~ismember('idx', ctx.F_map.Properties.VariableNames)
    ctx.F_map.idx = (1:height(ctx.F_map)).';
end

if ~isfield(ctx, 'qodo') || isempty(ctx.q)
    % si no hay odometría en ctx, la calculamos
    if ~isfield(out,'v')
        error('Falta out.v para integrar la odometría');
    end
    ctx.qodo = cumtrapz(out.t(:), out.v(:));
end


ctx.T.a   = (out.t(end) - out.t(1)) / (imu.t(end) - imu.t(1));
ctx.T.b   = out.t(1) - ctx.T.a * imu.t(1);
ctx.T.map = @(t_imu) ctx.T.a * t_imu + ctx.T.b;

% ====== Ventana de matching (en metros), basada en velocidad ======

ctx.tauLB = 2.0;
ctx.tauUB = 5.0;

ctx.deltaLB_fn = @(t) max(20, ctx.tauLB * interp1(out.t, out.v, t, 'linear','extrap'));
ctx.deltaUB_fn = @(t) max(20, ctx.tauUB * interp1(out.t, out.v, t, 'linear','extrap'));

  % ====== 1) todo en el eje del odómetro ======
t_odo   = out.t(:);          % tiempo del odómetro
q_odo   = ctx.qodo(:);          % odometría sin corregir  (q̃(t))
ctx.q_truth = q_truth(:);        % tu trayectoria "verdad" en metros
t_truth = linspace(0,t_odo(end),length(q_truth));
q_truth = interp1(t_truth', q_truth, t_odo,'nearest');  % si hace falta

ctx.q_truth = q_truth(:);        % tu trayectoria "verdad" en metros
% === Paso 2 del Algoritmo (l.19–22) ===
% Construir f = (t*, ρ*, ρ′*) y ejecutar selección+matching por cada rasgo
[matches, f_list, ctx] = postprocess_features(features, out, ctx);

n = numel(f_list);
%q_est = NaN(n,1); %%%%%%%%%%%%%%%%%  NUEVO  %%%%%%%%%%%%%%%%%%%%%%%%%
t_star    = NaN(n,1);
rho_star  = NaN(n,1);
rhop_star = NaN(n,1);
map_idx   = NaN(n,1);
accepted  = NaN(n,1);
q_matched =  NaN(n,1);
lambda =  NaN(n,1);
for i = 1:n %%%%%%%%%%%%%%%%%  ANTERIOR  %%%%%%%%%%%%%%%%%%%%%%%%%
%for i = 1:min(n, numel(matches)) %%%%%%%%%%%%%%%%%  NUEVO  %%%%%%%%%%
    % datos del rasgo detectado
    if ~isempty(f_list(i).t),    t_star(i)    = f_list(i).t;    end
    if ~isempty(f_list(i).rho),  rho_star(i)  = f_list(i).rho;  end
    if ~isempty(f_list(i).rhop), rhop_star(i) = f_list(i).rhop; end

    % datos del match
    if ~isempty(matches(i).map_idx)
        map_idx(i) = matches(i).map_idx;
    end
    % accepted es lógico; lo paso a double para la tabla
    if ~isempty(matches(i).accepted)
        accepted(i) = double(matches(i).accepted);
    end
        % datos del match
    if ~isempty(matches(i).q_matched)
        q_matched(i) = matches(i).q_matched;
    end
            % datos del match
    if ~isempty(matches(i).q_est)
        q_est(i) = matches(i).q_est;
    end
    if ~isempty(matches(i).q_est)
        lambda(i) = matches(i).lambda;
    end
end

summary_tbl = table(t_star, rho_star, rhop_star, map_idx, accepted,q_matched,q_est',lambda, ...
    'VariableNames', {'t_star','rho_star','rhop_star','map_idx','accepted','q_matched','q_est','lamnda'});


q_est = q_est(:);
q_matched = q_matched(:);
accepted = accepted(:);

% 2) lambda del paper: q_odo - q_mapa, pero solo cuando accepted==1
lambda_paper = nan(size(q_est));
valid = accepted==1 & isfinite(q_matched) & isfinite(q_est);
lambda_paper(valid) = q_est(valid) - q_matched(valid);

% 3) tu lambda actual la renombramos a innov
lambda_innov = lambda(:);

% 4) armamos la tabla con las dos
summary_tbl = table(t_star, rho_star, rhop_star, map_idx, accepted, ...
    q_matched, q_est, lambda_paper, lambda_innov, ...
    'VariableNames', {'t_star','rho_star','rhop_star','map_idx', ...
    'accepted','q_matched','q_est','lambda_paper','lambda_innov'});



disp(summary_tbl);

summary_tbl2 = table(F(:,1), F(:,2), F(:,3), ...
    'VariableNames', {'q_GPS','rho_GPS','rhop_GPS'});

disp(summary_tbl2);

% v = out.v(:);
% odo.v
%ctx.qodo

% 1) error sin corregir
eps_tilde = q_odo - q_truth;

% 2) inicializar
lambda_likepaper_hold = zeros(size(out.t));
last_lambda = 0;
tol_t = 1e-3;

for i = 1:numel(out.t)
    t0 = out.t(i);

    % mirar si hay match en este t
    for j = 1:numel(matches)
        if matches(j).accepted
            % tiempo del match
            if isfield(ctx,'T') && isfield(ctx.T,'map')
                t_match = ctx.T.map(matches(j).t_star);
            else
                t_match = matches(j).t_star;
            end

            if abs(t0 - t_match) < tol_t
                
                last_lambda = eps_tilde(i);   
            end
        end
    end

    lambda_likepaper_hold(i) = last_lambda;
end

% 3) error corregido
eps_f_likepaper = eps_tilde - lambda_likepaper_hold;

figure;
plot(out.t, eps_tilde, 'LineWidth', 1.4); hold on; grid on;
plot(out.t, eps_f_likepaper, 'LineWidth', 1.4);

xlabel('t [s]', 'Interpreter', 'latex');
ylabel('Error de odometría [m]', 'Interpreter', 'latex');
title('Error de odometría: sin corrección vs corregido (estilo paper)', ...
      'Interpreter', 'latex');

legend({'$\tilde{\epsilon}(t) = q_{\text{odo}} - q_{\text{truth}}$', ...
        '$\epsilon_f(t) = \tilde{\epsilon}(t) - \lambda(t)$ (reset en match)'}, ...
        'Interpreter', 'latex', 'Location', 'best');

% ya tienes:
% out.t
% q_odo
% lambda_likepaper_hold

% odometría corregida
q_odo_corr = q_odo - lambda_likepaper_hold;

figure;

%% --- EJE PRINCIPAL ---
axMain = axes;
hold(axMain, 'on'); grid(axMain, 'on');

plot(axMain, out.t, q_odo,      'LineWidth', 1.4);
plot(axMain, out.t, q_odo_corr, 'LineWidth', 1.4);
plot(axMain, out.t, q_truth,    'LineWidth', 1.4);

xlabel(axMain, 't [s]', 'Interpreter','latex');
ylabel(axMain, 'Recorrido [m]', 'Interpreter','latex');
title(axMain, 'Odometría original vs corregida vs verdad', 'Interpreter','latex');

legend(axMain, {'$q_{\text{odo}}(t)$', ...
                '$q_{\text{odo,corr}}(t)$', ...
                '$q_{\text{truth}}(t)$'}, ...
                'Interpreter','latex', 'Location','best');

%% --- ZOOMS (parametrizados en un solo bucle) ---

w = 0.18;   % ancho
h = 0.18;   % alto
zoom_windows = struct( ...
    'trange', {[0 50], [80 220], [260 390]}, ...
    'pos',    {[0.16 0.36 w h], [0.39 0.56 w h], [0.72 0.42 w h]}, ...
    'titulo', {'Zoom 35–50 s', 'Zoom 195–220 s', 'Zoom 320–390 s'});

for zi = 1:numel(zoom_windows)
    trange = zoom_windows(zi).trange;
    idxz = (out.t >= trange(1)) & (out.t <= trange(2));
    axz = axes('Parent', gcf, 'Position', zoom_windows(zi).pos); % [left bottom width height]
    hold(axz, 'on'); grid(axz, 'on');
    plot(axz, out.t(idxz), q_odo(idxz),      'LineWidth', 1.0);
    plot(axz, out.t(idxz), q_odo_corr(idxz), 'LineWidth', 1.0);
    plot(axz, out.t(idxz), q_truth(idxz),    'LineWidth', 1.0);
    xlim(axz, trange);
    title(axz, zoom_windows(zi).titulo, 'Interpreter','latex');
    set(axz, 'FontSize', 7, 'Box','on');
    xlabel(axz, ''); ylabel(axz, '');
end

% --- FIX strapdown_basico ("Tm guard")

if ~isempty(t_mark)
    fprintf('DEBUG marcas/tiempo/odometro -> [t_mark(end)=%.3f, t(end)=%.3f, ctx.qodo(end)=%.3f]\n', ...
        t_mark(end), t(end), ctx.qodo(end));
else
    fprintf('DEBUG (sin marcas) -> [t(end)=%.3f, ctx.qodo(end)=%.3f]\n', t(end), ctx.qodo(end));
end


uu=0;
pg=CoordTruth.pg;
Eg=CoordTruth.Eg;
Ng=CoordTruth.Ng;
Ug=CoordTruth.Ug;
p0u=CoordTruth.p0u;
LatU=CoordTruth.LatU;
LonU=CoordTruth.LonU;
lat0=CoordTruth.lat0;
lon0=CoordTruth.lon0;
h0=CoordTruth.h0 ;

% ==== REQUIERE en el workspace ====
% asumimos: pg, Eg, Ng       (mapa)
%           ctx.qodo         (odo sin corregir)
%           lambda_hold      (corrección)
% queremos el error a lo largo de pg

% 1) odometría sin corregir llevada al eje del mapa
% asumimos: pg, Eg, Ng       (mapa)
%           ctx.qodo         (odo sin corregir)
%           lambda_hold      (corrección)

% asumimos: pg, Eg, Ng       (mapa)
%           ctx.qodo         (odo sin corregir)
%           lambda_hold      (corrección)

% ====== datos que asumimos que ya existen ======
% pg, Eg, Ng          % mapa remuestreado
% ctx.qodo            % odometría sin corregir
% lambda_hold         % corrección (mismo largo que qodo)

% ---------- preparar corrección sobre el eje del mapa ----------
% ================== SUPUESTOS ==================
% pg, Eg, Ng          % mapa remuestreado (pg en metros)
% ctx.qodo            % odometría sin corregir (m)
% lambda_hold         % corrección aplicada (m)

% ----- 1) preparar odometría cruda y corrección -----
% ====== datos que asumimos que ya tienes ======
% pg, Eg, Ng      % mapa remuestreado (misma longitud los 3)
% ctx.qodo        % odometría sin corregir (metros)
% lambda_hold     % corrección (metros), mismo largo que qodo

% ====== SUPONEMOS QUE YA EXISTEN ======
% pg, Eg, Ng          % mapa remuestreado (pg en metros)
% ctx.qodo            % odometría sin corregir (metros)
% lambda_hold         % corrección (metros), mismo largo que qodo

% ====== SUPONEMOS QUE YA EXISTEN ======
% pg, Eg, Ng          % mapa remuestreado (pg en metros)
% ctx.qodo            % odometría sin corregir (metros)
% lambda_hold         % corrección (metros)
lambda_hold(:)=lambda_likepaper_hold;
% --- 1) señales base ---
q_raw  = ctx.qodo(:);
q_corr = ctx.qodo(:) - lambda_hold(:);

% quitamos NaN
q_raw  = q_raw(isfinite(q_raw));
q_corr = q_corr(isfinite(q_corr));

% --- 2) ponemos TODO en la misma "escala de progreso" 0..1 ---
Nmap  = numel(pg);
t_map = linspace(0, 1, Nmap);

t_raw  = linspace(0, 1, numel(q_raw));
t_corr = linspace(0, 1, numel(q_corr));

q_raw_on_map  = interp1(t_raw,  q_raw,  t_map, 'linear', 'extrap');
q_corr_on_map = interp1(t_corr, q_corr, t_map, 'linear', 'extrap');

% --- 3) error en metros respecto al mapa ---
err_before = q_raw_on_map  - pg(:)';   % ojo: pg puede ser columna
err_after  = q_corr_on_map - pg(:)';

% convertimos TODO a columna y recortamos a la misma longitud
pg   = pg(:);
Eg   = Eg(:);
Ng   = Ng(:);
err_before = err_before(:);
err_after  = err_after(:);

L = min([numel(pg), numel(Eg), numel(Ng), numel(err_before), numel(err_after)]);
pg          = pg(1:L);
Eg          = Eg(1:L);
Ng          = Ng(1:L);
err_before  = err_before(1:L);
err_after   = err_after(1:L);

% ====== FIGURA ======
fig = figure('Name','Error sobre trayectoria (antes / después)');
tl  = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

% misma escala de colores
err_max = max(abs([err_before; err_after]));
if err_max == 0
    err_max = 1;  % para que no reviente caxis si todo es cero
end
cax = [-err_max, err_max];

% --- IZQUIERDA: sin corregir ---
ax1 = nexttile(tl, 1);
hold(ax1,'on'); axis(ax1,'equal'); grid(ax1,'on'); box(ax1,'on');
plot(ax1, Eg, Ng, 'k-', 'LineWidth', 1, 'DisplayName','Mapa');
scatter(ax1, Eg, Ng, 30, err_before, 'filled');
colormap(ax1, parula);
caxis(ax1, cax);
cb1 = colorbar(ax1); cb1.Label.String = 'error (m)';
title(ax1, 'Odometía SIN corregir (error en m)');
xlabel(ax1,'E [m]'); ylabel(ax1,'N [m]');
legend(ax1,'Location','best');

% --- DERECHA: corregida ---
ax2 = nexttile(tl, 2);
hold(ax2,'on'); axis(ax2,'equal'); grid(ax2,'on'); box(ax2,'on');
plot(ax2, Eg, Ng, 'k-', 'LineWidth', 1, 'DisplayName','Mapa');
scatter(ax2, Eg, Ng, 30, err_after, 'filled');
colormap(ax2, parula);
caxis(ax2, cax);
cb2 = colorbar(ax2); cb2.Label.String = 'error (m)';
title(ax2, 'Odometía CORREGIDA (error en m)');
xlabel(ax2,'E [m]'); ylabel(ax2,'N [m]');
legend(ax2,'Location','best');


uu=0;




% ====== FIGURA N–U (antes / después) ======
figNU = figure('Name','Error sobre trayectoria N-U (antes / después)');
tlNU  = tiledlayout(figNU, 1, 2, 'TileSpacing','compact', 'Padding','compact');

% misma escala de colores cax ya calculada arriba
% --- IZQUIERDA: sin corregir ---
axNU1 = nexttile(tlNU, 1);
hold(axNU1,'on'); axis(axNU1,'equal'); grid(axNU1,'on'); box(axNU1,'on');
plot(axNU1, Ng, Ug, 'k-', 'LineWidth', 1, 'DisplayName','Mapa');
scatter(axNU1, Ng, Ug, 30, err_before, 'filled');
colormap(axNU1, parula);
caxis(axNU1, cax);
cbNU1 = colorbar(axNU1); cbNU1.Label.String = 'error (m)';
title(axNU1, 'Odometía SIN corregir (plano N-U)');
xlabel(axNU1,'N [m]'); ylabel(axNU1,'U [m]');
legend(axNU1,'Location','best');

% --- DERECHA: corregida ---
axNU2 = nexttile(tlNU, 2);
hold(axNU2,'on'); axis(axNU2,'equal'); grid(axNU2,'on'); box(axNU2,'on');
plot(axNU2, Ng, Ug, 'k-', 'LineWidth', 1, 'DisplayName','Mapa');
scatter(axNU2, Ng, Ug, 30, err_after, 'filled');
colormap(axNU2, parula);
caxis(axNU2, cax);
cbNU2 = colorbar(axNU2); cbNU2.Label.String = 'error (m)';
title(axNU2, 'Odometía CORREGIDA (plano N-U)');
xlabel(axNU2,'N [m]'); ylabel(axNU2,'U [m]');
legend(axNU2,'Location','best');

%% =========================================================================
%% EXPORT PARA PIPELINE PYTHON (Bi-LSTM compensador)
%% =========================================================================
% Genera un CSV por sesión con todos los datos sincronizados que el
% Bi-LSTM necesita para entrenarse en el pipeline Python posterior.

fprintf('\n=== Exportando CSV para pipeline Python ===\n');

% 1) Definir variables auxiliares: pg = q acumulado del path GPS;
%    Eg, Ng, Ug = coordenadas ENU del path GPS (vienen de CoordTruth)
%    q_odo, q_odo_corr, q_truth están a frecuencia del IMU (longitud N)

% 2) Construir posición BASELINE = forma GPS evaluada en q_odo_corr
%    Esto da las coordenadas (E,N,U) del monopatín según la odometría corregida
E_base = interp1(pg, Eg, q_odo_corr, 'linear', 'extrap');
N_base = interp1(pg, Ng, q_odo_corr, 'linear', 'extrap');
U_base = interp1(pg, Ug, q_odo_corr, 'linear', 'extrap');

% 3) Construir posición TRUTH = forma GPS evaluada en q_truth (la referencia)
E_truth_imu = interp1(pg, Eg, ctx.q_truth, 'linear', 'extrap');
N_truth_imu = interp1(pg, Ng, ctx.q_truth, 'linear', 'extrap');
U_truth_imu = interp1(pg, Ug, ctx.q_truth, 'linear', 'extrap');

% 4) RESIDUAL = truth - baseline (lo que la Bi-LSTM debe predecir)
res_E = E_truth_imu - E_base;
res_N = N_truth_imu - N_base;
res_U = U_truth_imu - U_base;

% 5) Flag de marca manual (1 si hay marca dentro de 0.5 s de la muestra)
mark_flag = zeros(size(t));
if exist('t_mark','var') && ~isempty(t_mark)
    for ii = 1:numel(t_mark)
        [~, k] = min(abs(t - t_mark(ii)));
        if abs(t(k) - t_mark(ii)) < 0.5
            mark_flag(k) = 1;
        end
    end
end

% 6) Asegurar que todas las columnas tengan la misma longitud N
N_imu = numel(t);
assert(all(cellfun(@numel, {ax, ay, az, wx, wy, wz, ax_lin, ay_lin, az_lin}) == N_imu), ...
    'Longitudes de IMU no coinciden con t');

% 7) Construir tabla
T_out = table(t(:), ax(:), ay(:), az(:), wx(:), wy(:), wz(:), ...
    ax_lin(:), ay_lin(:), az_lin(:), ...
    ctx.qodo(:), q_odo_corr(:), ctx.q_truth(:), ...
    E_truth_imu(:), N_truth_imu(:), U_truth_imu(:), ...
    E_base(:), N_base(:), U_base(:), ...
    res_E(:), res_N(:), res_U(:), ...
    mark_flag(:), ...
    'VariableNames', {'t','ax','ay','az','wx','wy','wz', ...
    'axL','ayL','azL', ...
    'q_odo','q_corr','q_truth', ...
    'E_truth','N_truth','U_truth', ...
    'E_base','N_base','U_base', ...
    'res_E','res_N','res_U','mark'});

% 8) Carpeta de salida y escritura
export_dir = fullfile(carpeta_base, '..', 'export_python');
if ~exist(export_dir, 'dir'), mkdir(export_dir); end
sesion_nombre = regexprep(carpeta_sesion, '.*[\\/]', '');
out_name = sprintf('ruta_%s_%s.csv', sesion_nombre, sesion_id);
out_path = fullfile(export_dir, out_name);
writetable(T_out, out_path);

fprintf('✓ CSV exportado: %s\n', out_path);
fprintf('  Filas: %d  |  Columnas: %d  |  Tamaño aprox: %.1f MB\n', ...
    height(T_out), width(T_out), height(T_out)*width(T_out)*8/1024/1024);
fprintf('  Stats: q_truth_final = %.2f m | RMSE baseline-truth (2D) = %.2f m\n', ...
    ctx.q_truth(end), sqrt(mean((res_E.^2 + res_N.^2), 'omitnan')));

% --- FIX strapdown_basico

% ================== SUBFUNCIONES (archivo function) ==================
function T = read_csv_generic(fname)
    % Validación robusta del argumento
    if ~(ischar(fname) || (isstring(fname) && isscalar(fname)))
        error('read_csv_generic:badarg', 'fname debe ser char o string escalar.');
    end
    fname = string(fname);
    if strlength(fname) == 0
        error('read_csv_generic:empty', 'fname vacío (string sin contenido).');
    end
    if ~isfile(fname)
        error('read_csv_generic:notfound', 'No existe el archivo: %s', fname);
    end

    try
        opts = detectImportOptions(fname);
        if isprop(opts,'PreserveVariableNames'), opts.PreserveVariableNames = true; end
        T = readtable(fname, opts);
    catch
        T = readtable(fname);  % fallback
    end
end

function reportMatch(tag, idx, T)
    if isempty(idx)
        fprintf('• %s: NO encontrado\n', tag);
    else
        fprintf('• %s ← columna "%s"\n', tag, T.Properties.VariableNames{idx});
    end
end

function name = varNameOrEmpty(T, idx)
    if isempty(idx), name = ""; else, name = string(T.Properties.VariableNames{idx}); end
end

function idx = pickVarSmart(T, exactList, fuzzyList)
    names = lower(T.Properties.VariableNames);
    idx = [];
    % exacto
    for c = string(exactList)
        k = find(strcmp(names, char(c)), 1, 'first');
        if ~isempty(k), idx = k; return; end
    end
    % fuzzy
    compact = erase(names, ["_","-","(",")","[","]","{","}"]);
    compact = regexprep(compact, '\d', '');
    for c = string(fuzzyList)
        tok = char(erase(regexprep(lower(c),'\d',''),"_"));
        k = find(contains(compact, tok), 1, 'first');
        if ~isempty(k), idx = k; return; end
    end
end

function x = valOrNaNRob(T, idx)
    if isempty(idx), x = nan(height(T),1);
    else, x = toDoubleColumn(T{:,idx});
    end
end

function v = toDoubleColumn(col)
    if iscell(col)
        v = cellfun(@parseNumeric, col);
    elseif isstring(col)
        v = arrayfun(@parseNumeric, cellstr(col));
    elseif iscategorical(col)
        v = arrayfun(@parseNumeric, cellstr(col));
    elseif isa(col,'char')
        v = parseNumeric(cellstr(col));
    elseif isnumeric(col)
        v = double(col);
    else
        try, v = double(col); catch, v = nan(size(col,1),1); end
    end
    v = v(:);
end

function y = parseNumeric(s)
    if iscell(s), s = s{1}; end
    if (isstring(s) && strlength(s)==0) || (ischar(s) && isempty(s))
        y = NaN; return;
    end
    s = string(strtrim(s));
    if contains(s, ".") && contains(s, ",")
        lastDot = find(s=='.', 1, 'last'); lastCom = find(s==',', 1, 'last');
        if lastCom > lastDot, s = replace(s, ".", ""); s = replace(s, ",", ".");
        else, s = replace(s, ",", "");
        end
    elseif contains(s, ",")
        s = replace(s, ",", ".");
    end
    s = regexprep(s, '[^\d\.\-\+eE]', '');
    if strlength(s)==0, y = NaN; else, y = str2double(s); end
end

function [a, unit, converted] = ensureAngleRadians(a)
    unit = 'rad'; converted = false;
    if all(isnan(a)), return; end
    a_clean = a(isfinite(a));
    if isempty(a_clean), return; end
    if max(abs(a_clean)) > 3*pi || (max(a_clean)-min(a_clean)) > (2*pi+0.5)
        a = deg2rad(a); unit = 'rad (desde °)'; converted = true;
    end
end

function [w, unit, converted] = ensureRadians(w)
    unit = 'rad/s'; converted = false;
    if all(isnan(w)), return; end
    w_clean = w(isfinite(w));
    if isempty(w_clean), return; end
    if max(abs(w_clean)) > 50
        w = deg2rad(w); unit = 'rad/s (desde °/s)'; converted = true;
    end
end

function [ax, ay, az, accUnits, units_state] = normalize_acc_units(ax,ay,az, name_ax,name_ay,name_az, g0)
    accUnits    = 'm/s^2';
    units_state = 'mps2';
    if any(contains(lower([string(name_ax) string(name_ay) string(name_az)]), "_g"))
        ax = ax*g0; ay = ay*g0; az = az*g0;
        accUnits    = 'm/s^2 (desde g)';
        units_state = 'from_g_by_name';
        return;
    end
    amag = sqrt(ax.^2 + ay.^2 + az.^2); mA = median(amag(isfinite(amag)));
    if mA < 3
        if mA < 1.62
            ax = ax*g0; ay = ay*g0; az = az*g0;
            accUnits    = 'm/s^2 (desde g)';
            units_state = 'from_g_by_mag';
        else
            units_state = 'mps2_linear';
        end
    else
        units_state = 'mps2';
    end
end

function R = rot_from_two_vectors(u, v)
    u = u(:); v = v(:);
    if any(~isfinite([u; v])) || norm(u)==0 || norm(v)==0, R = eye(3); return; end
    u = u / norm(u); v = v / norm(v);
    c = max(-1, min(1, dot(u, v)));
    if c > 1 - 1e-12, R = eye(3); return; end
    if c < -1 + 1e-12
        [~, idx] = min(abs(u)); e = zeros(3,1); e(idx) = 1;
        k = cross(u, e); k = k / norm(k);
        K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];
        R = eye(3) + 2*(K*K);  % 180°
        return;
    end
    k = cross(u, v); s = norm(k);
    K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];
    R = eye(3) + K + (K*K) * ((1 - c)/(s^2));
end

function add_markers(axs, tmarks, labels, colorRGB)
    if nargin<4 || isempty(colorRGB), colorRGB = [0 0 0]; end
    if isempty(tmarks) || ~all(isfinite(tmarks)), return; end
    for a = axs
        axes(a); %#ok<LAXES>
        hold on;
        for i=1:numel(tmarks)
            xline(tmarks(i),':','Color',colorRGB,'LineWidth',0.8);
        end
        if ~isempty(labels)
            idxShow = 1:min(10,numel(tmarks)); 
            for i=idxShow
            yl = ylim;
                text(tmarks(i), yl(2), [' ' char(labels(i))], ...
                    'Color', colorRGB, 'VerticalAlignment','top', 'Rotation',0, 'FontSize',8);
            end
        end
    end
end
