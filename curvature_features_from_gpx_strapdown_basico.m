function [F, idxF,p,t,CoordTruth] = curvature_features_from_gpx_strapdown_basico(gpxFile)
% CURVATURE_FEATURES_FROM_GPX_STRAPDOWN_BASICO — Curvatura y rasgos a partir de un archivo GPX (robusto)
%
% Salidas:
%   F    : tabla de features [q, rho, rhop, E,N,U, lat, lon, idx]
%   rho  : curvatura en rejilla de arco q
%   rhop : derivada d(rho)/dq
%   out  : struct (q, E,N,U, lat, lon, t)
%

% ---------- Parámetros ----------
ds_remuestreo = 1.0;   % [m]
Rmin          = 5;     % [m]
winSG_pts     = 31;    % impar
sg_poly       = 3;
do_plots      = true;

assert(exist(gpxFile,'file')==2,'No existe el archivo: %s', gpxFile);

%% Leer GPX
try
    Data = gpxread(gpxFile);
    Lat  = Data.Latitude(28:end)';     % recorte del archivo
    Lon  = Data.Longitude(28:end)';
    H    = Data.Elevation(28:end)';
    t    = []; if isfield(Data,'Time'); t = Data.Time(:); end
catch ME
    error('Lectura GPX falló: %s', ME.message);
end

% Si hay muy pocos puntos
if numel(Lat) < 2
    [F, rho, rhop, out] = salida_vacia(Lat, Lon, H, t);
    return;
end

%% LLA->ENU
lat0 = Lat(1); lon0 = Lon(1); h0 = H(1);
wgs84 = wgs84Ellipsoid('meter');
[E,N,U] = geodetic2enu(Lat, Lon, H, lat0, lon0, h0, wgs84, 'degrees');

%% Limpieza suave
E = smoothdata(filloutliers(E,'linear','movmedian',440), 'sgolay',  100);
N = smoothdata(filloutliers(N,'linear','movmedian',440), 'sgolay',  100);
U = smoothdata(filloutliers(U,'linear','movmedian',440), 'sgolay',  100);
% plot(Ex);hold on;plot(E)
% figure
% plot(Nx);hold on;plot(N)
% figure
% plot(Ux);hold on;plot(U)
% Distancia acumulada y únicos
p0 = [0; cumsum(hypot(diff(E), diff(N)))];
[p0u, iu] = unique(p0, 'stable');
Eu = E(iu); Nu = N(iu); Uu = U(iu); LatU = Lat(iu); LonU = Lon(iu);

% Si después de unique quedan <2 puntos
if numel(p0u) < 2 || p0u(end) <= 0
    [F, rho, rhop, out] = salida_minima(Eu, Nu, Uu, LatU, LonU, t);
    return;
end

%% Remuestreo uniforme en arco
pg = (0:ds_remuestreo:p0u(end)).';
Eg = interp1(p0u, Eu, pg, 'pchip', 'extrap');
Ng = interp1(p0u, Nu, pg, 'pchip', 'extrap');
Ug = interp1(p0u, Uu, pg, 'pchip', 'extrap');

% Remuestreo produjo >=2 puntos
if numel(pg) < 2
    [F, rho, rhop, out] = salida_minima(Eg, Ng, Ug, Lat, Lon, t);
    return;
end

%% Curvatura 2D simple como magnitud (respaldo general)
%% --- Curvatura (3D -> plano local) por Taubin + SVD ---
winSG = 31;                % ventana SG interna de tu función (impar)
winM  = 50;                % semiventana física [m]
Rmin  = 5;                 % radio mínimo plausible [m]

[p, rho_raw, ~] = curvature_profile_taubin3D(Eg, Ng, Ug, ...
    'dmin',2, 'win_m',winM, 'sg_win',winSG, 'Rmin',Rmin, 'planarityRatio',0.15);

%% Signo robusto con derivadas E(s), N(s)
dsu = max(1e-6, median(diff(p)));
if numel(p) < 2
    [F, rho, rhop, out] = salida_minima(Eg, Ng, Ug, Lat, Lon, t);
    return;
end
su = (p(1):dsu:p(end)).';
if numel(su) < 2
    su = p; % grid mínimo
end

% Interpola E,N a su (si se puede)
if numel(pg) >= 2
    E_u = interp1(pg, Eg, su, 'pchip','extrap');
    N_u = interp1(pg, Ng, su, 'pchip','extrap');
else
    E_u = Eg; N_u = Ng; su = pg;
end

% Suavizado y derivadas
[E0, poly_su, frame_su] = sgolay_safe(E_u, sg_poly, winSG_pts);
[N0, ~,       ~       ] = sgolay_safe(N_u, sg_poly, winSG_pts);
[dE, d2E] = deriv_sg_or_grad(E0, su, poly_su, frame_su);
[dN, d2N] = deriv_sg_or_grad(N0, su, poly_su, frame_su);

v2  = dE.^2 + dN.^2;
den = (v2 .* sqrt(v2));  den(den < 1e-9) = 1e-9;
k_s = (dE .* d2N - dN .* d2E) ./ den;

% Histéresis del signo
tau = 2e-4;
k_abs_s = try_sgolay_abs(k_s, min(21, odd_leq(numel(k_s))));
sgn = sign(k_s);
for i=2:numel(sgn)
    if k_abs_s(i) < tau || sgn(i)==0
        sgn(i) = sgn(i-1);
    end
end

% Proyectar signo a p y combinar con magnitud
if numel(su) >= 2
    sgn_on_p = interp1(su, sgn, p, 'nearest','extrap');
else
    sgn_on_p = ones(size(p));
end
rho_mag  = min(abs(rho_raw), 1/Rmin);     % recorte anti-espurios
rho = rho_mag(:) .* sgn_on_p(:);

%% dρ/dq y suavizado leve
ds_p = max(1e-6, median(diff(p)));
rhop = gradient(rho, ds_p);
wr = odd_leq(numel(rhop));
if wr >= 7 && exist('sgolayfilt','file')==2
    rhop = sgolayfilt(rhop, 3, min(wr,21));
end

%% Selección de rasgos
abs_rhop = abs(rhop);
etaM = median(abs_rhop) + 0.5*1.4826*mad(abs_rhop,1);

etaM = median(rhop) + .5*1.4826*mad(rhop,1);
%etaM = quantile(rhop, 0.7);
%etaM = 2e-4;   % umbral típico para |rho'| 

[F, idxF, ESEN] = features_selection_curvature2_strapdown_basico(p, rho, rhop, etaM , struct( ...
    'Dmin_m',12, 'prom_rel',0.20, 'prom_absFrac',0.50, 'dilate_frac',0.25, 'doPlot',false));
%if isempty(ESEN)
    %ESEN = abs_rhop;
%end

E_at = interp1(pg, Eg, p(idxF), 'linear','extrap');
N_at = interp1(pg, Ng, p(idxF), 'linear','extrap');
U_at = interp1(pg, Ug, p(idxF), 'linear','extrap');
lat_at = interp1(p0u, LatU, p(idxF), 'linear','extrap');
lon_at = interp1(p0u, LonU, p(idxF), 'linear','extrap');



%% 10) Plots
if do_plots && numel(p) >= 2
s        = p(:);                 % abscisa
rho      = rho(:);
abs_rhop = abs(rhop(:));

q_map = p(idxF);                     
ctx.map = table(q_map,'VariableNames',{'q'});

% E(s), N(s) para pintar trayectoria al mismo s
E_at_s = interp1(pg, Eg, s, 'pchip', 'extrap');
N_at_s = interp1(pg, Ng, s, 'pchip', 'extrap');

fig = figure('Name','Curvatura y |rho''| con rasgos');
tl  = tiledlayout(fig, 2, 2, 'TileSpacing','compact','Padding','compact');

% rho(s) — arriba-izquierda
ax1 = nexttile(tl, 1); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
plot(ax1, s, rho, 'LineWidth', 1.4, 'DisplayName','\rho(s)');
if ~isempty(idxF)
    scatter(ax1, s(idxF), rho(idxF), 42, 'filled', 'DisplayName','features (idxF)');
end
xlabel(ax1,'s [m]'); ylabel(ax1,'\rho [1/m]'); title(ax1,'\rho(s) con features');
legend(ax1,'Location','best');

% |rho'(s)| — abajo-izquierda (alineado con el de arriba)
ax2 = nexttile(tl, 3); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
plot(ax2, s, abs_rhop, 'b-', 'LineWidth', 1.2, 'DisplayName','|\rho''(s)|');
plot(ax2, s, ESEN, 'Color',[0.85 0.33 0.10], 'LineWidth', 1.2, 'DisplayName','ESEN (movmean)');
yline(ax2, etaM, 'k--', 'LineWidth', 1.1, 'DisplayName','\eta_M');
if ~isempty(idxF)
    
    scatter(ax2, s(idxF), ESEN(idxF), 42, 'filled', ...
        'MarkerFaceColor',[1 .7 0], 'MarkerEdgeColor','k', ...
        'DisplayName','features (en ESEN)');
end
xlabel(ax2,'s [m]'); ylabel(ax2,'|\rho''(s)| [1/m^2]');
title(ax2,'|\rho''(s)|, ESEN y umbral \eta_M'); legend(ax2,'Location','best');

% Alinear ejes x de (1) y (2)
linkaxes([ax1, ax2],'x');

% Trayectoria E–N — derecha (ocupa dos filas)
ax3 = nexttile(tl, 2, [2 1]); hold(ax3,'on'); grid(ax3,'on'); axis(ax3,'equal'); box(ax3,'on');
plot(ax3, E_at_s, N_at_s, '-', 'LineWidth', 1.2, 'DisplayName','Trayectoria');
if ~isempty(idxF)
    scatter(ax3, E_at_s(idxF), N_at_s(idxF), 45, 'filled', 'DisplayName','features (idxF)');
end
xlabel(ax3,'E [m]'); ylabel(ax3,'N [m]'); title(ax3,'idxF proyectados en E–N');
legend(ax3,'Location','best');
end
uu=0;


% ===== salidas =====
CoordTruth = struct();
CoordTruth.pg   = pg;
CoordTruth.Eg   = Eg;
CoordTruth.Ng   = Ng;
CoordTruth.Ug   = Ug;
CoordTruth.p0u  = p0u;
CoordTruth.LatU = LatU;
CoordTruth.LonU = LonU;
CoordTruth.lat0 = lat0;
CoordTruth.lon0 = lon0;
CoordTruth.h0   = h0;


end % ===== FIN PRINCIPAL =====

% ================== AUXILIARES ==================
function [F, rho, rhop, out] = salida_vacia(Lat, Lon, H, t)
F   = table('Size',[0,9],'VariableTypes',repmat("double",1,9), ...
      'VariableNames',{'q','rho','rhop','E','N','U','lat','lon','idx'});
rho = zeros(0,1); rhop = rho;
out.q = []; out.rho = []; out.rhop = [];
out.E = []; out.N = []; out.U = [];
out.lat = Lat(:); out.lon = Lon(:); out.t = t;
end

function [F, rho, rhop, out] = salida_minima(Eg, Ng, Ug, Lat, Lon, t)
F   = table('Size',[0,9],'VariableTypes',repmat("double",1,9), ...
      'VariableNames',{'q','rho','rhop','E','N','U','lat','lon','idx'});
rho = zeros(numel(Eg),1); rhop = zeros(numel(Eg),1);
out.q = (0:max(0,numel(Eg)-1)).'; out.rho = rho; out.rhop = rhop;
out.E = Eg(:); out.N = Ng(:); out.U = Ug(:);
out.lat = Lat(:); out.lon = Lon(:); out.t = t;
end

function [rho, rhop] = curvature2D_simple(E, N, q)
q = q(:); E = E(:); N = N(:);
if numel(q) < 3
    rho = zeros(size(q)); rhop = zeros(size(q)); return;
end
dE  = gradient(E, q);  dN  = gradient(N, q);
d2E = gradient(dE, q); d2N = gradient(dN, q);
den = (dE.^2 + dN.^2).^(3/2) + eps;
rho = (dE .* d2N - dN .* d2E) ./ den;
rhop = gradient(rho, q);
end

function [y, poly_out, frame_out] = sgolay_safe(x, poly_in, frame_in)
N = numel(x);
if N < 3, y = x; poly_out = 0; frame_out = 1; return; end
frame = round(frame_in); if mod(frame,2)==0, frame = frame+1; end
frame = min(frame, odd_leq(N)); frame = max(frame, 3);
poly  = round(poly_in); poly = max(1, min(poly, frame-1));
if exist('sgolayfilt','file')==2
    y = sgolayfilt(x, poly, frame);
else
    y = movmean(x, frame);
end
poly_out = poly; frame_out = frame;
end

function [d1, d2] = deriv_sg_or_grad(x, s, poly, frame)
x = x(:); s = s(:); N = numel(x);
if N < 3, d1 = zeros(N,1); d2 = zeros(N,1); return; end
ds = max(1e-9, median(diff(s)));
poly_d  = max(2, min(poly, frame-1));
frame_d = max(3, min(odd_leq(N), frame));
if exist('sgolay','file')==2 && poly_d >= 2 && frame_d >= 3
    try
        [~,G] = sgolay(poly_d, frame_d);
        d1 = conv(x, (1/ds)   * G(:,2), 'same');
        d2 = conv(x, (2/ds^2) * G(:,3), 'same');
        return;
    catch
    end
end
d1 = gradient(x, s);
d2 = gradient(d1, s);
end

function w = odd_leq(N), w = 2*floor((N-1)/2)+1; end

function y = try_sgolay_abs(x, w)
x = abs(x); w = max(3, min(odd_leq(numel(x)), w));
if exist('sgolayfilt','file')==2, y = sgolayfilt(x,3,w); else, y = movmean(x,w); end
end
