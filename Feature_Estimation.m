%% ================= FUNCIONES LOCALES =================
function [features, idx_feat] = Feature_Estimation(t, rho, drho_dt, eta_l)
%FEATURE_ESTIMATION  Detecta tramos donde |drho_dt| supera eta_l y
% toma el máximo absoluto dentro de cada tramo (marca de entrada/salida).
%
% Entrada:
%   t        : Nx1 tiempo (o abscisa)
%   rho      : Nx1 curvatura
%   drho_dt  : Nx1 derivada de la curvatura
%   eta_l    : umbral escalar sobre |drho_dt|
%
% Salida:
%   features : tabla con columnas:
%              t_star, rho_star, rhop_star, idx_star
%   idx_feat : vector de indices idx_star

    % --- columnas y NaN-safe
    t       = t(:);
    rho     = rho(:);
    rhop = sqrt(movmean(drho_dt.^2, 50));  % envolvente RMS (pasa-bajos)     Filtrado derivada Rho       
    ok = isfinite(t) & isfinite(rho) & isfinite(rhop);
    t(~ok)    = []; 
    rho(~ok)  = []; 
    rhop(~ok) = [];
    N = numel(t);
    if N < 2
        features = table([],[],[],[], 'VariableNames',{'t_star','rho_star','rhop_star','idx_star'});
        idx_feat = [];
        return;
    end

    % --- variables del barrido
    in_run = false;               % estamos dentro de un tramo |rhop| >= eta_l
    qmax   = 0;                   % maximo de |rhop| en el tramo actual
    t_star = NaN; rho_star = NaN; rhop_star = NaN; idx_star = NaN;

    t_list = []; rho_list = []; rhop_list = []; idx_list = [];

    for k = 2:N
        % Inicio de tramo: cruce desde |rhop| < eta_l a >= eta_l
        if ~in_run && (abs(rhop(k-1)) < eta_l) && (abs(rhop(k)) >= eta_l)
            in_run    = true;
            qmax      = abs(rhop(k));
            t_star    = t(k);
            rho_star  = rho(k);
            rhop_star =  drho_dt(k);
            idx_star  = k;
        end

        if in_run
            % Actualiza máximo dentro del tramo
            if abs(rhop(k)) > qmax
                qmax      = abs(rhop(k));
                t_star    = t(k);
                rho_star  = rho(k);
                rhop_star = drho_dt(k);
                idx_star  = k;
            end

            % Fin de tramo: vuelve a caer por debajo del umbral
            if abs(rhop(k)) < eta_l
                % guarda el pico del tramo
                t_list(end+1,1)    = t_star;    %#ok<AGROW>
                rho_list(end+1,1)  = rho_star;  %#ok<AGROW>
                rhop_list(end+1,1) = rhop_star; %#ok<AGROW>
                idx_list(end+1,1)  = idx_star;  %#ok<AGROW>

                % resetea estado
                in_run = false;
                qmax   = 0;
                t_star = NaN; rho_star = NaN; rhop_star = NaN; idx_star = NaN;
            end
        end
    end

    % Si terminamos dentro de un tramo, también lo cerramos
    if in_run && ~isnan(idx_star)
        t_list(end+1,1)    = t_star;
        rho_list(end+1,1)  = rho_star;
        rhop_list(end+1,1) = rhop_star;
        idx_list(end+1,1)  = idx_star;
    end

    features = table(t_list, rho_list, rhop_list, idx_list, ...
        'VariableNames', {'t_star','rho_star','rhop_star','idx_star'});
    idx_feat = idx_list;
end
