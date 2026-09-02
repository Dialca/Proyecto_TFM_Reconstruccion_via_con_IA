function odo = cargar_odometro_strapdown_basico(odo_file, varargin)
% CARGAR_ODOMETRO_STRAPDOWN_BASICO
% Carga un CSV de odómetro/Hall (p.ej. "hall_0XX.csv") SIN diálogo
% (uigetfile) y reproduce la misma cadena de cálculo que la sección
% "4) (OPCIONAL) CARGAR ODOMETRO" de Algoritmo_2.m:
%   pulsos -> (incremental o acumulado) -> bineado a bin_dt -> distancia/vel.
%
% Uso:
%   odo = cargar_odometro_strapdown_basico(odo_file)
%   odo = cargar_odometro_strapdown_basico(odo_file, 'wheel_radius_m', 0.14, ...
%                                           'ticks_per_rev', 6, 'bin_dt', 0.5, ...
%                                           'dt_guess_odo', 0.5)
%
% Entradas:
%   odo_file       : ruta al CSV del odómetro (típicamente "hall_0XX.csv",
%                    una sola columna "pulses": pulsos contados en cada
%                    bin de dt_guess_odo segundos; para CSVs
%                    con columna de tiempo explícita).
%   'wheel_radius_m' : radio de la rueda [m]            (default 0.14)
%   'ticks_per_rev'  : pulsos por vuelta de rueda        (default 6)
%   'bin_dt'         : ancho de bin para integrar/contar [s] (default 0.5)
%   'dt_guess_odo'   : paso temporal asumido si el CSV NO trae columna de
%                      tiempo (default 0.5)
%
% Salida (struct odo):
%   .t          : centros de bin [s]              (vector columna)
%   .v          : velocidad por bin [m/s]          (= speed_bin)
%   .dist_odo   : distancia acumulada [m]           (= cum_dist)
%   .pulses_bin : pulsos sumados por bin
%   .dist_bin   : distancia recorrida en cada bin [m]
%   .t_raw      : tiempo de cada muestra cruda (antes de binear) [s]
%   .pulses_raw : pulsos crudos (tal como se leyeron) por muestra
%   .params     : struct con los parámetros usados (wheel_radius_m, etc.)

    p = inputParser;
    addParameter(p, 'wheel_radius_m', 0.14);
    addParameter(p, 'ticks_per_rev', 6);
    addParameter(p, 'bin_dt', 0.5);
    addParameter(p, 'dt_guess_odo', 0.5);
    parse(p, varargin{:});
    wheel_radius_m = p.Results.wheel_radius_m;
    ticks_per_rev  = p.Results.ticks_per_rev;
    bin_dt         = p.Results.bin_dt;
    dt_guess_odo   = p.Results.dt_guess_odo;

    wheel_circumference_m = 2*pi*wheel_radius_m;

    odo = struct('t', [], 'v', [], 'dist_odo', [], 'pulses_bin', [], ...
                 'dist_bin', [], 't_raw', [], 'pulses_raw', [], 'params', struct( ...
                 'wheel_radius_m', wheel_radius_m, 'ticks_per_rev', ticks_per_rev, ...
                 'bin_dt', bin_dt, 'dt_guess_odo', dt_guess_odo, ...
                 'wheel_circumference_m', wheel_circumference_m));

    assert(exist(odo_file, 'file') == 2, 'cargar_odometro_strapdown_basico: no existe el archivo "%s".', odo_file);

    To = readtable(odo_file);
    if isempty(To)
        warning('cargar_odometro_strapdown_basico:ArchivoVacio', '"%s" no tiene filas.', odo_file);
        return;
    end

    names_o = lower(string(To.Properties.VariableNames));

    % --- columna de tiempo  ---
    it_o = find(contains(names_o, ["t", "time", "tiempo", "timestamp", "sec", "seconds", "ts"]) & ...
                ~contains(names_o, "pulse"), 1);
    
    if width(To) < 2
        it_o = [];
    end

    % --- columna de pulsos ---
    ip = find(contains(names_o, ["pulse", "tick", "count", "odo", "odomet", "encoder", "enc"]), 1);
    if isempty(ip)
        ip = width(To);   % por defecto, última (o única) columna
    end

    if isempty(it_o)
        n = height(To);
        t_odo = (0:n-1).' * dt_guess_odo;
        fprintf('ODO: "%s" sin columna de tiempo -> se asume dt=%.3f s/fila (%d filas).\n', odo_file, dt_guess_odo, n);
    else
        t_odo = double(To{:, it_o});
        dto = median(diff(t_odo(isfinite(t_odo))));
        if isfinite(dto) && dto > 1 && max(t_odo, [], 'omitnan') > 1000
            t_odo = t_odo * 1e-3;
            fprintf('ODO: "%s" tiempo en ms -> convertido a s.\n', odo_file);
        end
    end

    pulses_raw = double(To{:, ip});

    % --- limpiar / ordenar ---
    good = isfinite(t_odo) & isfinite(pulses_raw);
    t_odo = t_odo(good); pulses_raw = pulses_raw(good);
    [t_odo, ord] = sort(t_odo); pulses_raw = pulses_raw(ord);

    odo.t_raw = t_odo(:);
    odo.pulses_raw = pulses_raw(:);

    if numel(t_odo) < 2
        warning('cargar_odometro_strapdown_basico:PocasMuestras', '"%s" tiene menos de 2 muestras válidas.', odo_file);
        return;
    end

    % --- ¿incremental o acumulado? ---
    is_accum = all(diff(pulses_raw) >= -eps);
    if is_accum
        odo_count = pulses_raw;
        dp_odo = [0; diff(odo_count)];
    else
        dp_odo = pulses_raw;
    end

    % --- bineado a bin_dt ---
    t0 = ceil(min(t_odo)/bin_dt) * bin_dt;
    t1 = floor(max(t_odo)/bin_dt) * bin_dt;
    if t1 <= t0
        t0 = min(t_odo); t1 = max(t_odo);
    end
    edges = t0:bin_dt:t1;

    if numel(edges) < 2
        warning('cargar_odometro_strapdown_basico:PocosBins', '"%s": muy pocos datos para generar bins de %.2f s.', odo_file, bin_dt);
        return;
    end

    tb = edges(1:end-1) + bin_dt/2;
    binIdx = discretize(t_odo, edges);
    m = isfinite(binIdx) & isfinite(dp_odo);
    pulses_bin = accumarray(binIdx(m), dp_odo(m), [numel(edges)-1, 1], @sum, 0);

    dist_bin = (pulses_bin / max(ticks_per_rev, eps)) * wheel_circumference_m;  % m/bin
    speed_bin = dist_bin / bin_dt;                                              % m/s
    cum_dist = cumsum(dist_bin);                                                % m

    odo.t = tb(:);
    odo.v = speed_bin(:);
    odo.dist_odo = cum_dist(:);
    odo.pulses_bin = pulses_bin(:);
    odo.dist_bin = dist_bin(:);

    fprintf('ODO: "%s" -> %d muestras crudas | %d bins de %.2fs | dist=%.3f m | v_media=%.3f m/s | v_mediana=%.3f m/s\n', ...
        odo_file, numel(t_odo), numel(tb), bin_dt, cum_dist(end), ...
        mean(speed_bin, 'omitnan'), median(speed_bin, 'omitnan'));
end
