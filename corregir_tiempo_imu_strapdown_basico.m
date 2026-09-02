function [t_corr, factor, info] = corregir_tiempo_imu_strapdown_basico(t_raw, timestamps_file, factor_fallback)
% CORREGIR_TIEMPO_IMU_STRAPDOWN_BASICO
% Corrige el eje de tiempo crudo del IMU a partir de un archivo de
% checkpoints "timestamps_0XX.csv" (columnas: index, seconds).

%       factor = true_span / raw_span
%       t_corr = factor .* t_raw
%
% Entradas:
%   t_raw            : vector de tiempo crudo del IMU [s], ANTES de corregir.
%   timestamps_file   : ruta a "timestamps_0XX.csv" (columnas index,seconds).
%                       Puede ser "" o no existir; en ese caso se usa el fallback.
%   factor_fallback    : (opcional) factor a usar si no hay archivo válido.
%                       Default: 2.05 (valor histórico de Algoritmo_2.m).
%
% Salidas:
%   t_corr : t_raw .* factor
%   factor : factor efectivamente aplicado
%   info   : struct con detalles del cálculo:
%            .used_fallback  (logical)
%            .true_span      [s] (NaN si no se pudo calcular)
%            .raw_span       [s]
%            .n_checkpoints  (num. de checkpoints válidos leídos)
%            .message        (texto explicativo / motivo del fallback)

    if nargin < 3 || isempty(factor_fallback)
        factor_fallback = 2.05;
    end
    if nargin < 2
        timestamps_file = "";
    end

    t_raw = t_raw(:);
    raw_span = max(t_raw) - min(t_raw);

    info = struct('used_fallback', true, 'true_span', NaN, 'raw_span', raw_span, ...
                   'n_checkpoints', 0, 'message', '');

    % ---- ¿Hay archivo de timestamps utilizable? ----
    if strlength(string(timestamps_file)) == 0 || exist(char(timestamps_file), 'file') ~= 2
        factor = factor_fallback;
        info.message = sprintf(['No se encontró archivo de timestamps ("%s"); ' ...
            'se usa factor_fallback = %.4f.'], char(string(timestamps_file)), factor_fallback);
        warning('corregir_tiempo_imu_strapdown_basico:SinArchivo', '%s', info.message);
        t_corr = factor .* t_raw;
        return;
    end

    try
        Tts = readtable(char(timestamps_file));
    catch ME
        factor = factor_fallback;
        info.message = sprintf(['No se pudo leer "%s" (%s); ' ...
            'se usa factor_fallback = %.4f.'], char(timestamps_file), ME.message, factor_fallback);
        warning('corregir_tiempo_imu_strapdown_basico:LecturaFallida', '%s', info.message);
        t_corr = factor .* t_raw;
        return;
    end

    % ---- Localizar columna "seconds" ----
    vn = lower(string(Tts.Properties.VariableNames));
    icol = find(vn == "seconds" | contains(vn, "second"), 1);
    if isempty(icol) && width(Tts) >= 2
        icol = 2;   % fallback razonable: index,seconds -> 2da columna
    end

    if isempty(icol)
        factor = factor_fallback;
        info.message = sprintf(['"%s" no tiene una columna "seconds" reconocible; ' ...
            'se usa factor_fallback = %.4f.'], char(timestamps_file), factor_fallback);
        warning('corregir_tiempo_imu_strapdown_basico:SinColumnaSeconds', '%s', info.message);
        t_corr = factor .* t_raw;
        return;
    end

    seconds_checkpoints = Tts{:, icol};
    seconds_checkpoints = seconds_checkpoints(isfinite(seconds_checkpoints));
    info.n_checkpoints = numel(seconds_checkpoints);

    if info.n_checkpoints < 2
        factor = factor_fallback;
        info.message = sprintf(['"%s" tiene menos de 2 checkpoints "seconds" válidos; ' ...
            'se usa factor_fallback = %.4f.'], char(timestamps_file), factor_fallback);
        warning('corregir_tiempo_imu_strapdown_basico:PocosCheckpoints', '%s', info.message);
        t_corr = factor .* t_raw;
        return;
    end

    true_span = seconds_checkpoints(end) - seconds_checkpoints(1);
    info.true_span = true_span;

    % ---- Validar spans antes de dividir ----
    if ~isfinite(true_span) || true_span <= 0 || ~isfinite(raw_span) || raw_span <= 0
        factor = factor_fallback;
        info.message = sprintf(['Span inválido (true_span=%.4f s, raw_span=%.4f s) en "%s"; ' ...
            'se usa factor_fallback = %.4f.'], true_span, raw_span, char(timestamps_file), factor_fallback);
        warning('corregir_tiempo_imu_strapdown_basico:SpanInvalido', '%s', info.message);
        t_corr = factor .* t_raw;
        return;
    end

    % ---- Factor derivado de los datos ----
    factor = true_span / raw_span;
    info.used_fallback = false;
    info.message = sprintf(['Factor derivado de "%s": true_span=%.4f s, raw_span=%.4f s ' ...
        '-> factor=%.6f.'], char(timestamps_file), true_span, raw_span, factor);
    fprintf('%s\n', info.message);

    t_corr = factor .* t_raw;
end
