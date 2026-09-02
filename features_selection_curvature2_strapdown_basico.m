function [F, idxF, ESEN] = features_selection_curvature2_strapdown_basico(q, rho, rhop, etaM, opts)
% FEATURES_SELECTION_CURVATURE2_STRAPDOWN_BASICO

% FIX (una línea): fs se deriva de dq = median(diff(q))
%
% - Segmenta y detecta picos sobre la señal elegida (por defecto ESEN = movmean(|rhop|)).
% - Los features se toman en los máximos de la señal de detección (ESEN).
%
% F  = [q_peak, rho(q_peak), rhop(q_peak)]

    if nargin < 5 || ~isstruct(opts), opts = struct(); end
    q   = q(:); rho = rho(:); rhop = rhop(:);
    ok  = isfinite(q) & isfinite(rho) & isfinite(rhop);
    q   = q(ok); rho = rho(ok); rhop = rhop(ok);
    N   = numel(q);
    if N < 3 || ~isfinite(etaM) || etaM <= 0
        F = zeros(0,3); idxF = []; ESEN = zeros(0,1); return;
    end

    % -------- parámetros --------
    W_m          = getf(opts,'W_m',          2);
    Dmin_m       = getf(opts,'Dmin_m',       4); % 12
    prom_rel     = getf(opts,'prom_rel',     0.20);
    prom_absFrac = getf(opts,'prom_absFrac', 0.50);
    dilate_frac  = getf(opts,'dilate_frac',  0.10);% 0.25
    eps_over     = getf(opts,'eps_over',     0.05);
    peak_on      = upper(string(getf(opts,'peak_on','ESEN')));  % 'ESEN'|'ABS'
    doPlot       = logical(getf(opts,'doPlot', false));

    % -------- a muestras --------
    dq   = median(diff(q));  if ~isfinite(dq) || dq<=0, dq = 1; end
    W_n  = max(7, 2*floor(round(W_m/dq)/2)+1);      % ventana impar
    Dmin = max(1, round(Dmin_m/dq));
    L    = max(1, round(dilate_frac * Dmin));

    % -------- señales --------
N = numel(rhop);
if N > 9
    rhop(1:4) = rhop(5);rhop(N-3:N) = rhop(N-4);
else
    warning('El vector es muy corto para hacer ese reemplazo.');
end

    absr = abs(rhop);
fs = 1/dq;   fc = 0.1*fs;         % [muestras/m]
n  = 4;Wn = fc/(fs/2);
[b,a] = butter(n, Wn, 'low');
absr = filtfilt(b, a, absr);   % y es tu señal


%plot(absr);hold on;plot(absrx )
    ESEN =absr;% movmean(absr, W_n, 'Endpoints','shrink');   % roja

    % señal para buscar picos
    if peak_on == "ABS"
        sig = absr;
    else
        sig = ESEN;   % por defecto, ESEN
    end

    % -------- máscara por umbral y dilatación --------
    rawMask = sig >= etaM;
    mask    = dilate_mask(rawMask, L);

    % -------- segmentación --------
    dmask = diff([false; mask; false]);
    ibeg  = find(dmask == 1);
    iend  = find(dmask == -1) - 1;

    % -------- picos por segmento (siempre forzando columna) --------
    idxF = zeros(0,1);   % columna
for s = 1:numel(ibeg)
    ii   = ibeg(s):iend(s);
    nseg = numel(ii);
    if nseg < 3, continue; end

    seg    = sig(ii);
    segMax = max(seg);
    segMed = median(seg);

    prom_min = max(prom_absFrac*etaM, prom_rel*max(segMax - segMed, eps));

    Dseg = round(0.3 * Dmin);      
    Dseg = max(Dseg, 1);           % al menos 1
    Dseg = min(Dseg, nseg-1);      
    % ------------------------

    [pk, locs, ~, proms] = findpeaks(seg, ...
        'MinPeakHeight',   etaM, ...
        'MinPeakDistance', Dseg);

    if isempty(locs), continue; end
    keep = (proms >= prom_min) | (pk >= (1+eps_over)*etaM);
    locs = locs(keep);
    if isempty(locs), continue; end

    cand = ii(locs(:));
    idxF = [idxF; cand(:)];
end



% salida
    idxF = unique(idxF(:), 'stable');
    if isempty(idxF)
        F = zeros(0,3);
    else
        F = [q(idxF), rho(idxF), rhop(idxF)];
    end

    % -------- plot opcional --------
    if doPlot
        figure('Name','|rho''(s)| y ESEN (features en ESEN)'); hold on; grid on; box on;
        plot(q, absr, 'b-', 'LineWidth',1.2, 'DisplayName','|\rho''(s)| (azul)');
        plot(q, ESEN, 'Color',[0.85 0.33 0.10], 'LineWidth',1.2, 'DisplayName','ESEN (movmean, roja)');
        yline(etaM, 'k--', 'LineWidth',1.1, 'DisplayName','\eta_M');
        if ~isempty(idxF)
            scatter(q(idxF), ESEN(idxF), 42, 'filled', ...
                'MarkerFaceColor',[1 .7 0], 'MarkerEdgeColor','k', ...
                'DisplayName','features (en ESEN)');
        end
        xlabel('s [m]'); ylabel('|\rho''(s)| [1/m^2]');
        title('|\rho''(s)| y umbral \eta_M — picos tomados en ESEN');
        legend('Location','best');
    end
end

% ===== helpers =====
function v = getf(S, name, defaultVal)
    if ~isempty(S) && isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        v = S.(name);
    else
        v = defaultVal;
    end
end

function y = dilate_mask(x, L)
    x = logical(x(:));
    if L <= 0, y = x; return; end
    if exist('movmax','file') == 2
        y = movmax(x, [L L]);
    else
        k = ones(2*L+1,1);
        y = conv(double(x), k, 'same') > 0;
    end
    y = logical(y);
end
