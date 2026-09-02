function [matches, f_list, ctx] = postprocess_features(features, out, ctx)
% POSTPROCESS_FEATURES
% Selecciona candidatos, hace matching y corrige la odometría usando q_star.

% ---------- Bootstrap de contexto ----------
if nargin < 3 || isempty(ctx), ctx = struct(); end
if ~isfield(ctx,'q_LM')     || ~isfinite(ctx.q_LM),     ctx.q_LM = -inf; end
if ~isfield(ctx,'margin_q') || ~isfinite(ctx.margin_q), ctx.margin_q = 5;  end   % [m]

if ~isfield(ctx,'F_map')
    error('ctx.F_map no definido (se espera tabla con columnas al menos {qtruth, idx, rho, rhop}).');
end
if ~isfield(out,'t') || isempty(out.t)
    error('out.t requerido.');
end
if ~isfield(out,'v') || isempty(out.v)
    warning('out.v ausente; se usará v=ones para integrar q(t).');
    out.v = ones(size(out.t));
end

% ---------- Odometría base ----------
if ~isfield(ctx,'q') || isempty(ctx.q)
    ctx.q = cumtrapz(out.t(:), out.v(:));
end
q_traj = ctx.q(:);

% odometría corregida
if ~isfield(ctx,'q_tilde') || isempty(ctx.q_tilde)
    ctx.q_tilde = q_traj;
end

% ---------- Preasignaciones ----------
n = height(features);
matches = repmat(canonical_match_struct(), n, 1);
f_list(n,1) = struct('t',NaN,'rho',NaN,'rhop',NaN,'idx',NaN); %%%%%%%%%% ANTERIOR %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NUEVO %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%N = max(1, numel(features)); 
%f_list = repmat(struct('t',NaN,'rho',NaN,'rhop',NaN,'idx',NaN), N, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NUEVO %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------- Bucle principal ----------
for i = 1:n
    % 1) rasgo detectado (IMU)
    t_star    = features.t_star(i);
    rho_star  = features.rho_star(i);
    rhop_star = features.rhop_star(i);
    idx_star  = features.idx_star(i);

    f = struct('t',t_star,'rho',rho_star,'rhop',rhop_star,'idx',idx_star);
    f_list(i) = f;

    % 2) selección de candidatos del mapa
    Fcand = Feature_Selection(t_star, f, out, ctx);

    % 3) barrera previa
    if isfield(ctx,'q_L') && isfinite(ctx.q_L)
        prev_qL = ctx.q_L;
    else
        prev_qL = 0;
        ctx.q_L = 0;
    end

    % ===== MATCHING (Alg. 4 simple) =====
    m_raw = canonical_match_struct();
    % sólo candidatos por delante de la barrera
    FN = Fcand(Fcand.qtruth > ctx.q_L, :);

    % mismo signo
    same_rho  = (FN.rho  .* f.rho  > 0);
    same_rhop = (FN.rhop .* f.rhop > 0);
    FC = FN(same_rho & same_rhop, :);

    if ~isempty(FC)
        [~,k] = min(FC.qtruth);
        m_raw.accepted  = true;
        m_raw.q_matched = FC.qtruth(k);    % <-- q* del MAPA
        if ismember('idx', FC.Properties.VariableNames)
            m_raw.map_idx = FC.idx(k);
        else
            m_raw.map_idx = k;
        end
        m_raw.info = "match Alg4";
    else
        m_raw.info = "sin candidatos";
    end

    % ===== q_star (odometría) en este instante =====
    % si hay mapeo de tiempo; si no, t_star directo
    if isfield(ctx,'T') && isfield(ctx.T,'map')
        t_ref = ctx.T.map(t_star);
    else
        t_ref = t_star;
    end

    q_star = interp1(out.t(:), q_traj, t_ref, 'linear','extrap');   % <-- TU q_star
    m_raw.q_star = q_star;   % lo guardamos para la tabla

    % ===== Tope según siguiente rasgo del mapa =====
    base_q = max(q_star, ctx.q_L);      % base: lo que dice mi odo o la barrera

    Fmap = ctx.F_map;
    ahead = Fmap.qtruth > base_q;

    same_rho_map  = (Fmap.rho  .* f.rho  > 0);
    same_rhop_map = (Fmap.rhop .* f.rhop > 0);
    ahead_same = Fmap(ahead & same_rho_map & same_rhop_map, :);

    if ~isempty(ahead_same)
        q_next = min(ahead_same.qtruth);
    elseif any(ahead)
        q_next = min(Fmap.qtruth(ahead));
    else
        q_next = inf;
    end

    margin   = 20;
    max_cap  = 200;

    if isfinite(q_next)
        gap = q_next - base_q;
        max_ahead_m = min(gap + margin, max_cap);
    else
        max_ahead_m = max_cap;
    end

    if m_raw.accepted && isfinite(m_raw.q_matched)
        if m_raw.q_matched > base_q + max_ahead_m
            m_raw.accepted = false;
            m_raw.info     = "rechazado: muy adelante";
            ctx.q_L        = prev_qL;
        end
    end

    % ===== filtro de similitud + corrección =====
    if m_raw.accepted && isfinite(m_raw.q_matched)
        cand = Fcand(Fcand.qtruth == m_raw.q_matched, :);
        if ~isempty(cand)
            cand = cand(1,:);

            rho_ratio  = abs(f.rho)  / abs(cand.rho);
            rhop_ratio = abs(f.rhop) / abs(cand.rhop);

            if rho_ratio < 0.25 || rhop_ratio < 0.25
                m_raw.accepted = false;
                m_raw.info     = "rechazado: rasgo muy débil respecto al mapa";
                ctx.q_L        = prev_qL;
            else
 
if isfield(ctx,'T') && isfield(ctx.T,'map')
    t_ref = ctx.T.map(t_star);
else
    t_ref = t_star;
end

% odometría en ese instante
q_star = interp1(out.t(:), ctx.q(:), t_ref, 'linear','extrap');

q_truth_here = interp1(out.t(:), ctx.q_truth(:), t_ref, 'linear','extrap');

lambda_t = q_star - q_truth_here;
m_raw.lambda = lambda_t;



if isfield(ctx, 'T') && isfield(ctx.T, 'map') && ~isempty(ctx.T.map)   
    if isa(ctx.T.map, 'function_handle')
        t_query = ctx.T.map(t_star);
    else
        t_query = ctx.T.map;
    end
else
    t_query = t_star;
end

[~, k_odo] = min( abs(out.t(:) - t_query) );

                if k_odo <= numel(ctx.q_tilde)
                    ctx.q_tilde(k_odo:end) = ctx.q_tilde(k_odo:end) - lambda_t;
                end

                ctx.q_L = m_raw.q_matched;
            end
        end
    end

    
    matches(i) = normalize_match_struct(m_raw, f, Fcand, ctx, out);

    % barrera larga
    if matches(i).accepted && isfinite(matches(i).q_matched)
        ctx.q_LM = max(ctx.q_LM, matches(i).q_matched + ctx.margin_q);
    end
end
end


%====================================================================
function s = canonical_match_struct()
s = struct( ...
    'accepted',   false, ...
    'map_idx',    NaN,   ...
    'q_matched',  NaN,   ...
    'score',      NaN,   ...
    'info',       "",    ...
    't_star',     NaN,   ...
    'rho_star',   NaN,   ...
    'rhop_star',  NaN,   ...
    'idx_star',   NaN,   ...
    'q_est',      NaN,   ...
    'lambda',     NaN    ...   % <<< agregado
    );
end

%====================================================================
function mout = normalize_match_struct(minput, f, Fcand, ctx, out)
    mout = canonical_match_struct();

    % rasgo online
    mout.t_star    = f.t;
    mout.rho_star  = f.rho;
    mout.rhop_star = f.rhop;
    mout.idx_star  = f.idx;

    
    if isfield(minput,'map_idx'),   mout.map_idx   = double(minput.map_idx);   end
    if isfield(minput,'q_matched'), mout.q_matched = double(minput.q_matched); end
    if isfield(minput,'accepted'),  mout.accepted  = logical(minput.accepted); end
    if isfield(minput,'info'),      mout.info      = string(minput.info);      end


    if isfield(minput,'q_star') && isfinite(minput.q_star)
        mout.q_est = double(minput.q_star);
    else

        if isfield(ctx,'q_tilde') && numel(ctx.q_tilde)==numel(out.t)
            q_src = ctx.q_tilde;
        else
            q_src = ctx.q;
        end

        % mapear tiempo si hace falta
        if isfield(ctx,'T') && isfield(ctx.T,'map')
            t_ref = ctx.T.map(f.t);
        else
            t_ref = f.t;
        end

        mout.q_est = interp1(out.t(:), q_src(:), t_ref, 'linear','extrap');
    end

    
    if isfield(minput,'lambda') && isfinite(minput.lambda) && mout.accepted
        mout.lambda = double(minput.lambda);
    else
        mout.lambda = NaN;
    end

    % mensaje por defecto
    if strlength(mout.info) == 0
        if mout.accepted
            mout.info = "ok";
        else
            mout.info = "rechazado";
        end
    end
end



%====================================================================
function y = tern(cond, a, b), if cond, y=a; else, y=b; end, end


function Fcand = Feature_Selection(t_star, f, out, ctx)
    Fmap = ctx.F_map;

    % --- 1) odometría de referencia ---
    if isfield(ctx,'q_tilde') && numel(ctx.q_tilde) == numel(out.t)
        q_src = ctx.q_tilde;          
    else
        q_src = ctx.q;                % fallback
    end

    % pasar tiempo IMU -> odómetro si existe mapeo
    if isfield(ctx,'T') && isfield(ctx.T,'map')
        t_ref = ctx.T.map(t_star);
    else
        t_ref = t_star;
    end

    q_est = interp1(out.t(:), q_src(:), t_ref, 'linear','extrap');

    % --- barrera actual ---
    if ~isfield(ctx,'q_L') || ~isfinite(ctx.q_L)
        ctx.q_L = -inf;
    end

    % --- ventana hacia adelante con mínimo fijo ---
    base_q = max(q_est, ctx.q_L);

    min_ahead = 200;     
    q_min = ctx.q_L;     
    q_max = base_q + min_ahead;

    % --- filtrar el mapa ---
    mask = Fmap.qtruth >= q_min & Fmap.qtruth <= q_max;
    Fcand = Fmap(mask, :);
end


% =============== Helpers (subfunciones locales) ==========================
function [dLB, dUB] = get_window_deltas(t, ctx)

dLB = 200; dUB = 200;

hasFnLB = isfield(ctx,'deltaLB_fn') && isa(ctx.deltaLB_fn,'function_handle');
hasFnUB = isfield(ctx,'deltaUB_fn') && isa(ctx.deltaUB_fn,'function_handle');

if hasFnLB, dLB = ctx.deltaLB_fn(t); end
if hasFnUB, dUB = ctx.deltaUB_fn(t); end

% Si no hay funciones, mirar constantes asimétricas
if ~hasFnLB && isfield(ctx,'deltaLB') && ~isempty(ctx.deltaLB)
    dLB = ctx.deltaLB;
end
if ~hasFnUB && isfield(ctx,'deltaUB') && ~isempty(ctx.deltaUB)
    dUB = ctx.deltaUB;
end

% Si tampoco hay constantes, usar simétrica si existe
noFuncs  = ~hasFnLB && ~hasFnUB;
noConsts = ~(isfield(ctx,'deltaLB') && ~isempty(ctx.deltaLB)) ...
    && ~(isfield(ctx,'deltaUB') && ~isempty(ctx.deltaUB));
if noFuncs && noConsts && isfield(ctx,'window_m') && ~isempty(ctx.window_m)
    dLB = ctx.window_m;
    dUB = ctx.window_m;
end

% Sanitizar
dLB = max(0, double(dLB));
dUB = max(0, double(dUB));
if ~isfinite(dLB), dLB = 200; end
if ~isfinite(dUB), dUB = 200; end
end

function v = getfielddef(S, name, default)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    v = S.(name);
else
    v = default;
end
end

%==========================================================================



function match = Feature_Matching(f, Fcand, ctx)
% FEATURE_MATCHING — Algoritmo 4 (paper):
%   Input : f (struct con campos .rho, .rhop, .t opcional), Fcand (tabla con q,rho,rhop), ctx.q_LM
%   Output: match (struct con map_idx, q_star, accepted, score, info)
%
% Pasos:
%   1) Si es la primera iteración, q_L <- 0   (usamos ctx.q_LM si existe)
%   2) F^N <- { f_i in F | q_i > q_L }
%   3) F^C <- { f_i in F^N | rho(q_i)*rho* > 0  ∧  rhop(q_i)*rhop* > 0 }
%   4) Si F^C no es vacío:
%          q* <- min{ q_i | f_i in F^C }     (el más cercano hacia adelante)
%          q_L <- q*    (ACTUALIZA fuera de esta función con ctx.q_LM = match.q_star)
%      Si no:
%          MATCH_NOT_FOUND

% ---- Validaciones ----
if ~(istable(Fcand) && ~isempty(Fcand))
    match = struct('map_idx',NaN,'q_star',NaN,'q_matched',NaN, ...
        'accepted',false,'score',NaN,'info',"Sin candidatos (F vacío)");
    return;
end
need = {'q','rho','rhop'};
if ~all(ismember(need, Fcand.Properties.VariableNames))
    error('Fcand debe tener columnas {"q","rho","rhop"}.');
end
if ~isfield(f,'rho') || ~isfield(f,'rhop')
    error('f debe contener campos .rho y .rhop');
end

% ---- q_L inicial (paper: 0 en primera iteración) ----
if isfield(ctx,'q_LM') && isfinite(ctx.q_LM)
    qL = ctx.q_LM;
else
    qL = 0; 
end


if isfield(ctx,'sign_eps') && ~isempty(ctx.sign_eps)
    sign_eps = ctx.sign_eps;
else
    sign_eps = 0;   % paper: condición estricta > 0
end

% ---- Sanitizar y máscaras lógicas sobre la Fcand original ----
qv   = Fcand.qtruth(:);
rhov = Fcand.rho(:);
rhpv = Fcand.rhop(:);
finite = isfinite(qv) & isfinite(rhov) & isfinite(rhpv);

% ---- F^N: hacia adelante de q_L ----
keepN = finite & (qv > qL);

% ---- F^C: consistencia de signos con f ----
prod_rho  = rhov .* f.rho;
prod_rhop = rhpv .* f.rhop;
keepC = keepN & (prod_rho  > sign_eps) & (prod_rhop > sign_eps);

idxC = find(keepC);
if isempty(idxC)
    % ---- MATCH_NOT_FOUND ----
    match = struct('map_idx',NaN,'q_star',NaN,'q_matched',NaN, ...
        'accepted',false,'score',NaN,'info',"MATCH_NOT_FOUND");
    return;
end

% ---- Elegir q* = min q_i en F^C (más cercano hacia adelante) ----
qC = qv(idxC);
[q_star, krel] = min(qC);
map_idx = idxC(krel);

% score avance respecto a q_L.
score = q_star - qL;

match = struct( ...
    'map_idx',  map_idx, ...
    'q_star',   q_star, ...
    'q_matched',q_star, ...     
    'accepted', true, ...
    'score',    score, ...
    'info',     "OK: forward & sign-consistent" ...
    );
end

function [match, ctx] = Feature_Matching_Alg4_from_subset(f, Fcand, ctx)
if ~isfield(ctx,'q_L') || ~isfinite(ctx.q_L)
    ctx.q_L = 0;
end

% F^N ← { f_i ∈ F | q_i > q_L }
FN = Fcand(Fcand.qtruth > ctx.q_L, :);

% F^C ← { f_i ∈ F^N | ρ(q_i)ρ* > 0 ∧ ρ'(q_i)ρ'* > 0 }
same_rho  = FN.rho  .* f.rho  > 0;
same_rhop = FN.rhop .* f.rhop > 0;
FC = FN(same_rho & same_rhop, :);

match = canonical_match_struct();

if ~isempty(FC)
    [~,k] = min(FC.qtruth);
    match.accepted  = true;
    match.q_matched = FC.qtruth(k);
    match.map_idx   = FC.idx(k);
    match.info      = "match Alg4";
    ctx.q_L = FC.qtruth(k);
else
    match.info = "MATCH_NOT_FOUND";
end
end
