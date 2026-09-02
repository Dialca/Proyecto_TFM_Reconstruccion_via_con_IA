function [q, rho, rhop] = curvature_profile_taubin3D(E, N, Up, varargin)
% Curvatura local (3D->plano) con SVD por ventana, continuidad de base y filtros

p = inputParser;
p.addParameter('dmin',   3,  @(x)isnumeric(x)&&isscalar(x));
p.addParameter('win_m', 70,  @(x)isnumeric(x)&&isscalar(x));   % ↑ ventana por estabilidad
p.addParameter('sg_win', 21,  @(x)isnumeric(x)&&isscalar(x));  % impar
p.addParameter('Rmin',  15,  @(x)isnumeric(x)&&isscalar(x));   % radio mínimo plausible [m]
p.addParameter('planarityRatio', 0.2, @(x)isnumeric(x)&&isscalar(x)); % S3/S1 umbral
p.parse(varargin{:});
dmin  = p.Results.dmin;  win_m = p.Results.win_m;
sgwin = p.Results.sg_win; if mod(sgwin,2)==0, sgwin=sgwin+1; end
Rmin  = p.Results.Rmin;  pr    = p.Results.planarityRatio;

E = E(:); N = N(:); Up = Up(:);

% Filtrado por proximidad en planta (respecto al punto anterior)
keep = true(numel(E),1);
for k=2:numel(E)
    if keep(k-1) && hypot(E(k)-E(k-1), N(k)-N(k-1)) < dmin
        keep(k) = false;
    end
end
E = E(keep); N = N(keep); Up = Up(keep);

% Abscisa q en planta
dq = hypot(diff(E), diff(N));
q  = [0; cumsum(dq)];
Npts = numel(q);

rho = nan(Npts,1);
Vprev = [];   % para continuidad de base

for k=1:Npts
    idx = find(q >= q(k)-win_m & q <= q(k)+win_m);
    if numel(idx) < 8, continue; end

    P  = [E(idx), N(idx), Up(idx)];
    mu = mean(P,1);
    Pc = P - mu;

    % SVD local
    [~,Ssvd,V] = svd(Pc,'econ');
    % Continuidad de base: alinear con Vprev para evitar flips
    if ~isempty(Vprev)
        for c=1:3
            if dot(V(:,c), Vprev(:,c)) < 0
                V(:,c) = -V(:,c);
            end
        end
    end
    Vprev = V;

    XYp = Pc * V;         % Nwin x 3
    XY2 = XYp(:,1:2);

    % Si poca planaridad, cae a 2D EN
    if Ssvd(3,3)/Ssvd(1,1) > pr
        XY2 = [E(idx)-mean(E(idx)), N(idx)-mean(N(idx))];
    end

    Pare = CircleFitByTaubin(XY2);     % [xc,yc,R]
    Rloc = Pare(3);
    if ~isfinite(Rloc) || Rloc<=0
        continue;
    end

    % Clipping físico
    if Rloc < Rmin
        Rloc = Rmin;
    end

    % Signo en 2D local
    j = find(idx==k,1,'first');
    if isempty(j) || j<=1 || j>=numel(idx)
        sgn = 0;
    else
        v1 = XY2(j,:)   - XY2(j-1,:);
        v2 = XY2(j+1,:) - XY2(j,:);
        crossz = v1(1)*v2(2) - v1(2)*v2(1);
        sgn = sign(crossz);
    end
    rho(k) = sgn * (1/Rloc);
end

% Derivada d(rho)/dq con SG (más estable que gradient)
rhop = gradient(rho, q);
try
    rhop = sgolayfilt(rhop, 3, sgwin);
end                                                                                                                                                                                   
end