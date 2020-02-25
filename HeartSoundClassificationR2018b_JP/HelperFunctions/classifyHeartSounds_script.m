% CLASSIFYHEARTSOUNDS_SCRIPT classifyHeartSounds から MEX 関数
%  classifyHeartSounds_mex を生成します。
% 
% プロジェクト 'classifyHeartSounds.prj' から 13-Mar-2019 に生成されたスクリプトです。
% 
% CODER、CODER.CONFIG、CODER.TYPEOF、CODEGEN も参照してください。
%Copyright (c) 2016-2019, MathWorks, Inc. 

%% クラス 'coder.MexCodeConfig' の構成オブジェクトを作成します。
cfg = coder.config('mex');
cfg.GenerateReport = true;
cfg.ReportPotentialDifferences = false;

%% エントリ ポイント 'classifyHeartSounds' の引数のタイプを定義します。
ARGS = cell(1,1);
ARGS{1} = cell(2,1);
ARGS{1}{1} = coder.typeof(0,[Inf  1],[1 0]);
ARGS{1}{2} = coder.typeof(0);

%% MATLAB Coder を起動します。
codegen -config cfg -I classifyHeartSounds -args ARGS{1} -nargout 1

