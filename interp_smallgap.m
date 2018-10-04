% This function interpolates small gaps in time series. Gaps are understood
% to be small sequences of NaNs for which data was unavailable. Assumes
% that data are evenly spaced.
%
% IN:
% dat: vector of data for which the small gaps should be interpolated. Can
%   also be array, with data in columns and rows corresponding to
%   observations. In this case, missing data need to occur simultaneously
%   in each column.
% gap: maximum gap size in units of smaples to interpolate
% 'ndseq': array of missing data sequences, with beginning indices of 
%   missing data in the first column, end indices of missing data in the 
%   second column, and the number of missing data in the given gap in the 
%   third column. Thus, rows in ndseq correspond to gaps. This output is 
%   the 2nd, 3rd, and 4th columns produced by function findseq(). If not
%   provided, then will be automatically generated.
%
% OUT:
% dat: data vector with small gaps interpolated
%
% TO DO:
% - generalize to unevenly spaced data vectors
% - allow for interpolation of matrix with data in columns missing values
%   from same rows 
%
% Adrian Tasistro-Hart, 02.10.2018

function dat = interp_smallgap(dat,gap,varargin)

parser = inputParser;
addRequired(parser,'dat',@isnumeric);
addRequired(parser,'gap',@isscalar);
addParameter(parser,'ndseq',[],@isnumeric);

parse(parser,dat,gap,varargin{:});

dat = parser.Results.dat;
gap = parser.Results.gap;
ndseq = parser.Results.ndseq;

% make column
if isvector(dat)
    dat = dat(:);
end
% number of data
ndat = size(dat,1);

% if ndseq not provided, then compute
if isempty(ndseq)
    ndseq = findseq(dat);
    ndseq = ndseq(isnan(ndseq(:,1)),2:4);
else
    assert(size(ndseq,2) == 3, 'ndseq must have 3 columns')
end

% get all nans in data
nandatidx = isnan(dat(:,1));

% get indices of big gaps 
bigidx = find(ndseq(:,3) > gap);
datbigidx = false(ndat,1);
for ii = 1:length(bigidx)
    datbigidx(ndseq(bigidx(ii),1):ndseq(bigidx(ii),2)) = true;
end

% get indices of small gaps in gap array
smallidx = find(ndseq(:,3) <= gap); 
% generate logical indices into data vector; these are the locations where
% data will be interpolated. i.e. these are the indices into the nans
% corresponding to small gaps
datsmallidx = false(ndat,1);
for ii = 1:length(smallidx)
    datsmallidx(ndseq(smallidx(ii),1):ndseq(smallidx(ii),2)) = true;
end

% these are the indices of single isolated nans, i.e. gap of one
singlenanidx = nandatidx & ~datbigidx;

% since we want to interpolate single nans as well, include them in small
% gaps
datsmallidx = singlenanidx | datsmallidx;

% create temporary time vector
t = 1:ndat;
% interpolate values
datint = interp1(t(~datsmallidx),dat(~datsmallidx,:),t(datsmallidx),...
    'linear');
% replace corresponding nans
dat(datsmallidx,:) = datint;

end