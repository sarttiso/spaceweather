% This function parses ACE hdf data, returning requested fields as well as
% a time vector for the requested data.
%
% IN:
% file: string of file name to load
% fields: cell array of field names to extract from the hdf file.
%
% OUT:
% dat: data corresponding to requested fields, with each field as a column
% t: if desired, matlab time vector of measurements
%
% TO DO:
%
% Adrian Tasistro-Hart, 02.10.2018


function [dat,t] = ACEhdf_parse(file,fields,varargin)

% parse inputs
parser = inputParser;
addRequired(parser,'file',@ischar)
addRequired(parser,'fields',@iscell)

parse(parser,file,fields,varargin{:})

file = parser.Results.file;
fields = parser.Results.fields;

% get hdf file information
finfo = hdfinfo(file);
% now load data
dat = hdfread(finfo.Vgroup(1).Vdata,...
    'Fields',fields);
% convert to numeric
dat = cell2mat(dat)';

% if user requested time, get it
if nargout > 1
    % load time data
    curt = hdfread(finfo.Vgroup(1).Vdata,...
        'Fields',{'year','day','hr','min','sec'});
    % convert DOY to mm, dd
    [mm,dd] = ddd2mmdd(double(curt{1}'),double(curt{2}'));
    % create time vector
    t = double([curt{1}',mm,dd,curt{3}',curt{4}',curt{5}']);
    % convert to matlab time
    t = datenum(t);
end

end