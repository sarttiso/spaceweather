% This function returns the hdf fields of an hdf file from the ACE level 2
% datasets.
%
% IN:
% file: hdf file name to read 
%
% OUT:
% fields: cell array of field names
%
% Adrian Tasistro-Hart, 02.10.2018


function fields = ACEhdf_fields(file,varargin)

% parse inputs
parser = inputParser;
addRequired(parser,'file',@ischar)

parse(parser,file,varargin{:})

file = parser.Results.file;

% get hdf file information
finfo = hdfinfo(file);

fields = {finfo.Vgroup(1).Vdata.Fields.Name};

end