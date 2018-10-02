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


function [dat,t] = ACEparse(file,fields,varargin)
    

end