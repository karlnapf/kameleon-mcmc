"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

import os
import sys
class LatexTools(object):
    @staticmethod
    def transpose_table(lines):
        """
        Takes a list of strings which are interpreted to be latex tabular lines
        (ending with \\) and transposes the tabular, meaning that the rows
        become the columns and vice versa
        """
        entries=[]
        for i in range(len(lines)):
            if lines[i].strip()!="":
                entries.append(lines[i].strip("\\").split("&"))
        
        
        transposed=[]
        for line_idx in range(len(entries[0])):
            cols=[]
            for col_idx in range(len(entries)):
                cols.append(entries[col_idx][line_idx].strip())
                
            transposed.append(" & ".join(cols) + "\\\\")
        
        return transposed
    
    @staticmethod
    def transpose_table_in_file(filename):
        f=open(filename)
        lines=[line.strip() for line in f.readlines()]
        f.close()
        
        transposed_lines=LatexTools.transpose_table(lines)
        print os.linesep.join(transposed_lines)
        
if __name__ == '__main__':
    if len(sys.argv)!=2:
        print "usage:", str(sys.argv[0]).split(os.sep)[-1], "<filename>"
        print "example:"
        print "python " + str(sys.argv[0]).split(os.sep)[-1] + " table.txt"
        exit()
        
    LatexTools.transpose_table_in_file(str(sys.argv[1]))