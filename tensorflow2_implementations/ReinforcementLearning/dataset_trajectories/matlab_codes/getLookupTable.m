function lookuptab = getLookupTable(positions, vertical, actions)
v1=0:(positions-1);
v2 = reshape(v1, vertical, []).';
for k=2:2:positions/vertical
    v2(k,:) = fliplr(v2(k,:));
end
%v3 = fliplr(v2)';
lookuptab = zeros(positions, actions);
for k=1:positions
    [r,c,~] = find(v2==k-1);
    if r==1 % can't go backward
         lookuptab(k,1) = v2(r,c);
    else
        disp(r)
        disp(c)
        lookuptab(k,1) = v2(r-1,c);
    end
    if r == positions/vertical % can't go forward
        lookuptab(k,2) = v2(r,c);
    else
        lookuptab(k,2) = v2(r+1,c);
    end
    if c == 1 % can't go left
        lookuptab(k,4) = v2(r,c);
    else
        lookuptab(k,4) = v2(r,c-1);
    end
    if c == vertical % can't go right
        lookuptab(k,3) = v2(r,c);
    else
        lookuptab(k,3) = v2(r,c+1);
    end
end
save('lookuptab2','lookuptab','-v7.3')