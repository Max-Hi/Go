#TODO: Data comes in looking like this: B[dq];W[pd];B[cd];W[pp];B[nc];W[qf];B[fc];W[do];B[dl];W[gp];B[eo];W[en];B[fo];W[dp];B[fq];W[cq];B[eq];W[fn];B[go];W[cr];B[dn];W[dm];B[cn];W[cm];B[bo];W[co];B[bn];W[bm];B[bp];W[cp];B[bq];W[br];B[em];W[bk];B[ck];W[bl];B[ar];W[an];B[pe];W[el];B[fm];W[gn];B[qe];W[gm];B[ip];W[qd];B[rd];W[oe];B[pf];W[rc];B[re];W[oc];B[nd];W[od];B[rb];W[ne];B[ld];W[md];B[mc];W[me];B[kd];W[qc];B[fl];W[gl];B[fk];W[gk];B[gj];W[hj];B[fj];W[hi];B[qi];W[ci];B[dh];W[jc];B[jd];W[gd];B[ic];W[gc];B[fb];W[fd];B[dc];W[fh];B[cj];W[di];B[bj];W[bi];B[ai];W[dj];B[dk];W[cg];B[am];W[al];B[dr];W[ao];B[aq];W[as];B[bs];W[cs];B[ds];W[as];B[bh];W[ap];B[ch];W[ei];B[aj];W[de];B[ce];W[kq];B[jr];W[df];B[eg];W[fg];B[ef];W[dd];B[ff];W[cc];B[bc];W[cb];B[db];W[bb];B[bd];W[ib];B[jb];W[id];B[kc];W[bg];B[ab];W[ag];B[ah];W[hf];B[he];W[ie];B[hg];W[gf];B[fe];W[ed];B[ge];W[if];B[dg];W[hc];B[cf];W[lb];B[mb];W[jc];B[kb];W[ob];B[kf];W[lg];B[kg];W[kh];B[ic];W[ho];B[hp];W[jc];B[jh];W[ig];B[ic];W[hr];B[ir];W[jc];B[lh];W[ki];B[ic];W[gq];B[io];W[jc];B[le];W[lf];B[ic];W[hn];B[fr];W[jc];B[mh];W[ng];B[ic];W[ec];B[eb];W[jc];B[ji];W[kj];B[ic];W[gr];B[hq];W[jc];B[jj];W[ic];B[kk];W[lj];B[la];W[ja];B[na];W[nb];B[ml];W[mj];B[jg];W[jk];B[nh];W[mg];B[kl];W[oh];B[ni];W[nk];B[oi];W[jl];B[om];W[pk];B[pl];W[ol];B[ql];W[qk];B[nl];W[ok];B[mk];W[nj];B[rk];W[pm];B[rl];W[nm];B[on];W[km]
# and it needs to be a 4x19x19 tensor. 
# I only want to do that calculation once, so I need to preprocess here.