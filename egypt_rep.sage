# egypt_rep.sage (c) 2021,25 by Max Alekseyev

__doc__ = r'''
Compute representations of a given rational number as Egyptian fractions,
utilizing parallelization via Sage's recursively enumerable sets (RES) functionality.

It provides the following general functions:
* all_rep(s, N): computes the length-`N` representations of a rational number `s`.
* count_rep(s, N): counts the length-`N` representations of a rational number `s`.
* exist_rep(s, N): tests if there exists a length-N representation of a rational number `s`.

* all_cf_rep(s, cf): computes the representations of `s` with numerators given by list `cf`.
* count_cf_rep(s, cf): counts the representations of `s` with numerators given by list `cf`.

It further provides functions for computing representations corresponding to the fusion ring types used in the paper:
"Classifying integral Grothendieck rings up to rank 5 and beyond" https://arxiv.org/abs/2507.07023

* count_rep_MNSD(s, N): same as above under MNSD (maximally non self-dual) constraint.
* all_rep_MNSD(s, N): same as above under MNSD (maximally non self-dual) constraint.

Functions may have optional parameters not mentioned above.

-----------------------

Brief history:
* 20250720 - first public release
'''

######################################################## Global parameters and imports
# Ed Bertram wants all denominators to divide the last one
BERTRAM = False

import itertools
import multiprocessing


######################################################## Approaches
r'''
We have
p/q = 1/a + 1/b
or more generally: p/q = n_a/a + n_b/b

Two equivalent forms:

(I) p*a*b = q*(a+b), or more generally p*a*b = q*(a*n_b + b*n_a)
If a = g*A and b = g*B with (A,B) = 1
p*A*B*g = q*(A+B)

p | (A+B),          A*B | q


(II) (p*a - q) * (p*b - q) = q^2

Under BERTRAM, we assume that b is the largest denominator and thus a|b:
(BI) a = g, A = 1, and
p*B*g = q*(1+B)

p | (1+B),          B | q | g*B = b

(BII)   (p*a - q) | q
'''


######################################################### Auxiliary functions

def gen_divisors(n, F=None):
    if F is None:
        F = list( factor(n) )
    #print(n,F)
    if not F:
        yield n
        return
    p,d = F.pop()
    yield from gen_divisors(n,F)
    for i in range(d):
        n //= p
        yield from gen_divisors(n,F)
    F.append( (p,d) )

def unitary_divisors(n):
    fn = factor(n)
    D = {p:(k-1) for p,k in fn if k>1}
    return sorted( prod(p^k for p,k in D.items() if d%p==0)*d for d in divisors(prod(p for p,_ in fn)) )


######################################################### Approach (Ia)

def complete1trial(p,q,L,U,n_a=1,n_b=1,odd_only=False):
    '''
    Find a,b based on trial division within the given interval [L,U].
    Solve p/q = n_a/a + n_b/b, that is, p*a*b = q*(n_b*a + n_a*b) for a <= b.
    Under BERTRAM: a|b and q|b and b|q*n_b*a.
    '''

    S = []

    # we have (p*a - q*n_a)*b = q*n_b*a   ==>     (p*a - q*n_a) | q*n_b*a
    # under BERTRAM: (p*a - q*n_a) | n_b*a and (p*a - q*n_a) | q*n_b

    # we want p*a - q*n_a >= 1, i.e. a >= (q*n_a+1)/p
    L = max(L, q*n_a//p + 1)

    step = 1
    if odd_only:
        if L%2==0:
            L += 1
        step *= 2

    # we want a <= b = q*n_b*a / (p*a - q*n_a), i.e. p*a <= (n_a+n_b)*q
    # under BERTRAM: a <= q*n_a / (p - n_b)

    U = min(U, (n_a+n_b)*q//p)
    if BERTRAM and p>n_b:
        U = min(U, n_a*q//(p-n_b))

    for a in range(L,U+1,step):
        d = p*a - q*n_a
        z = divmod(q*a*n_b,d)
        if z[1]:
            continue
        S.append( (a,z[0]) )

    return S


######################################################### Approach (Ib)


def complete1fact(p,q,L,U,n_a=1,n_b=1):
    '''
    Find a,b based on integer factorization of q.
    Solve p/q = n_a/a + n_b/b, that is, p*a*b = q*(n_b*a + n_a*b) for a <= b.
    Under BERTRAM: q|b and a|b.
    '''

    # let g := gcd(a,b) and (A,B) := (a,b)/g
    # then p*A*B*g = q*(n_b*A + n_a*B). Notice that it's possible that gcd(n_b*A + n_a*B, g) > 1.
    # We have A | q*n_a and B | q*n_b  ===>  A*B | lcm(n_a,n_b)*q

    # under BERTRAM: g = a and A = 1  and q | g*B =====>  p*B*g = q*(n_b + n_a*B)     ===>   B | q*n_b,  q | B*g
    #                p | (n_b + n_a*B),       B | q*n_b,    q*(n_b + n_a*B) is a square
    #                (p*g - q*n_a)*B = q*n_b
    # Approach: iterate over B satisfying B | q*n_b, and compute g = ((n_b*q/B) + n_a*q)/p

    # below we use a,b in place of A,B !!!

    S = []

    if BERTRAM:
        # n_b*q/(p*L - q*n_a) >= bb >= n_b*q/(p*U^2 - q*n_a)
        # for co-factor: (p*L^2 - q*n_a) <= n_b*q / bb <= (p*U^2 - q*n_a)
        #bbound = n_b*q//(p*L - q*n_a)

        # this takes ~ numdiv(n_b*q) iterations
        #for bb in divisors( n_b*q ):                           # too much memory
        #for bb in divisors_below( n_b*q, bbound ): 
        for bb in gen_divisors(n_b*q):
            a,r = divmod(n_b*q//bb + q*n_a,p)
            if r:
                continue
            if a and a>=L and a<=U:
                b = bb*a
                S.append((a,b))
        return S

    qq = lcm(n_a,n_b)*q

    #cnt = set()

    # this takes ~ numdiv(qq)/2 iterations (each p^d contributes a factor 1+2d)
    qqud = unitary_divisors(qq)
    for aa,bb in zip(qqud,reversed(qqud)):     # aa * bb = qq are unitary co-divisors
        r = radical(aa)
        for d in divisors(aa//r):
            a = r*d
            if a>min(U,bb):     # a is too large
                break
            # b goes over divisors of bb, or altermatively it's a solution to b == -n_b/n_a * a (mod p)   # <--------------------- CAN SPEED UP if p is large???
            for b in reversed(divisors(bb)):
                if b<a:         # b is too small
                    break
                #cnt |= {(a,b)}
                z = divmod( n_b*a + n_a*b, p )
                if z[1]:
                    continue
                g = z[0]*q//(a*b)
                #print(aa,bb,a,b,g)
                if g and a*g>=L and a*g<=U:
                    S.append( (a*g,b*g) )

    #print("Iterations:",cnt)
    return S


######################################################### Approach (II)


def complete2fact2(p,q,L,U,n_a=1,n_b=1):
    '''
    Solve p/q = n_a/a + n_b/b in a <= b via representation: (p*a - q) * (p*b - q) = q^2.
    Under BERTRAM: (p*a - q) * (p*b/q - 1) = q.
    '''

    q2 = q if BERTRAM else q^2

    S = []
    LL = p*L - q
    UU = p*U - q
    for d in divisors(q2):
        if d < LL:              # to make sure a>=L
            continue
        if d > UU:
            break

        if BERTRAM:
            if d*(p-1)>q:            # here we have p|(d+q), d|(d+q), gcd(p,d)=1 ==> (p*d)|(d+q) ==> (p-1)*d<=q
                break                # alternatively, q = d*(p*b/q - 1) >= d*(p-1)
        elif d>q:                    # to make sure a <= b
            break

        a,r = divmod(d+q,p)
        if r==0:
            b = (q^2//d+q)//p
            S.append( (a,b) )
    return S


#########################################################################################################################


def res_rep(s, N, ODD_ONLY = False, MIN_DENOM = 1, MAX_DENOM = +oo, DISTINCT = False):
    '''
    Represent `s` as Egyptian fractions of length `N`.
    '''

    # MIN_DENOM = max(MIN_DENOM, ceil(1/s))

    if ODD_ONLY and MIN_DENOM%2==0:
        MIN_DENOM += 1

    def succ(t):
        s0, m = t
        if s0==0 or len(m)>=N:
            return []

        p = numerator(s0)
        q = denominator(s0)

        if len(m)==N-1:
            if p==1:
                r = q
                if m[-1]+int(DISTINCT)<=r<=MAX_DENOM and (r%2 or not ODD_ONLY):
                    return [(0,m+(r,))]
            return []

        M_D = MAX_DENOM         # local copy of MAX_DENOM

        U_mean = floor((N-len(m))/s0)
        U = min( M_D, U_mean )

        if BERTRAM:
            D = lcm(m)                  # the last denominator must be a multiple of lcm(m)
            DL = D * ceil(U_mean/D)     # the last denominator >= U_mean
            if M_D < +oo:
                M_D = (M_D // D) * D
                if ODD_ONLY and M_D%2==0:
                    M_D -= D
                if DL > M_D:
                    return []
                U = min(U, M_D)
            U = min( U, floor((N-len(m)-1)/(s0-1/DL)) )

        if DISTINCT:
            # Note if ODD_ONLY, m[-1] is odd, and the number of odd elements in [m[-1]+2,M_D] is [ (M_D-m[-1])/2 ]
            if (M_D - m[-1])//(1+int(ODD_ONLY)) < N-len(m):
                return []
            s_ = sum( 1/(M_D-i*(1+int(ODD_ONLY))) for i in range(N-1-len(m)) )
            if s_ >= s0:
                return [(0, m+tuple(M_D-i*(1+int(ODD_ONLY)) for i in range(N-2-len(m),-1,-1)) )] if s_ == s0 else []
            L = max( m[-1]+1, ceil(1/(s0 - s_)) )
        else:
            L = max( m[-1], ceil(1 / (t if (t := s0 - (N-1-len(m))/M_D) else M_D)) )            # 1/x_ + (N-1-m)/MAX_DENOM <= p/q   ========>  x_ > q/p  

        if L > U:
            return []

        if len(m)==N-2:
            if BERTRAM:
                step = 2 if ODD_ONLY else 1

                if (U-L)//step < number_of_divisors(q):
                    return [(0,m+t) for t in complete1trial(p,q,L,U,odd_only=ODD_ONLY) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m) and (not DISTINCT or t[0]<t[1])]
                else:
                    return [(0,m+t) for t in complete1fact(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m) and (not DISTINCT or t[0]<t[1])]
                # return [(0,m+t) for t in use_all([complete2fact,complete2trial,complete2pell],p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m) and (not DISTINCT or t[0]<t[1])]

                #return [(0,m+t) for t in complete2pell(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                #return [(0,m+t) for t in complete2fact(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                #return [(0,m+t) for t in complete2trial(p,q,L,U,odd_only=ODD_ONLY) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]

            # we cannot run complete2fact unconditionally, since (i) q may be hard to factor (deadblock); or (ii) q has too many divisors
            # but deadblock does not happen here since all divisors of q are small

            #if U-L < fact_threshold:
            #    return [(0,m+t) for t in complete2trial(p,q,L,U)]

            if (U-L)*2 < number_of_divisors(q):
            #if True:
                return [(0,m+t) for t in complete1trial(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and (not DISTINCT or t[-1]>t[-2]) and t[-1]<=MAX_DENOM]
            else:
                return [(0,m+t) for t in complete1fact(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and (not DISTINCT or t[-1]>t[-2]) and t[-1]<=MAX_DENOM]

            # return [(0,m+t) for t in use_all([complete2fact,complete2trial],p,q,L,U) if all(e%2 for e in t) or not ODD_ONLY]


            #print(U-L,end=' ')
            #return [(0,m+t) for t in use_all([complete2fact,complete2trial],p,q,L,U)]

            #return [(0,m+t) for t in concurrent_run(complete2fact,complete2trial,p,q,L,U)]
            #return [(0,m+t) for t in complete2trial(p,q,L,U)]

        return ( (s0-1/r, m+(r,)) for r in range(L,U+1) if r%2 or not ODD_ONLY )

    #return RecursivelyEnumeratedSet(seeds=[(s-1/r,(r,)) for r in range(MIN_DENOM,floor(N/s)+1,1+int(ODD_ONLY))], successors=succ, structure='forest')

    S = s if type(s) in [list,tuple] else [s]
    my_seeds = []
    for s in S:
        md = max(MIN_DENOM, ceil(1/s))
        if ODD_ONLY and md%2==0:
            md += 1
        my_seeds.extend( (s-1/r,(r,)) for r in range(md,min(floor(N/s),MAX_DENOM)+1,1+int(ODD_ONLY)) )
    if len(S)>1:
        import random
        random.shuffle(my_seeds)

    return RecursivelyEnumeratedSet(seeds=my_seeds, successors=succ, structure='forest')


def all_rep(s, N, odd_only = False, min_denom = 1, max_denom = +oo, distinct = False, verbose = False):
    def func(t):
        if verbose:
            print('all_rep:', t)
        return {t}
    return res_rep(s,N,ODD_ONLY=odd_only,MIN_DENOM=min_denom,MAX_DENOM=max_denom,DISTINCT=distinct).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N else set(), set.union, set() )

def count_rep(s, N, odd_only = False, min_denom = 1, max_denom = +oo, distinct = False):
    return res_rep(s,N,ODD_ONLY=odd_only,MIN_DENOM=min_denom,MAX_DENOM=max_denom,DISTINCT=distinct).map_reduce(lambda t: int(t[0]==0 and len(t[1])==N))

def exist_rep(s, N, odd_only = False, distinct = False, min_denom = 1, max_denom = +oo):
    # import os, signal
    import psutil, multiprocessing

    q = multiprocessing.Queue()

    def res_search():
        def found(t):
            if t[0]==0 and len(t[1])==N:
                q.put(t[1])
            return 0

        res_rep(s,N,ODD_ONLY=odd_only,DISTINCT=distinct,MIN_DENOM=min_denom,MAX_DENOM=max_denom).map_reduce(found)
        q.put(tuple())

    p1 = multiprocessing.Process(target=res_search)
    p1.start()

    res = q.get()

    #p1.terminate()
    #p1.kill()

    if res:
        # os.system(f'pkill -TERM -P {p1.pid}')         # does not work
        parent = psutil.Process(p1.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass

    return res


#####################################################################################################################
################################################################## (possibly distinct) denominators with coefficients
# MAX_DENOM is not well supported yet

def res_cf_rep(s, cf, ODD_ONLY = False, MIN_DENOM = 1, MAX_DENOM = +oo, PRIME_BOUND = None, DISTINCT = False):

    N = len(cf)

    def succ(t):
        #print(t)
        s0, m = t
        M = len(m)
        if s0==0 or M>=N:
            return []

        p = numerator(s0)
        q = denominator(s0)

        if M==N-1:
            s1 = s0 / cf[M]
            if numerator(s1)==1:
                r = denominator(s1)
                if r>=m[-1]+int(DISTINCT) and r<=MAX_DENOM and (r%2 or not ODD_ONLY) and (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND) and (not BERTRAM or all(r%e==0 for e in m)):
                    return [(0,m+(r,))]
            return []

        M_D = MAX_DENOM

        U_mean = floor(sum(cf[M:])/s0)
        U = min( U_mean, M_D )

        # print([L,U])


        if BERTRAM:
            D = lcm(m)
            DL = D * ceil(U_mean/D)     # the last denominator >= U_mean
            if M_D < +oo:
                M_D = (M_D // D) * D
                if ODD_ONLY and M_D%2==0:
                    M_D -= D
                if DL > M_D:
                    return []
                U = min(U, M_D)
            U = min( U, floor(sum(cf[M:-1])/(s0-cf[-1]/DL)) )

        if DISTINCT:
            if M_D - m[-1] < N-M:
                return []
            L = max( m[-1]+1, ceil(cf[M] / (s0 - sum(c/d for c,d in zip(reversed(cf[M+1:]),range(M_D,0,-1-int(ODD_ONLY)))))) )
        else:
            L = max( m[-1], ceil(cf[M] / (t if (t := s0 - sum(cf[M+1:])/M_D) else M_D)) )            # c[M]/x_ + sum(cf[M+1:])/MAX_DENOM <= p/q   ==>  x_ > ...

        if ODD_ONLY:
            if L%2==0:
                L += 1
            if U%2==0:
                U -= 1

        if U<L:
            return []

        if M==N-2:
            '''
            we have
                p/q = cf[-2]/a + cf[-1]/b
            '''

            if BERTRAM:
                '''
                We have:
                    p*y - q*cf[-2]*(y/x) = q*cf[-1]
                '''
                sq = q*cf[-1]
                step = 2 if ODD_ONLY else 1
                if (U-L)//step < number_of_divisors(sq):
                    t_gen = filter(lambda t: t[1]%t[0]==0 and all(t[1]%e==0 for e in m), complete1trial(p,q,L,U,n_a=cf[-2],n_b=cf[-1],odd_only=ODD_ONLY))
                else:
                    t_gen = complete1fact(p,q,L,U,n_a=cf[-2],n_b=cf[-1])

            else:
                qq = lcm(*cf[-2:])*q
                if (U-L)*2 < number_of_divisors(qq):
                    t_gen = complete1trial(p,q,L,U,n_a=cf[-2],n_b=cf[-1],odd_only=ODD_ONLY)
                else:
                    t_gen = complete1fact(p,q,L,U,n_a=cf[-2],n_b=cf[-1])

            return [(0,m+t) for t in t_gen if t[0]+int(DISTINCT)<=t[1] and (not ODD_ONLY or all(e%2 for e in t)) and (not PRIME_BOUND or max(map(largest_prime_factor,t))<=PRIME_BOUND)]

        return ( (s0-cf[M]/r, m+(r,)) for r in range(L,U+1,1+int(ODD_ONLY)) if (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND) )

    # we have s >= cf[0]/r  ==>  r >= ceil( s/cf[0] )
    MIN_DENOM = max(MIN_DENOM,ceil(cf[0]/s))
    if ODD_ONLY and MIN_DENOM%2==0:
        MIN_DENOM += 1

    return RecursivelyEnumeratedSet(seeds=[(s-cf[0]/r,(r,)) for r in range(MIN_DENOM,floor(sum(cf)/s)+1,1+int(ODD_ONLY)) if (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND)], successors=succ, structure='forest')


def all_cf_rep(s, cf, odd_only = False, min_denom = 1, max_denom = +oo, distinct = False):
    N = len(cf)
    def func(t):
        print(t)
        return {t}
    return res_cf_rep(s,cf,ODD_ONLY=odd_only,MIN_DENOM=min_denom,MAX_DENOM=max_denom,DISTINCT=distinct).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N else set(), set.union, set() )


def count_cf_rep(s, N, odd_only = False, min_denom = 1, distinct = False):
    N = len(cf)
    return res_cf_rep(s,cf,ODD_ONLY=odd_only,MIN_DENOM=min_denom,DISTINCT=distinct).map_reduce(lambda t: int(t[0]==0 and len(t[1])==N))


################################################################ MNSD (maximally non self-dual) constraint
'''
MNSD (maximally non self-dual) constraint implies that we only need to consider the Egyptian fractions with odd denominators
1 = 1/x_1 + ... + 1/x_n
such that x_1 = x_2  <=  x_3 = x_4  <=  x_5 = x_6  <= ....
'''

def res_rep_MNSD(s, N, MIN_DENOM = 1):
    if not BERTRAM:
        raise ValueError('Global parameter BERTRAM must be set to True.')
    return res_cf_rep(s, [2]*(N-1) + [1], ODD_ONLY = True, MIN_DENOM = MIN_DENOM, DISTINCT = False)

def count_rep_MNSD(s, N, min_denom = 1):
    assert N%2
    N_ = (N+1)//2
    return res_rep_MNSD(s,N_,MIN_DENOM=min_denom).map_reduce(lambda t: int(t[0]==0 and len(t[1])==N_))

def all_rep_MNSD(s, N, min_denom = 1):
    assert N%2
    N_ = (N+1)//2
    def func(t):
        print(t)
        return { tuple(sorted(t*2)[:-1]) }
    return res_rep_MNSD(s,N_,MIN_DENOM=min_denom).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N_ else set(), set.union, set())

