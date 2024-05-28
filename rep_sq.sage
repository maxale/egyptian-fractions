# rep_sq.sage (c) 2021,24 by Max Alekseyev

__doc__ = r'''
Compute representations of a given rational number as Egyptian fractions with squared denominators.
This module utilizes parallelization via Sage's recursively enumerable sets (RES) functionality.

It provides the following general functions:
* all_rep(s, N): computes the length-N representations of rational number s
* count_rep(s, N): counts the length-N representations of rational number s

* all_cf_rep(s, cf): computes the representations of s with numerators given by list cf
* count_cf_rep(s, cf): counts the representations of s with numerators given by list cf

It further provides functions for computing representations corresponding to the fusion ring types used in the paper:
"Classification of integral modular data up to rank 13" https://arxiv.org/abs/2302.01613

* count_rep_MNSD(s, N): same as above under MNSD (maximally non self-dual) constraint
* all_rep_MNSD(s, N): same as above under MNSD (maximally non self-dual) constraint

* all_rep_th87(N): representations of length N satisfying Theorems 8.5 and 8.7
* all_rep_above_r(N): same as above where at least one denominator has prime factor > N
* all_rep_above_half_r(N): same as above where at least one denominator has prime factor > N/2
* all_rep_primes23(N): same as above where denominators are restricted to have prime factors 2 and 3 only

Functions may have optional parameters not mentioned above.

-----------------------

Brief history:
* 20240524 - first public release
* 20240522 - res_cf_rep() was buggy with ODD_ONLY = True
* 20240317 - Fixed bug in complete2trial() when n_b is not coprime to q
'''

###################### Global parameters
# Ed Bertram wants all denominators to divide the last one
BERTRAM = True

#fact_threshold = 10^4  # not used
######################


import itertools
import time
import multiprocessing


'''
Approaches to solving for a and b:
p/q = 1/a^2 + 1/b^2

Two equivalent forms:

(I) p*a^2*b^2 = q*(a^2+b^2)
If a = g*A and b = g*B with (A,B) = 1
p*A^2*B^2*g^2 = q*(A^2+B^2)

p | (A^2+B^2),          A^2*B^2 | q,    and     p*q*(A^2+B^2) is a square


(II) (p*a^2 - q) * (p*b^2 - q) = q^2   <=== this does not make use of the fact that a^2 and b^2 are squares

(III) Quadratic form:
p*(a*b)^2 - q*a^2 - q*b^2 = 0


Under BERTRAM:
(BI) a = g, A = 1, and B^2 | q | (g*B)^2
p*B^2*g^2 = q*(1+B^2)

p | (1+B^2),          B^2 | q,    and     p*q*(1+B^2) is a square      (solve via PELL EQUATION ???)


(BII)   (p*a^2 - q) | q

(BIII) 
p*b^2 - q*(b/a)^2 = q   (same Pell equation)
p*(b/sq)^2 - q/sq^2 * (b/a)^2 = q/sq^2, where q = c * sq^2 with c := core(q).
'''



######################################################### Auxiliary functions


def test(func,B=10^6):
    a = floor(random() * B + 0.5)
    b = floor(random() * B + 0.5)
    pq = 1/a^2 + 1/b^2
    return func(pq.numerator(),pq.denominator(),1,B)


# square part of integer n
def sqpart(n):
    #return Integer( gp.core(n,1)[2] )                # <------------- SLOW
    return prod( p^(d//2) for p,d in factor(n) )

def sqfree(n):
    return sign(n) * prod( p^(d%2) for p,d in factor(n) )

def f_divisors(f):
    return ( Factorization((p,i) for (p,_),i in zip(f,e)) for e in itertools.product(*(range(d+1) for _,d in f)) )

def f_unitary_divisors(f):
    return sorted( Factorization(f).value() for f in Set(f).subsets() )

def unitary_divisors(n):
    fn = factor(n)
    D = {p:(k-1) for p,k in fn if k>1}
    return sorted( prod(p^k for p,k in D.items() if d%p==0)*d for d in divisors(prod(p for p,_ in fn)) )

''' SLOW!!!
def unitary_divisors(n):
    return sorted( Factorization(f).value() for f in Set(factor(n)).subsets() )
'''


# see https://ask.sagemath.org/question/66763/efficient-generation-of-restricted-divisors/
def divisors_below(n,B):
    div = [1]
    for p, e in factor(n):
        ext = div
        for i in range(e):
            ext = [r for d in ext if (r := d * p) <= B]
            div.extend(ext)
    return div

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


def largest_prime_factor(n):
    if n<=0:
        raise ValueError("n must be positive")
    return 1 if n==1 else prime_divisors(n)[-1]


# we need a cached function to avoid memory leak issues with PARI
@cached_function
def is_square_c(t):
    return is_square(t)

# quick test for feasibility of p/q = n_a / a^2 + n_b / b^2
def legendre_condition(p,q,n_a=1,n_b=1):
    if n_a==1 and n_b==1:
        try:
            two_squares(p)
            two_squares(q)
        except ValueError:
            return False
    else:
        # apply Legendre theorem to triple of coefficients (p,-q*n_a,-q*n_b)
        v = vector( (p,-q*n_a,-q*n_b) )
        v /= gcd(v)
        v = list( map(sqfree,v) )
        return is_square_c(Mod(-v[0]*v[1],v[2])) and is_square_c(Mod(-v[0]*v[2],v[1])) and is_square_c(Mod(-v[1]*v[2],v[0]))
    return True


# compute multiplicities
def get_mult(m):
    mult = [1]
    it = iter(m); next(it)
    for e1,e2 in zip(m,it):
        if e1==e2:
            mult[-1] += 1
        else:
            mult.append(1)
    return mult


# test given tuple of denominators for whether it satisfies Th 8.7
def test_th87(t):
    m = sorted(get_mult(t),reverse=True)
    #print(m)

    P = set( prime_divisors(t[-1]>>valuation(t[-1],2)) )
    #if len(P) < len(m):
    #    return False

    # to do: sumbit request for SetPartitions
    return len(P)==0 or any( all(mi >= li for mi,li in zip(m, sorted((lcm(p-1 for p in s)//2 for s in S),reverse=True))) for k in range(1,min(len(m),len(P))+1) for S in SetPartitions(P,k) )


def postprocess_th87(t,cf=None):
    z = sum( ((t_,)*c for t_,c in zip(t,cf)), tuple() ) if cf else t
    if test_th87(z):
        print(z)
        return {z}
    return set()


######################################################### Approach (Ia)


# find a,b based on trial division within the given interval [L,U]
# solve p/q = n_a/a^2 + n_b/b^2, that is, p*a^2*b^2 = q*(n_b*a^2 + n_a*b^2) in  a <= b
# it follows that p*a^2 > q*n_a
# under BERTRAM: a|b and q|b^2, implying (q/sqpart(q))|b. Then (q/sqpart(q))^2|b^2|q*n_b*a^2  ==> (q/sqpart(q)^2)|(n_b*a^2) ==> (q/sqpart(q)^2)|(n_b*a)
def complete2trial(p,q,L,U,n_a=1,n_b=1,odd_only=False):


    if not legendre_condition(p,q,n_a,n_b):
        return []

    S = []

    # we have (p*a^2 - q*n_a)*b^2 = q*n_b*a^2   ==>     (p*a^2 - q*n_a) | q*n_b*a^2
    # under BERTRAM: (p*a^2 - q*n_a) | n_b*a^2 and (p*a^2 - q*n_a) | q*n_b

    # we want p*a^2 - q*n_a >= 1, i.e. a^2 >= (q*n_a+1)/p
    L = max(L, (q*n_a//p).isqrt() + 1)

    if BERTRAM:
        step = sqfree(q)
        step //= gcd(step,n_b)
        if L%step:
            L += step-(L%step)
    else:
        step = 1

    '''
    THIS WAS A BUG:
    step = q//sqpart(q)^2 if BERTRAM else 1            # divisor of a
    if L%step:
        L += step-(L%step)
    '''

    if odd_only:
        if L%2==0:
            L += step                                  # step is odd in this case
        step *= 2

    # we want a^2 <= b^2 = q*n_b*a^2 / (p*a^2 - q*n_a), i.e. p*a^2 <= (n_a+n_b)*q
    # under BERTRAM: a^2 <= q*n_a / (p - n_b)

    U = min(U, ((n_a+n_b)*q//p).isqrt())
    if BERTRAM and p>n_b:
        U = min(U, (n_a*q//(p-n_b)).isqrt())

    # print(L,U,step)

    for a in range(L,U+1,step):
        a2 = a^2
        d = p*a2 - q*n_a                # note that d >= 1 by the updated L
        z = divmod(q*a2*n_b,d)
        if z[1]:
            continue
        b = is_square(z[0],root=True)[1]
        if b:
            S.append( (a,b) )

    return S



######################################################### Approach (Ib)


# find a,b based on integer factorization of q
# solve p/q = n_a/a^2 + n_b/b^2, that is, p*a^2*b^2 = q*(n_b*a^2 + n_a*b^2) in  a <= b
# under BERTRAM: q|b^2 and a|b
def complete2fact(p,q,L,U,n_a=1,n_b=1):

    if not legendre_condition(p,q,n_a,n_b):
        return []

    # let g := gcd(a,b) and (A,B) := (a,b)/g
    # then p*A^2*B^2*g^2 = q*(n_b*A^2 + n_a*B^2). Notice that it's possible that gcd(n_b*A^2 + n_a*B^2,g^2)>1.
    # We have A^2 | q*n_a and B^2 | q*n_b  ===>  A*B | sqpart( lcm(n_a,n_b)*q )

    # under BERTRAM: g = a and A = 1  and q | (g*B)^2 =====>  p*B^2*g^2 = q*(n_b + n_a*B^2)     ===>   B | sqpart( n_b*q ),  sqpart( q ) | B*g
    #                p | (n_b + n_a*B^2),       B^2 | q*n_b,    q*(n_b + n_a*B^2) is a square
    #                (p*g^2 - q*n_a)*B^2 = q*n_b
    # Approach: iterate over B satisfying B^2 | q*n_b, check if g^2 = ((n_b*q/B^2) + n_a*q)/p is an integer square, and compute g

    # below we use a,b in place of A,B !!!

    S = []

    if BERTRAM:
        # n_b*q/(p*L^2 - q*n_a) >= bb^2 >= n_b*q/(p*U^2 - q*n_a)
        # for co-factor: (p*L^2 - q*n_a) / c <= sq^2/bb^2 <= (p*U^2 - q*n_a) / c, where c := core(n_b*q) and n_b*q = c*sq^2
        #bbound = (n_b*q//(p*L^2 - q*n_a)).isqrt()

        # this takes ~ numdiv( sqpart(n_b*q) ) iterations
        #for bb in divisors( sqpart(n_b*q) ):                           # too much memory
        #for bb in divisors_below( sqpart(n_b*q), bbound ): 

        for bb in gen_divisors( sqpart(n_b*q) ):
            g,r = divmod(n_b*q//bb^2 + q*n_a,p)
            if r:
                continue
            a = is_square(g,root=True)[1]
            if a and a>=L and a<=U:
                b = bb*a
                S.append((a,b))
        return S

    qq = sqpart(lcm(n_a,n_b)*q)

    #cnt = set()

    # this takes ~ numdiv( sqpart(q)^2 )/2 iterations (each p^d contributes a factor 1+2d)
    qqud = unitary_divisors(qq)
    for aa,bb in zip(qqud,reversed(qqud)):     # aa * bb = qq are unitary co-divisors
        r = radical(aa)
        for d in divisors(aa//r):
            a = r*d
            if a>min(U,bb):     # a is too large
                break
            # b goes over divisors of bb, or altermatively it's a solution to b == sqrt(-n_b/n_a)*a (mod p)             # <--------------------- CAN SPEED UP if p is large?????????????
            for b in reversed(divisors(bb)):
                if b<a:         # b is too small
                    break
                #cnt |= {(a,b)}
                z = divmod( n_b*a^2 + n_a*b^2, p )
                if z[1]:
                    continue
                g = is_square(z[0]*q//(a*b)^2,root=True)[1]
                #print(aa,bb,a,b,g)
                if g and a*g>=L and a*g<=U:
                    S.append( (a*g,b*g) )

    #print("Iterations:",cnt)
    return S



#########################################################################################################################


def res_rep(s, N, ODD_ONLY = False, MIN_DENOM = 1, PRIME_BOUND = None):

    if ODD_ONLY and MIN_DENOM%2==0:
        MIN_DENOM += 1

    def succ(t):
        s0, m = t
        if s0==0 or len(m)>=N:
            return []

        p = numerator(s0)
        q = denominator(s0)

        if len(m)==N-1:
            if p==1 and is_square(q):
                r = q.isqrt()
                if r>=m[-1] and (r%2 or not ODD_ONLY) and (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND):
                    return [(0,m+(r,))]
            return []

        L = max(m[-1], (q//p).isqrt()+1)            # 1/x_^2 < p/q   ========>  x_ > isqrt(q/p)  ===> x_ >= isqrt(q/p) + 1               # CHANGE ABOVE
        U = floor((N-len(m))/s0).isqrt()

        if ODD_ONLY:
            if L%2==0:
                L += 1
            if U%2==0:
                U -= 1

        if len(m)==N-2:
            if odd_part(p)%4==3 or odd_part(q)%4==3:
                return []

            if BERTRAM:
                sq = sqpart(q)
                step = q // sq^2
                if ODD_ONLY:
                    step *= 2

                if (U-L)//step < number_of_divisors( sq ):
                    t_gen = filter(lambda t: t[1]%t[0]==0 and all(t[1]%e==0 for e in m), complete2trial(p,q,L,U,odd_only=ODD_ONLY))
                    # return [(0,m+t) for t in complete2trial(p,q,L,U,odd_only=ODD_ONLY) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                else:
                    t_gen = filter(lambda t: t[1]%t[0]==0 and all(t[1]%e==0 for e in m), complete2fact(p,q,L,U))
                    #return [(0,m+t) for t in complete2fact(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                # return [(0,m+t) for t in use_all([complete2fact,complete2trial,complete2pell],p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]

                #return [(0,m+t) for t in complete2pell(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                #return [(0,m+t) for t in complete2fact(p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                #return [(0,m+t) for t in complete2trial(p,q,L,U,odd_only=ODD_ONLY) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]

            else:
                # we cannot run complete2fact unconditionally, since (i) q may be hard to factor (deadblock); or (ii) q has too many divisors
                # but deadblock does not happen here since all divisors of q are small

                #if U-L < fact_threshold:
                #    return [(0,m+t) for t in complete2trial(p,q,L,U)]

                if (U-L)*2 < number_of_divisors( sqpart(q)^2 ):
                    t_gen = complete2trial(p,q,L,U)
                    # return [(0,m+t) for t in complete2trial(p,q,L,U) if all(e%2 for e in t) or not ODD_ONLY]
                else:
                    t_gen = complete2fact(p,q,L,U)
                    # return [(0,m+t) for t in complete2fact(p,q,L,U) if all(e%2 for e in t) or not ODD_ONLY]
                # return [(0,m+t) for t in use_all([complete2fact,complete2trial],p,q,L,U) if all(e%2 for e in t) or not ODD_ONLY]


                #print(U-L,end=' ')
                #return [(0,m+t) for t in use_all([complete2fact,complete2trial],p,q,L,U)]

                #return [(0,m+t) for t in concurrent_run(complete2fact,complete2trial,p,q,L,U)]
                #return [(0,m+t) for t in complete2trial(p,q,L,U)]

            return [(0,m+t) for t in t_gen if (not ODD_ONLY or all(e%2 for e in t)) and (not PRIME_BOUND or max(map(largest_prime_factor,t))<=PRIME_BOUND)]


        # Legendre's three-square theorem
        if len(m)==N-3:
            t = p*q
            a = valuation(t,2)
            if a%2==0 and (t>>a)%8==7:
                return []

        # return ( (s0-1/r^2, m+(r,)) for r in range(L,U+1) if (r%2 or not ODD_ONLY) )

        # Note that L is odd when ODD_ONLY==True
        return ( (s0-1/r^2, m+(r,)) for r in range(L,U+1,1+int(ODD_ONLY)) if (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND) )

    return RecursivelyEnumeratedSet(seeds=[(s-1/r^2,(r,)) for r in range(max(MIN_DENOM,(ceil(1/s)-1).isqrt()+1),floor(N/s).isqrt()+1,1+int(ODD_ONLY)) if (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND)], successors=succ, structure='forest')


def all_rep(s, N, odd_only = False, min_denom = 1):
    def func(t):
        print(t)
        return {t}
    return res_rep(s,N,ODD_ONLY=odd_only,MIN_DENOM=min_denom).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N else set(), set.union, set() )

def count_rep(s, N, odd_only = False, min_denom = 1):
    return res_rep(s,N,ODD_ONLY=odd_only,MIN_DENOM=min_denom).map_reduce(lambda t: int(t[0]==0 and len(t[1])==N))


def all_rep_primes23(N):

    def func(t):
        print(t)
        return {t}

    res = set()
    for q in range(1,N+1):
        res = res_rep(q,N,PRIME_BOUND=3).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N else set(), set.union, res )
    return res


def save_all_rep_primes23(U=20):
    for N in range(1,U+1):
        st = time.time()
        res = sorted( all_rep_primes23(N) )
        print(f'{N}:\t{len(res)}\t{time.time()-st:.1f}s')
        with open(f'primes23_rank_{N:02}.txt','w') as f:
            f.writelines(str(r)+'\n' for r in res)


#####################################################################################################################
################################################################## (possibly distinct) denominators with coefficients


def res_cf_rep(s, cf, ODD_ONLY = False, MIN_DENOM = 1, PRIME_BOUND = None, DISTINCT = False):

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
            p_ = numerator(s1)
            q_ = denominator(s1)
            if p_==1 and is_square(q_):
                r = q_.isqrt()
                if r>=m[-1]+int(DISTINCT) and (r%2 or not ODD_ONLY) and (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND) and (not BERTRAM or all(r%e==0 for e in m)):
                    return [(0,m+(r,))]
            return []

        L = max(m[-1] + int(DISTINCT), (cf[M]*q//p).isqrt()+1)            # M < N-1 ==>   c/x_^2 < p/q   ========>  x_ > isqrt(c*q/p)  ===> x_ >= isqrt(c*q/p) + 1   # CHANGE ABOVE
        U = floor(sum(cf[M:])/s0).isqrt()

        # print([L,U])

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
            p/q = cf[-2]/a^2 + cf[-1]/b^2
            '''

            if not legendre_condition(p,q,cf[-2],cf[-1]):
                return []

            if BERTRAM:
                '''
                We have
                p*y^2 - q*cf[-2]*(y/x)^2 = q*cf[-1]
                '''

                sq = sqpart(q*cf[-1])
                step = q*cf[-1] // sq^2
                if ODD_ONLY:
                    step *= 2

                if (U-L)//step < number_of_divisors( sq ):
                    t_gen = filter(lambda t: t[1]%t[0]==0 and all(t[1]%e==0 for e in m), complete2trial(p,q,L,U,n_a=cf[-2],n_b=cf[-1],odd_only=ODD_ONLY))
                    #return [(0,m+t) for t in complete2trial(p,q,L,U,n_a=cf[-2],n_b=cf[-1],odd_only=ODD_ONLY) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                else:
                    t_gen = filter(lambda t: t[1]%t[0]==0 and all(t[1]%e==0 for e in m), complete2fact(p,q,L,U,n_a=cf[-2],n_b=cf[-1]))
                    #return [(0,m+t) for t in complete2fact(p,q,L,U,n_a=cf[-2],n_b=cf[-1]) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]
                # return [(0,m+t) for t in use_all([complete2fact,complete2trial,complete2pell],p,q,L,U,n_a=cf[-2],n_b=cf[-1]) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]

            else:            # GENERAL (not BERTRAM) case

                # we cannot run complete2fact unconditionally, since (i) q may be hard to factor (deadblock); or (ii) q has too many divisors
                # but deadblock does not happen here since all divisors of q are small

                #if U-L < fact_threshold:
                #    return [(0,m+t) for t in complete2trial(p,q,L,U)]

                if (U-L)*2 < number_of_divisors( sqpart(q*cf[-1])^2 ):
                    t_gen = complete2trial(p,q,L,U,n_a=cf[-2],n_b=cf[-1])
                    #return [(0,m+t) for t in complete2trial(p,q,L,U,n_a=cf[-2],n_b=cf[-1]) if all(e%2 for e in t) or not ODD_ONLY]
                else:
                    t_gen = complete2fact(p,q,L,U,n_a=cf[-2],n_b=cf[-1])
                    #return [(0,m+t) for t in complete2fact(p,q,L,U,n_a=cf[-2],n_b=cf[-1]) if all(e%2 for e in t) or not ODD_ONLY]
                # return [(0,m+t) for t in use_all([complete2fact,complete2trial],p,q,L,U) if all(e%2 for e in t) or not ODD_ONLY]


                #print(U-L,end=' ')
                #return [(0,m+t) for t in use_all([complete2fact,complete2trial],p,q,L,U)]

                #return [(0,m+t) for t in concurrent_run(complete2fact,complete2trial,p,q,L,U)]
                #return [(0,m+t) for t in complete2trial(p,q,L,U)]

            return [(0,m+t) for t in t_gen if t[0]+int(DISTINCT)<=t[1] and (not ODD_ONLY or all(e%2 for e in t)) and (not PRIME_BOUND or max(map(largest_prime_factor,t))<=PRIME_BOUND)]


        # Case M <= N-3:
        # Legendre's three-square theorem
        if M==N-3 and all(map(is_square,cf[-3:])):
            t = p*q
            a = valuation(t,2)
            if a%2==0 and (t>>a)%8==7:
                return []

        return ( (s0-cf[M]/r^2, m+(r,)) for r in range(L,U+1,1+int(ODD_ONLY)) if (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND) )

    # we have s >= cf[0]/r^2  ==>  r >= ceil( sqrt(s/cf[0]) ) = ceil( sqrt(ceil(s/cf[0])) ) = floor( sqrt(ceil(s/cf[0])-1) ) + 1
    MIN_DENOM = max(MIN_DENOM,(ceil(cf[0]/s)-1).isqrt()+1)
    if ODD_ONLY and MIN_DENOM%2==0:
        MIN_DENOM += 1

    return RecursivelyEnumeratedSet(seeds=[(s-cf[0]/r^2,(r,)) for r in range(MIN_DENOM,floor(sum(cf)/s).isqrt()+1,1+int(ODD_ONLY)) if (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND)], successors=succ, structure='forest')


def all_cf_rep(s, cf, odd_only = False, min_denom = 1, distinct = False):
    N = len(cf)
    def func(t):
        print(t)
        return {t}
    return res_cf_rep(s,cf,ODD_ONLY=odd_only,MIN_DENOM=min_denom,DISTINCT=distinct).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N else set(), set.union, set() )


def count_cf_rep(s, N, odd_only = False, min_denom = 1, distinct = False):
    N = len(cf)
    return res_cf_rep(s,cf,ODD_ONLY=odd_only,MIN_DENOM=min_denom,DISTINCT=distinct).map_reduce(lambda t: int(t[0]==0 and len(t[1])==N))


################################################################ MNSD (maximally non self-dual) condition

'''
MNSD (maximally non self-dual) constraint implies that we only need to consider the Egyptian fractions with odd squared denominators
1 = 1/x_1^2 + ... + 1/x_n^2
such that x_1 = x_2  <=  x_3 = x_4  <=  x_5 = x_6  <= ....
'''
def res_rep_MNSD(s, N, MIN_DENOM = 1):
    assert BERTRAM
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
        return {t}
    return res_rep_MNSD(s,N_,MIN_DENOM=min_denom).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N_ else set(), set.union, set())


########################################################## Dynamically take into account Theorem 8.5

# represent s as sum of N fractions with squared denominators. Let mult_i be multiplicities of denominators.
# Theorem 8.5 implies that if a prime p divides some denominator, then max( mult_i ) >= (p-1)/2
# L_MAX = lower bound for the maximum multiplicity
# PRIME_BOUND = 2*N+1 is not needed here, but it can be set up smaller if needed
# PRIME_LB = lower bound for max prime
def res_rep_th85(s, N, ODD_ONLY = False, MIN_DENOM = 1, PRIME_BOUND = None, L_MAX = 1, PRIME_LB = 0):

    if ODD_ONLY and MIN_DENOM%2==0:
        MIN_DENOM += 1

    # debug_m = (2,)*4 + (3,)*9

    def succ(t):
        s0, m, l_max = t     # s0 is the remainder to represent, m[] are denominators, mult[] are multiplicities, l_max is lower bound for max(mult[])

        '''
        if m == debug_m[:len(m)]:
            print(s0,m,l_max)
        '''

        SUCC = []

        if s0==0 or len(m)>=N:
            return SUCC

        mult = get_mult(m)

        # check if Theorem is not yet fulfilled
        if max(mult) < l_max:
            '''
            We have two non-exclusive cases:
            Case I: multiplicity of m[-1] will be >= l_max
            Case II: multiplicity of some term > m[-1] will be >= l_max

            We do not enforce case II unless C_ = l_max > 1 and len(m) >= N_ - 2. 
            In the latter case, we need to explicitly allow Case I alternative.

            If Case I is the only alternative, it was already addressed when m[-1] was added (see code below).

            '''

            # try to fulfil with new element
            N_ = N - (l_max-1)          # think of gluing a run of length l_max into a single element
            C_ = l_max

            if C_ > 1 and len(m) >= N_ - 2:
                # try to fulfil with appending the last element
                if ((ss := (mm:=l_max-mult[-1])/m[-1]^2) < s0 and len(m)+mm<N) or (ss==s0 and len(m)+mm==N and (not BERTRAM or all(m[-1]%e==0 for e in m))):
                    SUCC.append( (s0-mm/m[-1]^2,m+(m[-1],)*mm,l_max) )

        else:
            N_ = N
            C_ = 1

        #if m == (2, 2, 2, 3, 3):
        #    print('='*20)

        if len(m) >= N_:
            return SUCC

        p = numerator(s0)
        q = denominator(s0)

        if len(m) == N_ - 1:
            # we want p/q = C_ / r^2
            if C_%p or not is_square(r:=q*C_//p):
                return SUCC
            r = r.isqrt()
            if r>=m[-1]:                        ############# or (r==m[-1] and len(SUCC)==0):          # avoid duplication - SHOULD we worry?
                if (pp:=largest_prime_factor(r)) >= 3:
                    l_max = max(l_max, (pp-1)//2)
                if (r%2 or not ODD_ONLY) and (not PRIME_BOUND or pp<=PRIME_BOUND) and max(mult)>=l_max and (not BERTRAM or all(r%e==0 for e in m)):
                    SUCC.append( (0,m+(r,)*C_,l_max) )
            return SUCC

        L = max(m[-1], (q//p).isqrt()+1)            # 1/x_^2 < p/q   ========>  x_ > isqrt(q/p)  ===> x_ >= isqrt(q/p) + 1               # CHANGE ABOVE
        U = floor((N-len(m))/s0).isqrt()

        if ODD_ONLY:
            if L%2==0:
                L += 1
            if U%2==0:
                U -= 1
        if U<L:
            return []

        '''
        if m == debug_m[:len(m)] and len(m) == 11:
            print([L,U])
            print(N_,C_)
        '''

        if len(m) == N_ - 2:

            if BERTRAM:
                '''
                We have
                p*y^2 - q*(y/x)^2 = q*C_    OR       p*y^2 - q*C_*(y/x)^2 = q
                '''

                sq = sqpart(q*C_)
                step = q*C_ // sq^2
                if ODD_ONLY:
                    step *= 2

                if (U-L)//step < number_of_divisors( sq ):
                    t_gen = ((t[0],)*C_ + (t[1],) for t in complete2trial(p,q,L,U,odd_only=ODD_ONLY,n_a=C_,n_b=1) if t[1]%t[0]==0 and all(t[1]%e==0 for e in m))
                    if C_!=1:
                        t_gen = itertools.chain( t_gen, ((t[0],) + (t[1],)*C_ for t in complete2trial(p,q,L,U,odd_only=ODD_ONLY,n_a=1,n_b=C_) if t[1]%t[0]==0 and all(t[1]%e==0 for e in m)) )
                else:
                    t_gen = ((t[0],)*C_ + (t[1],) for t in complete2fact(p,q,L,U,n_a=C_,n_b=1) if t[1]%t[0]==0 and all(t[1]%e==0 for e in m))
                    if C_!=1:
                        t_gen = itertools.chain( t_gen, ((t[0],) + (t[1],)*C_ for t in complete2fact(p,q,L,U,n_a=1,n_b=C_) if t[1]%t[0]==0 and all(t[1]%e==0 for e in m)) )
                # return [(0,m+t) for t in use_all([complete2fact,complete2trial,complete2pell],p,q,L,U) if (not ODD_ONLY or all(e%2 for e in t)) and t[1]%t[0]==0 and all(t[1]%e==0 for e in m)]




            else:
                # we cannot run complete2fact unconditionally, since (i) q may be hard to factor (deadblock); or (ii) q has too many divisors
                # but deadblock does not happen here since all divisors of q are small

                #if U-L < fact_threshold:
                #    return [(0,m+t) for t in complete2trial(p,q,L,U)]

                if (U-L)*2 < number_of_divisors( sqpart(q)^2 ):
                    t_gen = ((t[0],)*C_ + (t[1],) for t in complete2trial(p,q,L,U,odd_only=ODD_ONLY,n_a=C_,n_b=1))
                    if C_!=1:
                        t_gen = itertools.chain( t_gen, ((t[0],) + (t[1],)*C_ for t in complete2trial(p,q,L,U,odd_only=ODD_ONLY,n_a=1,n_b=C_)) )
                else:
                    t_gen =  ((t[0],)*C_ + (t[1],) for t in complete2fact(p,q,L,U,n_a=C_,n_b=1))
                    if C_!=1:
                        t_gen = itertools.chain( t_gen, ((t[0],) + (t[1],)*C_ for t in complete2fact(p,q,L,U,n_a=1,n_b=C_)) )

                # return [(0,m+t) for t in use_all([complete2fact,complete2trial],p,q,L,U) if all(e%2 for e in t) or not ODD_ONLY]
                #return [(0,m+t) for t in concurrent_run(complete2fact,complete2trial,p,q,L,U)]

            for t in t_gen:
                if ODD_ONLY and any(e%2==0 for e in t):
                    continue
                t_p = max(map(largest_prime_factor,t))
                if (PRIME_BOUND and t_p>PRIME_BOUND):
                    continue
                l_p = max(l_max,(t_p//2 if t_p>=3 else 0))
                if max(get_mult(mt := m+t)) >= l_p:
                    SUCC.append( (0,mt,l_p) )

            return SUCC

        # Legendre's three-square theorem       <---  important speed-up
        if len(m) == N_ - 3 and C_.is_square():
            t = p*q
            a = valuation(t,2)
            if a%2==0 and (t>>a)%8==7:
                return SUCC

        '''
        if m == (2, 2, 2, 3, 3):
            print([L,U])
        '''

        if PRIME_LB and len(m) == N - 3:
            qq = largest_prime_factor(q)
        else:
            qq = PRIME_LB

        for r in range(L,U+1,1+int(ODD_ONLY)):
            p_r = largest_prime_factor(r)
            if PRIME_BOUND and p_r>PRIME_BOUND:
                continue
            l_r = max(l_max, (p_r//2 if p_r>=3 else 0))

            if qq < PRIME_LB and p_r < PRIME_LB:    # we want either qq or p_r >= PRIME_LB
                continue

            mult_r = mult[-1] if r==m[-1] else 0      # multiplicity of r in m

            if l_r > max(mult) and len(m) + 1 + l_r > N:
                # the only chance is to fulfil max(mult) >= l_r is with r
                if (len(m) + (new_mult := l_r - mult_r) < N and new_mult/r^2 < s0) or (len(m) + new_mult == N and new_mult/r^2 == s0 and (not BERTRAM or all(r%e==0 for e in m))):
                    SUCC.append( (s0-new_mult/r^2, m+(r,)*new_mult, l_r) )
                continue
            SUCC.append( (s0-1/r^2, m+(r,), l_r) )

        '''
        if m == (2, 2, 2, 3, 3):
            print( SUCC )
        '''

        return SUCC

    return RecursivelyEnumeratedSet(seeds=[ ( s-1/r^2, (r,), max(L_MAX,(1 if (p:=largest_prime_factor(r))<=2 else (p-1)//2)) ) for r in range(max(MIN_DENOM,(ceil(1/s)-1).isqrt()+1),floor(N/s).isqrt()+1,1+int(ODD_ONLY)) if (not PRIME_BOUND or largest_prime_factor(r)<=PRIME_BOUND)], successors=succ, structure='forest')



def all_rep_th87(N,l_max=1,prime_lb=0):

    def func(t):
        if test_th87(t) and (prime_lb==0 or largest_prime_factor(t[-1])>=prime_lb):
            print(t)
            return {t}
        else:
            return set()

    res = set()
    for q in range(1,N+1):
        res = res_rep_th85(q,N,L_MAX=l_max,PRIME_LB=prime_lb).map_reduce(lambda t: func(t[1]) if t[0]==0 and len(t[1])==N else set(), set.union, res )
    return res


# get representations with a prime > N
# search for counterexamples to https://mathoverflow.net/q/466864
def all_rep_above_r(N):
    return sorted(all_rep_th87(N,l_max=(next_prime(N)-1)//2,prime_lb=N+1))

# get representations with a prime > N/2
def all_rep_above_half_r(N):
    return sorted(all_rep_th87(N,l_max=(next_prime(N//2)-1)//2,prime_lb=N//2+1))

