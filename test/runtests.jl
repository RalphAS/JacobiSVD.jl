using JacobiSVD
using LinearAlgebra
using Test, Random

Random.seed!(1101)

#=
Ground rules:
These are mostly checks of sanity and API consistency;
correctness is up to LAPACK implementors.
=#

# construct a general matrix with specified singular values
function mkmat(m,n,s,T)
    @assert length(s) == n
    @assert n <= m
    if T <: Complex
        tmp1 = randn(m,n) + randn(m,n)*1im
        tmp2 = randn(n,n) + randn(n,n)*1im
    else
        tmp1 = randn(m,n)
        tmp2 = randn(n,n)
    end
    u,r = qr(tmp1)
    v,r = qr(tmp2)
    Atmp = u * diagm(0=>s) * v'
    A = T.(Atmp)
end

# TODO: 
@testset "jsvd interface $T" for T in (Float64, ComplexF64)
    tol = 10.0
    e = eps(real(T))
    m,n = 10,5
    s = rand(n) .+ 1.0
    A = mkmat(m,n,s,T)
    S = jsvd!(copy(A))
    @test norm(S.U * Diagonal(S.S) * S.Vt - A) / norm(A) < tol * m * e
    At = Matrix(A')
    S = jsvd!(copy(At))
    @test norm(S.U * Diagonal(S.S) * S.Vt - At) / norm(A) < tol * m * e
end

@testset "Squat $T" for T in (Float64, ComplexF64)
    m,n = 50,100
    A = rand(T,m,n)
    @test_throws ArgumentError gejsv!('G','U','V',copy(A))
    @test_throws ArgumentError gesvj!('G','U','V',copy(A))
end

@testset "Tall easy $T" for T in (Float64, ComplexF64)
    m,n = 100,50
    s = rand(n) .+ 1.0
    A = mkmat(m,n,s,T)
    U, S, V, scales, sconda = gejsv!('G','U','V',copy(A))
    tol = 10.0
    Ss = (scales[2] / scales[1]) * S
    e = eps(real(T))
    @test norm(U * Diagonal(Ss) * V' - A) / norm(A) < tol * m * e
    @test norm(U' * U - I) < tol * m * e
    @test norm(V' * V - I) < tol * m * e

    U, S, V, scale = gesvj!('G','U','V',copy(A))
    @test norm(U * Diagonal(scale * S) * V' - A) / norm(A) < tol * m * e
    @test norm(U' * U - I) < tol * m * e
    @test norm(V' * V - I) < tol * m * e
end

@testset "Tall serious $T" for T in (Float32, ComplexF32)
    m,n = 100,50
    r = -1.5 * log2(eps(real(T)))
    s = rand(n) .+ 1.0
    # make a challenging mtx with guaranteed:
    c = 2.0 .^ collect(range(0.0,stop=r,length=n))
    A = mkmat(m,n,s,T) * Diagonal(shuffle(T.(c)))
    Tw = widen(T)
    # could use
    #    Uw, Sw, Vw = svd(Tw.(A))
    # but we demonstrated basic sanity above
    Uw, Sw, Vw, scalew = gesvj!('G','U','V',Tw.(A))
    Sw .= scalew .* Sw
    U, S, V, scales, sconda = gejsv!('G','U','V',copy(A))
    tol = 10.0
    Ss = (scales[2] / scales[1]) * S
    e = eps(real(T))
    @test norm(U * Diagonal(Ss) * V' - A) / norm(A) < tol * m * e
    @test norm(U' * U - I) < tol * m * e
    @test norm(V' * V - I) < tol * m * e
    Sss = sort(Ss, rev=true)
    Twr = real(Tw)
    # the factor of m is a WAG here
    @test all((Twr.(Sss) - Sw) ./ Sw .< tol * m * e)

    U, S, V, scale = gesvj!('G','U','V',copy(A))
    @test norm(U * Diagonal(scale*S) * V' - A) / norm(A) < tol * m * e
    @test norm(U' * U - I) < tol * m * e
    @test norm(V' * V - I) < tol * m * e
    Sss = sort(S, rev=true)
    # the factor of m is a WAG here
    @test all((Twr.(Sss) - Sw) ./ Sw .< tol * m * e)
end

# Attempt to avoid surprises with unusual job params.
# This is intended to make sure we allocate correctly sized arrays etc.
@testset "SVJ jobs $T" for T in (Float32, ComplexF32)
    m,n = 10,5
    s = rand(n) .+ 1.0
    A = mkmat(m,n,s,T)
    sref = svdvals(A)
    for (joba, jobu, jobv) in
        (('G','U','V'),
         ('G','C','V'),
         ('G','N','N'),
         ('G','U','N'),
         ('G','N','V')
         )
        U, S, V, scale = gesvj!(joba,jobu,jobv,copy(A),ctol=m)
        Ss = sort(scale * S, rev=true)
        @test Ss ≈ sref
    end
    V0 = rand(T,2,n)
    for (joba, jobu, jobv) in
        (('G','U','A'),
         ('G','C','A'),
         ('G','N','A')
         )
        U, S, V, scale = gesvj!(joba,jobu,jobv,copy(A),Vinit=V0,ctol=m)
        Ss = sort(scale * S, rev=true)
        @test Ss ≈ sref
    end
    # TODO: triangular A (joba ∈ L,U)
end

@testset "JSV jobs $T" for T in (Float32, ComplexF32)
    m,n = 10,5
    s = rand(n) .+ 1.0
    A = mkmat(m,n,s,T)
    sref = svdvals(A)
    for (joba, jobu, jobv) in
        (('F','U','V'),
         ('C','U','V'),
         ('E','U','V'),
         ('G','U','V'),
         ('A','U','V'),
         ('R','U','V'),
         ('F','U','N'),
         ('F','F','V'),
         ('F','N','V'),
         )
        U, S, V, scales, sconda = gejsv!(joba,jobu,jobv,copy(A))
        Ss = sort((scales[2]/scales[1]) * S, rev=true)
        @test Ss ≈ sref
        if jobu == 'F'
            @test size(U) == (m,m)
        elseif jobu == 'U'
            @test size(U) == (m,n)
        end
        if joba ∈ ('E','G')
            @test sconda >= 0
        else
            @test sconda == -1
        end
    end
end
@testset "JSV perturb jobs $T" for T in (Float32, ComplexF32)
    m,n = 10,5
    s = rand(n) .+ 1.0
    A = mkmat(m,n,s,T)
    sref = svdvals(A)
    for (joba, jobu, jobv) in
        (('F','U','V'),
         ('F','N','N'),
         )
        # perturb to avoid denorms
        U, S, V, scales, sconda = gejsv!(joba,jobu,jobv,copy(A),jobp='P')
        Ss = sort((scales[2]/scales[1]) * S, rev=true)
        @test Ss ≈ sref
    end
end

@testset "JSV unrestricted jobs $T" for T in (Float32, ComplexF32)
    m,n = 10,5
    s = rand(n) .+ 1.0
    A = mkmat(m,n,s,T)
    sref = svdvals(A)
    for (joba, jobu, jobv) in
        (('F','U','V'),
         ('F','N','N'),
         )
        # don't restrict range
        U, S, V, scales, sconda = gejsv!(joba,jobu,jobv,copy(A),jobr='N')
        Ss = sort((scales[2]/scales[1]) * S, rev=true)
        @test Ss ≈ sref
    end
end

@testset "JSV transposable jobs $T" for T in (Float32, ComplexF32)
    m,n = 10,5
    # transpose obvs requires square A
    s = rand(m) .+ 1.0
    A = mkmat(m,m,s,T)
    sref = svdvals(A)
    for (joba, jobu, jobv) in
        (('F','U','V'),
         ('F','N','N'),
         ('F','W','V'),
         ('F','U','W')
         )
        U, S, V, scales, sconda = gejsv!(joba,jobu,jobv,copy(A),jobt='T')
        Ss = sort((scales[2]/scales[1]) * S, rev=true)
        @test Ss ≈ sref
    end
end

