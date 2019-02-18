
using LinearAlgebra
using LinearAlgebra: chkstride1, BlasInt
using LinearAlgebra.LAPACK: liblapack, chklapackerror
using LinearAlgebra.BLAS: @blasfunc
using Base: has_offset_axes

for (gesvj, elty, relty) in
    ((:dgesvj_, :Float64, :Float64),
     (:sgesvj_, :Float32, :Float32),
     (:zgesvj_, :ComplexF64, :Float64),
     (:cgesvj_, :ComplexF32, :Float32))
    @eval begin
#       SUBROUTINE ZGESVJ( JOBA, JOBU, JOBV, M, N, A, LDA, SVA, MV, V,
#                          LDV, CWORK, LWORK, RWORK, LRWORK, INFO )
#
#       .. Scalar Arguments ..
#       INTEGER            INFO, LDA, LDV, LWORK, LRWORK, M, MV, N
#       CHARACTER*1        JOBA, JOBU, JOBV
#       ..
#       .. Array Arguments ..
#       COMPLEX*16         A( LDA, * ),  V( LDV, * ), CWORK( LWORK )
#       DOUBLE PRECISION   RWORK( LRWORK ),  SVA( N )
        function gesvj!(joba::AbstractChar, jobu::AbstractChar,
                        jobv::AbstractChar, A::AbstractMatrix{$elty};
                        Vinit = similar(A, $elty, (0, size(A,2))),
                        ctol=zero($relty))
            @assert !has_offset_axes(A)
            @assert !has_offset_axes(Vinit)
            chkstride1(A)
            m, n   = size(A)
            if m < n
                throw(ArgumentError("matrix A must be tall or square"))
            end
            minmn  = min(m, n)
            S      = similar(A, $relty, minmn)
            if (jobv != 'N') && (size(Vinit,2) != n)
                throw(DimensionMismatch("matrix Vinit must have n columns"))
            end
            mv = 0
            if jobv ∈ ('A','N')
                mv = size(Vinit,1)
                V = Vinit
            elseif jobv ∈ ('V','J')
                V = similar(A, $elty, (n,n))
            end
            cmplx  = eltype(A) <: Complex
            if cmplx
                lcwork = -1
                cwork = Vector{$elty}(undef, 1)
                lwork = -1
                work   = Vector{$relty}(undef, 1)
            else
                lwork  = max(6,m+n)
                work   = Vector{$relty}(undef, lwork)
            end
            if jobu == 'C'
                if (ctol < 1)
                    throw(ArgumentError("appropriate ctol must be provided for jobu = 'C'"))
                end
                work[1] = ctol
            end
            info = Ref{BlasInt}()
            if cmplx
                for i = 1:2 # first call returns workspace requirements
                    ccall((@blasfunc($gesvj), liblapack), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, # jobA/U/V
                           Ref{BlasInt}, Ref{BlasInt}, # m,n
                           Ptr{$elty}, Ref{BlasInt}, # A,lda
                           Ptr{$relty}, Ref{BlasInt}, # SVA, mv
                           Ptr{$elty}, Ref{BlasInt}, # V, ldv
                           Ptr{$elty}, Ref{BlasInt}, # cwork, lcwork
                           Ptr{$relty}, Ref{BlasInt}, # rwork, lwork
                           Ptr{BlasInt}), # info
                          joba, jobu, jobv, m, n, A, max(1,stride(A,2)), S, mv, V, max(1,stride(V,2)),
                          cwork, lcwork, work, lwork, info)
                    chklapackerror(info[])
                    if (i == 1)
                        lwork = BlasInt(work[1])
                        resize!(work, lwork)
                        lcwork = BlasInt(real(cwork[1]))
                        resize!(cwork, lcwork)
                    end
                end
            else
                ccall((@blasfunc($gesvj), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, # jobA/U/V
                       Ref{BlasInt}, Ref{BlasInt}, # m,n
                       Ptr{$elty}, Ref{BlasInt}, # A,lda
                       Ptr{$relty}, Ref{BlasInt}, # SVA, mv
                       Ptr{$elty}, Ref{BlasInt}, # V, ldv
                       Ptr{$elty}, Ref{BlasInt}, # work, lwork
                       Ptr{BlasInt}), # info
                      joba, jobu, jobv, m, n, A, max(1,stride(A,2)), S, mv, V, max(1,stride(V,2)),
                      work, lwork, info)
                chklapackerror(info[])
            end

            scale = work[1]
            ranka = Int(work[2])
            extra = work[3:6]

            if jobu ∈ ('U','F','C')
                U = A[:,1:ranka]
            else
                U = similar(A, (m,0))
            end
            if jobv ∉ ('V','J','A')
                    V = similar(A, (n,0))
            end
            return U, S, V, scale, ranka, extra
        end # function
    end # eval
end # loop over types

"""
    gesvj!(joba, jobu, jobv, A; Vinit = similar(A, (0,n))) -> (U, S, V, scale, ranka, extras)

Compute the singular value decomposition of `A`, `A = U * (scale * Diagonal(S)) * V'`.

Uses the single-sided Jacobi scheme from LAPACK.

`ranka` is the number of significant singular values computed.

`extras` is `rwork[3:6]` returned from the LAPACK routine.

See the LAPACK documentation (`dgesvj` etc.) for details.
"""
function gesvj! end


for (gejsv, elty, relty) in
    ((:dgejsv_, :Float64, :Float64),
     (:sgejsv_, :Float32, :Float32),
     (:zgejsv_, :ComplexF64, :Float64),
     (:cgejsv_, :ComplexF32, :Float32))
    @eval begin
#     SUBROUTINE ZGEJSV( JOBA, JOBU, JOBV, JOBR, JOBT, JOBP,
#                         M, N, A, LDA, SVA, U, LDU, V, LDV,
#                         CWORK, LWORK, RWORK, LRWORK, IWORK, INFO )
#
#     .. Scalar Arguments ..
#     IMPLICIT    NONE
#     INTEGER     INFO, LDA, LDU, LDV, LWORK, M, N
#     ..
#     .. Array Arguments ..
#     COMPLEX*16     A( LDA, * ),  U( LDU, * ), V( LDV, * ), CWORK( LWORK )
#     DOUBLE PRECISION   SVA( N ), RWORK( LRWORK )
#     INTEGER     IWORK( * )
#     CHARACTER*1 JOBA, JOBP, JOBR, JOBT, JOBU, JOBV

        function gejsv!(joba::AbstractChar, jobu::AbstractChar,
                        jobv::AbstractChar, A::AbstractMatrix{$elty};
                        jobr::AbstractChar = 'R',
                        jobt::AbstractChar = 'N', jobp::AbstractChar = 'N')
            @assert !has_offset_axes(A)
            chkstride1(A)
            m, n   = size(A)
            if m < n
                throw(ArgumentError("matrix A must be tall or square"))
            end
            minmn  = min(m, n)
            S      = similar(A, $relty, minmn)
            if jobv == 'N'
                V = similar(A, $elty, (n,0))
            else
                V = similar(A, $elty, (n,n))
            end
            if jobu == 'U'
                U = similar(A, $elty, (m,n))
            elseif jobu == 'F'
                U = similar(A, $elty, (m,m))
            elseif jobu == 'N'
                U = similar(A, $elty, (m,0))
            else
                # use as workspace
                U = similar(A, $elty, (m,n))
            end
            cmplx  = eltype(A) <: Complex
            if cmplx
                lcwork = BlasInt(-1)
                cwork = Vector{$elty}(undef, 7)
                liwork = 1
                lwork  = BlasInt(-1)
            else
                liwork = m+3*n;
                lwork = max(n+m*32,2*m+n,6*n+2*n*n)
            end
            work   = Vector{$relty}(undef, max(7,lwork))
            iwork = Vector{BlasInt}(undef, liwork)
            info   = Ref{BlasInt}()
            if cmplx
                for i = 1:2 # first call returns workspace requirements
                    ccall((@blasfunc($gejsv), liblapack), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, # jobA/U/V
                           Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, # jobR/T/P
                           Ref{BlasInt}, Ref{BlasInt}, # m,n
                           Ptr{$elty}, Ref{BlasInt}, # A,lda
                           Ptr{$relty}, # SVA,
                           Ptr{$elty}, Ref{BlasInt}, # U, ldu
                           Ptr{$elty}, Ref{BlasInt}, # V, ldv
                           Ptr{$elty}, Ref{BlasInt}, # cwork, lwork
                           Ptr{$relty}, Ref{BlasInt}, # rwork, lrwork
                           Ptr{BlasInt}, # iwork
                           Ptr{BlasInt}), # info
                          joba, jobu, jobv, jobr, jobt, jobp, m, n,
                          A, max(1,stride(A,2)), S,
                          U, max(1,stride(U,2)), V, max(1,stride(V,2)),
                          cwork, lcwork, work, lwork, iwork, info)
                    chklapackerror(info[])
                    if (i == 1)
                        lwork = BlasInt(work[1])
                        resize!(work, lwork)
                        lcwork = BlasInt(real(cwork[1]))
                        resize!(cwork, lcwork)
                        resize!(iwork, iwork[1])
                    end
                end
            else
                ccall((@blasfunc($gejsv), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, # jobA/U/V
                       Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, # jobR/T/P
                       Ref{BlasInt}, Ref{BlasInt}, # m,n
                       Ptr{$elty}, Ref{BlasInt}, # A,lda
                       Ptr{$relty},  # SVA,
                       Ptr{$elty}, Ref{BlasInt}, # U, ldu
                       Ptr{$elty}, Ref{BlasInt}, # V, ldv
                       Ptr{$elty}, Ref{BlasInt}, # work, lwork
                       Ptr{BlasInt}, # iwork
                       Ptr{BlasInt}), # info
                      joba, jobu, jobv, jobr, jobt, jobp, m, n,
                      A, max(1,stride(A,2)), S,
                      U, max(1,stride(U,2)), V, max(1,stride(V,2)),
                      work, lwork, iwork, info)
                chklapackerror(info[])
            end
            if iwork[3] != 0
                @warn "loss of accuracy: some column norms were denormals"
            end
            scales = (work[1], work[2])
            if joba ∈ ('E','G')
                sconda = work[3]
            else
                # absurd value as a passive-aggressive warning
                sconda = -one($relty)
            end
            extra = work[4:7]
            return U,S,V,scales,sconda,extra
        end # function
    end # eval
end # loop over types

"""
    gejsv!(joba, jobu, jobv, A; jobr='R', jobt='N', jobp='N') -> (U, S, V, scales, sconda, extras)

Finds the singular value decomposition of `A`,
`A = U * (σ * Diagonal(S)) * V'`, using the
preconditioned Jacobi algorithm implemented in LAPACK.

`scales` is a 2-tuple specifying a scale factor for `S`
to avoid over/underflow: `σ=scales[2]/scales[1]` in the above formula.
(`rwork[1:2]` in the LAPACK result).

`sconda` is an estimate of equilibrated condition number (`rwork[3]` in the LAPACK result) or -1 if not requested via `joba`.

`extras` are `rwork[4:7]` from the LAPACK result.

Briefly, `jobr='N'` allows possibly unreliably extreme singular values;
`jobt='T'` allows transposition of `A` (if square), to speed convergence;
`jobp='P'` attempts to quash denormals for efficiency.

See the LAPACK documentation (`dgejsv` etc.) for details.
"""
function gejsv! end
