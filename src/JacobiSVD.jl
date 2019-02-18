module JacobiSVD
using LinearAlgebra

export gesvj!, gejsv!, jsvd!

include("lapack.jl")

"""
    jsvd!(A; full::Bool = false, precondition::Bool = true)

Compute the singular value decomposition (SVD) of `A` and return an `SVD` object.

Uses a one-sided Jacobi SVD algorithm, or (normally) a preconditioned variant.
These schemes often produce more accurate small singular values than the
standard algorithms. Preconditioning often improves performance.
Usage is similar to [`LinearAlgebra.svd!`](@ref).
"""
function jsvd!(A::StridedMatrix{T}; full::Bool = false,
               precondition::Bool = true,
               ) where T<:LinearAlgebra.BlasFloat
    m,n = size(A)
    squat = m < n
    if precondition
        if squat
            A = Matrix(A')
            xfull = true
        else
            xfull = full
        end
        U, S, V, scales = gejsv!('F', xfull ? 'F' : 'U', 'V', A)
        S .= (scales[2]/scales[1]) .* S
        if squat
            U, V = V, U
            if !full
                V = V[1:n,1:m]
            end
        else
            if !full
                U = U[1:m,1:n]
            end
        end
    else
        if full
            throw(ArgumentError("full U is not available from gesvj!"))
        end
        U, S, V, scale, ranka = gesvj!('G', 'U', 'V', A)
        if ranka < n
            # CHECKME: are trailing entries in S zeroed out?
            @warn "matrix is rank deficient; only $ranka left vectors are valid"
        end
        S .= scale .* S
    end
    # CHECKME: should we materialize the transpose?
    LinearAlgebra.SVD(U,S,V')
end

end # module
