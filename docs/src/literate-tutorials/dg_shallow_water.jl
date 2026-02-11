# # [Shallow Water (hyperbolic) with DG + RK time stepping](@id tutorial-swe-dg)
#
# ![](sw_dg_ferrite.gif)
#-
# ## Introduction
#
# This tutorial shows how to solve a **hyperbolic conservation law** with a
# **Discontinuous Galerkin (DG)** spatial discretization and an **explicit Runge–Kutta**
# time integrator using Ferrite.
#
# We use the 2D **shallow water equations (SWE)** in conservative form:
#
# ```math
# \partial_t U + \nabla\cdot F(U) = S(U, x),\qquad
# U = \begin{pmatrix} h \\ q_x \\ q_y \end{pmatrix},
# ```
#
# with fluxes
#
# ```math
# F(U) = \begin{pmatrix}F_x & F_y\end{pmatrix}, \qquad
# F_x(U)=\begin{pmatrix} q_x \\ \frac{q_x^2}{h} + g\frac{h^2}{2} \\ \frac{q_x q_y}{h}\end{pmatrix},
# \qquad
# F_y(U)=\begin{pmatrix} q_y \\ \frac{q_x q_y}{h} \\ \frac{q_y^2}{h} + g\frac{h^2}{2}\end{pmatrix},
# ```
#
# and a bathymetry source term
#
# ```math
# S(U,x) = \begin{pmatrix} 0 \\ -g h \partial_x b \\ -g h \partial_y b \end{pmatrix}.
# ```
#
#-
# ## The Discontinuous Galerkin method for hyperbolic PDEs
#
# We briefly recall the Discontinuous Galerkin (DG) method for hyperbolic PDEs, i.e. systems of the form
#
# ```math
# \partial_t U + \nabla \cdot F(U) = S(U) \quad \text{in } \Omega,
# ```
#
# where $\Omega \subset \mathbb{R}^2$ is the spatial domain (here a square), $U:\Omega\to\mathbb{R}^m$
# is the vector of conserved variables, $F(U)$ is the flux tensor and $S(U)$ a source term.
#
# ### Mesh and discrete space
#
# Let $\mathcal{M}_\Omega$ be a mesh (a partition of $\Omega$) into elements $E$.
# For a given polynomial degree $p$, the DG approximation space is the broken space
#
# ```math
# \mathbb{W}_p(\mathcal{M}_\Omega)
# = \{ V \;|\; V|_E \in \mathbb{P}_p(E) \;\; \forall E\in\mathcal{M}_\Omega \},
# ```
#
# where $\mathbb{P}_p(E)$ denotes polynomials of degree $\le p$ on the element $E$.
# Functions in $\mathbb{W}_p(\mathcal{M}_\Omega)$ are **not required to be continuous across elements**.
#
# We denote by $\mathcal{F}_\Omega^{\mathrm{int}}$ the set of **interior facets** (interfaces) of the mesh.
# Each interior facet $F \in \mathcal{F}_\Omega^{\mathrm{int}}$ is shared by two neighboring elements,
# which we denote $E^-$ and $E^+$, and we choose a unit normal $n$ pointing from $E^-$ to $E^+$.
# We also denote the boundary facets by $\mathcal{F}_\Omega^{\partial}$.
#
# For any (possibly discontinuous) field $W$, we define the **traces** on an interface:
# ```math
# W^- = W|_{E^-}, \qquad W^+ = W|_{E^+} \quad \text{on } F,
# ```
# and similarly on boundary facets only the interior trace $W^-$ is available.
#
# ### Element-wise weak formulation
#
# The global weak form reads: find $U$ such that for all test functions $V$,
#
# ```math
# \int_\Omega \partial_t U \, V \, d\Omega
# + \int_\Omega (\nabla\cdot F(U))\, V \, d\Omega
# = \int_\Omega S(U)\, V \, d\Omega.
# ```
#
# In DG we restrict both trial and test functions to $\mathbb{W}_p(\mathcal{M}_\Omega)$ and write
# the formulation element-by-element:
#
# ```math
# \sum_{E\in\mathcal{M}_\Omega}
# \left(
# \int_E \partial_t U \, V \, dE
# + \int_E (\nabla\cdot F(U))\, V \, dE
# \right)
# =
# \sum_{E\in\mathcal{M}_\Omega}
# \int_E S(U)\, V \, dE.
# ```
#
# We now integrate the flux-divergence term by parts on each element:
#
# ```math
# \int_E (\nabla\cdot F(U))\, V \, dE
# =
# -\int_E F(U) : \nabla V \, dE
# + \int_{\partial E} (F(U)\cdot n_E)\, V \, d\Gamma,
# ```
#
# where $n_E$ is the outward normal to $\partial E$ and $F(U):\nabla V$ denotes the contraction
# (for scalar $V$, this is simply $F(U)\cdot\nabla V$; for vector-valued test functions the contraction
# is done component-wise).
#
# Because $U$ is discontinuous, the boundary term is not uniquely defined on interior facets.
# We replace the physical normal flux $(F(U)\cdot n)$ by a **numerical flux**
# $\widehat{F}(U^-,U^+,n)$ which depends on both traces on the facet.
#
# The DG weak form becomes: for all $V\in\mathbb{W}_p(\mathcal{M}_\Omega)$,
#
# ```math
# \sum_{E\in\mathcal{M}_\Omega}\int_E \partial_t U \, V \, dE
# =
# \sum_{E\in\mathcal{M}_\Omega}\left(
# \int_E S(U)\,V\,dE
# + \int_E F(U):\nabla V\,dE
# \right)
# -
# \sum_{E\in\mathcal{M}_\Omega}\int_{\partial E} \widehat{F}(U^-,U^+,n_E)\, V \, d\Gamma,
# ```
#
# where on boundary facets $U^+$ is replaced by a boundary state (from the boundary condition).
#
# ### From element boundaries to mesh facets
#
# The sum over element boundaries can be rewritten as a sum over facets.
# On an interior facet $F$ shared by $E^-$ and $E^+$, one contributes with the normal $n$ from $E^-$ to $E^+$.
# The interface term then couples the two neighboring elements:
#
# ```math
# \sum_{F\in\mathcal{F}^{\mathrm{int}}_\Omega}
# \int_F
# \widehat{F}(U^-,U^+,n)\,V^- \, d\Gamma
# \;+\;
# \int_F
# \widehat{F}(U^-,U^+,n)\,V^+ \, d\Gamma,
# ```
#
# with opposite signs depending on the convention (in practice, implementations assemble the contribution
# to both elements in one interface loop).
#
# ### Choice of numerical flux: Rusanov (local Lax–Friedrichs)
#
# In this tutorial we use the robust Rusanov flux:
#
# ```math
# \widehat{F}(U^-,U^+,n)
# =
# \tfrac12\left(F(U^-)\cdot n + F(U^+)\cdot n\right)
# -
# \tfrac{c}{2}\left(U^+ - U^-\right),
# ```
#
# where $c$ is an estimate of the maximum wave speed in direction $n$.
# For the shallow water equations one may use
# ```math
# c(U,n) = |u\cdot n| + \sqrt{g h}.
# ```
#
# ### Semi-discrete system
#
# Choose, on each element $E$, a basis $(\Phi_i^E)_{i=1}^r$ of $\mathbb{P}_p(E)$.
# Expanding the discrete solution in that basis yields element degrees of freedom (DOFs).
# Testing successively with each basis function leads to a system of ODEs for the DOFs:
#
# ```math
# M \,\frac{d\mathbf{U}}{dt}
# =
# \mathbf{R}_{\mathrm{vol}}(\mathbf{U})
# +
# \mathbf{R}_{\mathrm{int}}(\mathbf{U})
# +
# \mathbf{R}_{\partial}(\mathbf{U}),
# ```
#
# where:
# - $M$ is the (block) **mass matrix** with entries $M_{ij}^E = \int_E \Phi_i^E \Phi_j^E\,dE$,
# - $\mathbf{R}_{\mathrm{vol}}$ contains the volume contributions $\int_E (S(U)\Phi_i^E + F(U):\nabla\Phi_i^E)\,dE$,
# - $\mathbf{R}_{\mathrm{int}}$ contains interior interface flux contributions,
# - $\mathbf{R}_{\partial}$ contains boundary flux contributions (using boundary states).
#
# This semi-discrete system is then advanced in time with an explicit Runge–Kutta method.

#-
# ## Commented program
#
using Ferrite, SparseArrays
using OrdinaryDiffEqTsit5, LinearSolve
using ConcreteStructs

#-
# ### Physics: bathymetry, flux, source, numerical flux
const g = 9.81

bat(X) = 0.5 * exp(-(X[1] - 5)^2 - (X[2] - 5)^2)
batpx(X) = -2 * (X[1] - 5) * 0.5 * exp(-(X[1] - 5)^2 - (X[2] - 5)^2)
batpy(X) = -2 * (X[2] - 5) * 0.5 * exp(-(X[1] - 5)^2 - (X[2] - 5)^2)

function F(U)
    h, qx, qy = U
    ux = qx / h
    uy = qy / h
    p = g * h * h / 2
    return Vec{3}((qx, qx * ux + p, qx * uy)), Vec{3}((qy, qy * ux, qy * uy + p))
end

function S(U, X)
    h, qx, qy = U
    return Vec{3}((0.0, -g * h * batpx(X), -g * h * batpy(X)))
end

function max_speed(U, n)
    h, qx, qy = U
    h = max(h, 1.0e-12)
    return abs(qx * n[1] + qy * n[2]) / h + sqrt(g * h)
end

function LF(UL, UR, n)
    c = max(max_speed(UL, n), max_speed(UR, n))
    FLx, FLy = F(UL)
    FRx, FRy = F(UR)
    FL = FLx * n[1] + FLy * n[2]
    FR = FRx * n[1] + FRy * n[2]
    return 0.5 * (FL + FR - c * (UR - UL))
end

#-
# ### Initial and boundary conditions
function initial_condition(X)
    return Vec{3}((2 - bat(X), 0.0, 0.0))
end

relu(x) = x > 0 ? x : 0.0

function outflow_riemann(Uin, n; h0 = 2.0, u0 = 0.0)
    h, qx, qy = Uin
    h = max(h, 1.0e-12)
    un = (qx * n[1] + qy * n[2]) / h
    c = sqrt(g * h)

    un >= c && return Uin

    wplus = un + 2c
    wminus = u0 - 2 * sqrt(g * h0)

    unB = (wplus + wminus) / 2
    cB = (wplus - wminus) / 4
    hB = max((cB * cB) / g, 1.0e-12)
    qnB = hB * unB

    t1, t2 = -n[2], n[1]
    qt = qx * t1 + qy * t2

    return Vec{3}((hB, qnB * n[1] + qt * t1, qnB * n[2] + qt * t2))
end

function bound_condition(Uin, X, tag, t, n)
    hin, qxin, qyin = Uin
    x, y = X

    if tag == "left"
        return t < 1 / 5 ?
            Vec{3}((2 + relu(sinpi(5t)) * exp(-(y - 5)^2), qxin, 0.0)) :
            Vec{3}((2.0, qxin, 0.0))
    elseif tag == "right"
        return outflow_riemann(Uin, n)
    else
        return Vec{3}((hin, qxin, -qyin))
    end
end

#-
# ### Mesh and DG space
dim = 2
nx, ny = 64, 64

grid = generate_grid(
    Quadrilateral,
    (nx, ny),
    Vec{dim}((0.0, 0.0)),
    Vec{dim}((10.0, 10.0)),
)

topo = ExclusiveTopology(grid)

∂Ω = union(
    getfacetset(grid, "left"),
    getfacetset(grid, "right"),
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
)

order = 1
ip = DiscontinuousLagrange{RefQuadrilateral, order}()

dh = DofHandler(grid)
add!(dh, :h, ip)
add!(dh, :hu, ip)
add!(dh, :hv, ip)
close!(dh)

nd = ndofs(dh)

qr = QuadratureRule{RefQuadrilateral}(2 * order + 1)
facet_qr = FacetQuadratureRule{RefQuadrilateral}(2 * order + 1)

cv = CellValues(qr, ip)
fv = FacetValues(facet_qr, ip)
iv = InterfaceValues(facet_qr, ip)

#-
# ### Mass matrix (block-diagonal per element)
function assemble_mass_matrix!(M, dh, cv)
    nb = getnbasefunctions(cv)
    ndpc = 3 * nb
    Me = zeros(ndpc, ndpc)

    asM = start_assemble(M)

    for cell in CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        dofs = celldofs(cell)
        fill!(Me, 0.0)

        for qp in 1:getnquadpoints(cv)
            dE = getdetJdV(cv, qp)
            for i in 1:nb
                Φᵢ = shape_value(cv, qp, i)
                for j in 1:nb
                    Φⱼ = shape_value(cv, qp, j)
                    m = (Φᵢ * Φⱼ) * dE
                    Me[i, j] += m
                    Me[i + nb, j + nb] += m
                    Me[i + 2nb, j + 2nb] += m
                end
            end
        end

        assemble!(asM, dofs, Me)
    end
    return M
end

M = allocate_matrix(dh)
assemble_mass_matrix!(M, dh, cv)

Ut = zeros(size(M, 1))
linprob = LinearProblem(M, Ut)

#-
# ### Assemble the residual: volume + interfaces + boundary facets
function volume_integral!(R, U, param, t)
    Re = param.Re
    cv = param.cv
    nb = getnbasefunctions(cv)

    for cell in CellIterator(param.dh)
        Ferrite.reinit!(cv, cell)
        dofs = celldofs(cell)
        Ue = @view U[dofs]
        fill!(Re, 0.0)

        coords = getcoordinates(cell)

        for qp in 1:getnquadpoints(cv)
            dE = getdetJdV(cv, qp)

            h = function_value(cv, qp, Ue, 1:nb)
            qx = function_value(cv, qp, Ue, (nb + 1):2nb)
            qy = function_value(cv, qp, Ue, (2nb + 1):3nb)
            UU = Vec{3}((h, qx, qy))

            Fx, Fy = F(UU)
            X = spatial_coordinate(cv, qp, coords)
            SS = S(UU, X)

            @inbounds for i in 1:nb
                Φᵢ = shape_value(cv, qp, i)
                ∇Φᵢ = shape_gradient(cv, qp, i)

                contrib = (Fx * ∇Φᵢ[1] + Fy * ∇Φᵢ[2] + SS * Φᵢ) * dE

                Re[i] += contrib[1]
                Re[i + nb] += contrib[2]
                Re[i + 2nb] += contrib[3]
            end
        end

        Ferrite.assemble!(R, dofs, Re)
    end
    return R
end

function interface_integral!(R, U, param, t)
    Ri = param.Ri
    iv = param.iv
    nb = getnbasefunctions(param.cv)
    ndpc = 3 * nb

    for ic in param.ii
        Ferrite.reinit!(iv, ic)
        idofs = interfacedofs(ic)
        Ui = @view U[idofs]
        fill!(Ri, 0.0)

        for qp in 1:getnquadpoints(iv)
            n = getnormal(iv, qp; here = true)
            dF = getdetJdV(iv, qp)

            hL = function_value(iv, qp, Ui, 1:nb, (1 + ndpc):(nb + ndpc); here = true)
            qxL = function_value(iv, qp, Ui, (1 + nb):2nb, (1 + nb + ndpc):(2nb + ndpc); here = true)
            qyL = function_value(iv, qp, Ui, (2nb + 1):3nb, (2nb + 1 + ndpc):(3nb + ndpc); here = true)

            hR = function_value(iv, qp, Ui, 1:nb, (1 + ndpc):(nb + ndpc); here = false)
            qxR = function_value(iv, qp, Ui, (1 + nb):2nb, (1 + nb + ndpc):(2nb + ndpc); here = false)
            qyR = function_value(iv, qp, Ui, (2nb + 1):3nb, (2nb + 1 + ndpc):(3nb + ndpc); here = false)

            UL = Vec{3}((hL, qxL, qyL))
            UR = Vec{3}((hR, qxR, qyR))

            fhat = LF(UL, UR, n)

            @inbounds for i in 1:nb
                ΦᵢL = shape_value(iv, qp, i; here = true)
                ΦᵢR = shape_value(iv, qp, i + nb; here = false)

                Ri[i] += (-ΦᵢL * fhat[1]) * dF
                Ri[i + nb] += (-ΦᵢL * fhat[2]) * dF
                Ri[i + 2nb] += (-ΦᵢL * fhat[3]) * dF

                Ri[i + ndpc] += (ΦᵢR * fhat[1]) * dF
                Ri[i + nb + ndpc] += (ΦᵢR * fhat[2]) * dF
                Ri[i + 2nb + ndpc] += (ΦᵢR * fhat[3]) * dF
            end
        end

        Ferrite.assemble!(R, idofs, Ri)
    end
    return R
end

function boundary_facets!(R, U, param, t)
    fv = param.fv
    nb = getnbasefunctions(param.cv)
    Rb = param.Re

    facetsets = param.dh.grid.facetsets

    for tag in ("left", "right", "top", "bottom")
        set = facetsets[tag]
        for fc in FacetIterator(param.dh, set)
            Ferrite.reinit!(fv, fc)
            dofs = celldofs(fc)
            @views u_e = U[dofs]
            coords = Ferrite.getcoordinates(fc)

            fill!(Rb, 0.0)

            for qp in 1:getnquadpoints(fv)
                n = getnormal(fv, qp)
                dF = getdetJdV(fv, qp)

                h = function_value(fv, qp, u_e, 1:nb)
                qx = function_value(fv, qp, u_e, (1 + nb):2nb)
                qy = function_value(fv, qp, u_e, (1 + 2nb):3nb)

                Uin = Vec{3}((h, qx, qy))
                X = spatial_coordinate(fv, qp, coords)

                Ubc = bound_condition(Uin, X, tag, t, n)
                fhat = LF(Uin, Ubc, n)

                @inbounds for i in 1:nb
                    Φᵢ = shape_value(fv, qp, i)
                    Rb[i] -= (Φᵢ * fhat[1]) * dF
                    Rb[i + nb] -= (Φᵢ * fhat[2]) * dF
                    Rb[i + 2nb] -= (Φᵢ * fhat[3]) * dF
                end
            end

            Ferrite.assemble!(R, dofs, Rb)
        end
    end
    return R
end

#-
# ### Time derivative: `dU = M^{-1} R(U)`
function ode!(dU, U, param, t)
    fill!(dU, 0.0)
    volume_integral!(dU, U, param, t)
    interface_integral!(dU, U, param, t)
    boundary_facets!(dU, U, param, t)

    lin = param.lin
    lin.b = dU
    copyto!(dU, solve!(lin).u)
    return nothing
end

#-
# ### Parameters container
@concrete struct Param
    dh
    cv
    fv
    topo
    iv
    ∂Ω
    lin
    Re
    Ri
    ii
end

function Param(dh, cv, fv, topo, iv, ∂Ω, prob)
    nb = Ferrite.getnbasefunctions(cv)
    ndpc = 3 * nb
    return Param(
        dh,
        cv,
        fv,
        topo,
        iv,
        ∂Ω,
        LinearSolve.init(prob),
        zeros(ndpc),
        zeros(2 * ndpc),
        InterfaceIterator(dh, topo),
    )
end

param = Param(dh, cv, fv, topo, iv, ∂Ω, linprob);

#-
# ### Project initial condition and solve
tspan = (0.0, 3.0)

U0 = zeros(nd)
apply_analytical!(U0, dh, :h, x -> initial_condition(x)[1])
apply_analytical!(U0, dh, :hu, x -> initial_condition(x[2]))
apply_analytical!(U0, dh, :hv, x -> initial_condition(x[3]))

prob = ODEProblem(ode!, U0, tspan, param)
sol = solve(prob, Tsit5(); saveat = (tspan[2] - tspan[1]) / 100)

#-
# ### Export to VTK (frames for a GIF)
isdir("datas") || mkdir("datas")
for i in eachindex(sol.u)
    ui = sol.u[i]
    VTKGridFile("./datas/swe_$(i)", dh) do vtk
        write_solution(vtk, dh, ui)   # writes :h, :hu, :hv
    end
end

# **Further important ingredients (not covered here).** For realistic shallow-water simulations, one typically adds: (i) a **positivity / slope limiter** to prevent non-physical negative water depths and to control oscillations near sharp gradients; (ii) a dedicated treatment of **dry states (wetting–drying / desiccation)** to handle cells where `h → 0` robustly (e.g. thresholds, modified fluxes, and consistent momentum handling); (iii) an **automatic CFL timestep selection**, e.g.
# ```math
# \Delta t = \mathrm{CFL}\;\min_{K}\frac{h_K}{(2p+1)\,\max_{\partial K} c(U,n)},\qquad c(U,n)=|u\cdot n|+\sqrt{g h},
# ```

# where `h_K` is a characteristic element size and `p` the DG polynomial degree; and (iv) a **hydrostatic reconstruction** to preserve the lake-at-rest steady state and improve balance between fluxes and the bathymetry source term.

#-
# ## [Plain program](@id tutorial-swe-dg-plain-program)
#
# ```julia
# @__CODE__
# ```
