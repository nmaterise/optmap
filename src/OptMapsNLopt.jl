module OptMapsNLopt

using DocStringExtensions
using Random
using Reexport
using Base.Iterators
using ColorSchemes
using ColorTypes
using Images
using Luxor
using CSV
using DataFrames
@reexport using MultivariateStats
@reexport using NLopt
@reexport using JuMP

export OptMapNLopt, create_map!, draw_solution!, save_all, save_map,
       save_base_img

"""
$(TYPEDEF)

Defines the optimizer structure for NLopt solved problems

$(TYPEDFIELDS)

"""
mutable struct OptMapNLopt
    variable_names::Array{String,1}
    variable_domains::Array{Tuple{R,R},1} where R <: Number
    constraint_names::Array{String,1}
    objective_f::Function
    objective_f_grad
    objective_f_args
    constraints_f
    constraints_f_grads
    constraints_f_args
    sample_points_budget::Int
    verbose::Bool
    min_2d::AbstractArray{Float64,1}
    max_2d::AbstractArray{Float64,1}
    X_nd::AbstractArray{Float64,2}
    C::AbstractArray{Float64,2}
    Z::AbstractArray{Float64,2}
    M::Array{RGB{Normed{UInt8,8}},2}
    S::Array{RGB{Normed{UInt8,8}},2}
    pca #::MultivariateStats.PCA{Float64}
    MAP_SIDE::Int64
end


"""
$(SIGNATURES)

Constructor for the OptMapNLopt structure

Arguments:
=========

variable_names::Array{String, 1}:               array of strings naming the
                                                variables
variable_domains::Array{Tuple{R, R}, 1}         [[a, b]] for all variables        
where R <: Numeber:                     
objective_f::Function:                          objective function
constraint_names:                               [(x) -> true] 
sample_points_budget:                           number points to sample for each
                                                variable
verbose:                                        generate more or less verbose
                                                information

Returns:
=======

O::OptMapNLopt:                                 struct object for the
                                                optimization map

"""
function OptMapNLopt(variable_names::Array{String,1},
                     variable_domains::Array{Tuple{R,R},1} where R <: Number,
                     objective_f::Function;
                     objective_f_grad=nothing,
                     objective_f_args=nothing,
                     constraint_names::Array{String,1}=["none"],
                     constraints_f=[(x) -> true],
                     constraints_f_grads=nothing,
                     constraints_f_args=nothing,
                     sample_points_budget=1000,
                     verbose=false, MAP_SIDE=800)::OptMapNLopt

    O = OptMapNLopt( variable_names,
                variable_domains,
                constraint_names,
                objective_f,
                objective_f_grad,
                objective_f_args,
                constraints_f,
                constraints_f_grads,
                constraints_f_args,
                sample_points_budget,
                verbose,
                zeros(1),
                zeros(1),
                zeros(1,1),
                zeros(1,1),
                zeros(1,1),
                zeros(RGB{Normed{UInt8, 8}}, MAP_SIDE, MAP_SIDE),
                zeros(RGB{Normed{UInt8, 8}}, MAP_SIDE, MAP_SIDE),
                nothing,
                MAP_SIDE)
    return O
end

"""
$(SIGNATURES)

Updates the optimization object as generates the heat map object

Arguments:
=========
O::OptMapNLopt:                             OptMapNLopt object
constraint_brightness::Float64=0.7:         setting the contrast factor in
                                            the brightness of the heatmap
                                            based on the constraint
                                            satisfiability
colormap=:viridi:                           colormap used for the heat map

Returns:
=======

"""
function create_map!(O::OptMapNLopt, constraint_brightness::Float64=0.7;
                     colormap=:viridis)
    # Generate the color map with the user-supplied colormap
    cmap = colorschemes[colormap]

    # Call the fit!() function to update the OptMapNLopt object
    fit!(O)

    # Extract the constraints and the arguments
    n_constraints = length(O.constraints_f)
    cargs         = O.constraints_f_args
    cgrads        = O.constraints_f_grads

    # Default behavior to generate the vectors of nothings
    if cargs === nothing
        args = [nothing for _ in 1:length(O.variable_domains)]
    end
    if cgrads === nothing
        cgrads = [nothing for _ in 1:length(O.variable_domains)]
    end

    # Extract the function arguments
    fargs = O.objective_f_args
    fgrad = O.objective_f_grad
    
    O.verbose && println("Creating OptMapNLopt")

    Z = zeros(Float64, O.MAP_SIDE, O.MAP_SIDE)
    A = ones(Float64, O.MAP_SIDE, O.MAP_SIDE)
    C = ones(Float64, O.MAP_SIDE^2, n_constraints)

    @inbounds for i in 1:O.MAP_SIDE
        @inbounds for j in 1:O.MAP_SIDE
            k = ((i - 1) * O.MAP_SIDE + j)

            @inbounds for c in 1:n_constraints
                if (cgrads[c] != nothing) && (cargs[c] != nothing)
                    C[k,c] = O.constraints_f[c](O.X_nd[:,k], cgrads[c],
                                                cargs[c]...)
                else
                    C[k,c] = O.constraints_f[c](O.X_nd[:,k])
                end
            end

            is_feasible_point = *(C[k,:]...) # product of all constraints
            if (fgrad != nothing) && (fargs != nothing)
                Z[i,j] = O.objective_f(O.X_nd[:,k], fgrad, fargs...)
            else
                Z[i,j] = O.objective_f(O.X_nd[:,k])
            end
            A[i,j] = Bool(is_feasible_point) ? 1.0 : constraint_brightness
        end
    end

    M = get(cmap, Z, (minimum(Z), maximum(Z)))
    M = RGB.(M .* A)
    M = convert(Array{RGB{N0f8}}, M)

    O.Z = Z
    O.M = M
    O.C = C
end


"""
$(SIGNATURES)

Draws the solution path from start to end vector projected onto the principal
axes from the ND space

Arguments:
=========

O::OptMapNLopt:
path_to_solution::AbstractArray{Float64,2}:
point_radius::Int=50:
line_width::Float64=3.0:

"""
function draw_solution!(O::OptMapNLopt,
        path_to_solution::AbstractArray{Float64,2}; point_radius::Int=50,
        line_width::Float64=3.0)
    O.verbose && println("Drawing OptMapNLopt")

    path_2d_tmp = MultivariateStats.transform(O.pca, path_to_solution)
    path_2d = zeros(Float64, 2, size(path_to_solution, 2))

    @inbounds for i in 1:size(path_to_solution, 2)
        path_2d[:,i] = coords_to_pixels(path_2d_tmp[:,i], O.min_2d, O.max_2d;
                                        maxgrid=O.MAP_SIDE)
    end
    
    tmp_map_name = tempname() * ".png"
    save(tmp_map_name, O.M)
    img = readpng(tmp_map_name)

    w = img.width
    h = img.height
    
    Drawing(w, h, tmp_map_name)
    placeimage(img, 0, 0, 1.0)
    setline(line_width)

    sethue("white")
    circle(path_2d[1,1], path_2d[2,1], point_radius, :stroke)
    sethue("red")
    circle(path_2d[1,end], path_2d[2,end], point_radius, :stroke)
    
    sethue(1.0, 0.0, 1.0) 
    @inbounds for i in 2:size(path_2d, 2)
        p1 = Point((path_2d[:,i-1])...)
        p2 = Point((path_2d[:,i])...)

        circle(p1, line_width*2, :fill)
        circle(p2, line_width*2, :fill)
        line(p1, p2, :stroke)
    end

    finish()

    O.S = load(tmp_map_name)
    rm(tmp_map_name)
end

function save_base_img(O::OptMapNLopt, file_name::String)
    O.verbose && println("Saving image to $file_name")
    save(file_name, O.M)
end

function save_map(O::OptMapNLopt, file_prefix::String, path::String=".")
    map_name = "$(path)/$(file_prefix)_optmap.png"

    O.verbose && println("Saving map to $map_name")
    save(map_name, O.S)
end

function save_all(O::OptMapNLopt, file_prefix::String, path::String=".")
    save_map(O, file_prefix, path)

    X_csv = "$(path)/$(file_prefix)_optmap_X.csv"
    C_csv = "$(path)/$(file_prefix)_optmap_C.csv"
    Z_csv = "$(path)/$(file_prefix)_optmap_Z.csv"

    # O.verbose && println("Saving X_nd to $X_csv")
    CSV.write(X_csv, DataFrame(Matrix(O.X_nd'), :auto), header=O.variable_names)
    
    O.verbose && println("Saving C to $C_csv")
    CSV.write(C_csv, DataFrame(O.C, :auto), header=O.constraint_names)

    O.verbose && println("Saving Z to $Z_csv")
    CSV.write(Z_csv, DataFrame(O.Z, :auto), header=false)
end

function fit!(O::OptMapNLopt)
    O.verbose && println("Sampling nD points")
    X = sample_nd_points(O.variable_domains, O.sample_points_budget)

    O.verbose && println("Projecting nD -> 2D ($(size(X)))")
    X_2d, p = project_2d(X)

    O.verbose && println("Unprojecting 2D point grid")
    X_nd = inverse_project(X_2d, p, O.MAP_SIDE)

    O.pca = p
    O.X_nd = X_nd

    O.min_2d = [minimum(X_2d[1,:]), minimum(X_2d[2,:])]
    O.max_2d = [maximum(X_2d[1,:]), maximum(X_2d[2,:])]
end

function sample_nd_points(variables_domains::Array{Tuple{R,R},1} where R <:
        Number, point_budget::Int)::AbstractArray{Float64,2}
    n_vars = length(variables_domains)

    max_points = convert(Int, floor(point_budget^(1 / n_vars)))

    variable_ranges = []

    @inbounds for vd in variables_domains
        if typeof(vd[1]) == Float64
            points = max_points
        else
            points = vd[2] - vd[1] + 1
            points = min(points, max_points)
        end
    
        var_range = range(vd[1], vd[2], length=points)
        push!(variable_ranges, var_range)
    end

    tmp = vec(collect(product(variable_ranges...)))
    X = zeros(n_vars, length(tmp))

    @inbounds for j in 1:n_vars
        for i in 1:length(tmp)
            X[j,i] = tmp[i][j]
        end
    end

    return X
end

function project_2d(X::AbstractArray{Float64,2})
    p = fit(PCA, X; maxoutdim=2, method=:svd)
    X_2d = MultivariateStats.transform(p, X)

    return X_2d, p
end

function inverse_project(X_2d::AbstractArray{Float64,2}, p,
                         MAP_SIDE::Int64)::AbstractArray{Float64,2}
    tmp_grid = vec(collect(product(
                            range(minimum(X_2d[1,:]), maximum(X_2d[1,:]),
                                  length=MAP_SIDE), range(minimum(X_2d[2,:]),
                                                          maximum(X_2d[2,:]),
                                                          length=MAP_SIDE))))
    X_2d_grid = zeros(Float64, 2, MAP_SIDE^2)

    @inbounds for i in 1:size(tmp_grid)[1]
        X_2d_grid[1,i] = tmp_grid[i][1]
        X_2d_grid[2,i] = tmp_grid[i][2]
    end
    
    X_nd = reconstruct(p, X_2d_grid)
    return X_nd
end

function coords_to_pixels(point::AbstractArray{Float64,1},
        min_2d::Array{Float64,1}, max_2d::Array{Float64,1}; mingrid=1,
        maxgrid=100)
    p = (point - min_2d) ./ (max_2d - min_2d)
    p_trunc = floor.(p .* (maxgrid - mingrid))

    return p_trunc
end

end
